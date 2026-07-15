# SPDX-License-Identifier: Apache-2.0
"""Adapters between stage objects and data-plane refs."""
from __future__ import annotations

import base64
import io
import pickle
from dataclasses import fields, is_dataclass
from multiprocessing.reduction import ForkingPickler
from typing import Any

import torch

from sglang_omni.comm.data_ref import (
    BackendRef,
    DataKind,
    DataLayout,
    DataRef,
    MetadataTensorRef,
    TensorMeta,
    TransportKind,
)
from sglang_omni.proto import DataReadyMessage, StagePayload
from sglang_omni.relay.base import Relay

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "torch.bool": torch.bool,
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.complex64": torch.complex64,
    "torch.complex128": torch.complex128,
}

_DIRECT_CUDA_IPC_STREAM_CHUNK_TYPE = "TorchCudaIpcStreamChunk"
_DIRECT_CUDA_IPC_PAYLOAD_TYPE = "TorchCudaIpcPayload"
_DIRECT_CUDA_IPC_STREAM_INLINE_BYTES_LIMIT = 64 * 1024


def relay_device(relay: Relay) -> str:
    device = relay.device
    if not isinstance(device, str):
        raise TypeError(
            f"{type(relay).__name__}.device must be str, got "
            f"{type(device).__name__}"
        )
    return device


def extract_tensors(obj: Any, path: str = "") -> tuple[Any, dict[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        return {
            "_tensor_placeholder": path,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }, {path: obj}
    if isinstance(obj, dict):
        out, tensors = {}, {}
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else key
            out[key], child_tensors = extract_tensors(value, child_path)
            tensors.update(child_tensors)
        return out, tensors
    if isinstance(obj, (list, tuple)):
        out, tensors = [], {}
        for idx, value in enumerate(obj):
            child, child_tensors = extract_tensors(value, f"{path}[{idx}]")
            out.append(child)
            tensors.update(child_tensors)
        return type(obj)(out), tensors
    return obj, {}


def extract_cuda_tensors(
    obj: Any, path: str = ""
) -> tuple[Any, dict[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            return {
                "_tensor_placeholder": path,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
            }, {path: obj}
        return obj, {}
    if isinstance(obj, dict):
        out, tensors = {}, {}
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else key
            out[key], child_tensors = extract_cuda_tensors(value, child_path)
            tensors.update(child_tensors)
        return out, tensors
    if isinstance(obj, (list, tuple)):
        out, tensors = [], {}
        for idx, value in enumerate(obj):
            child, child_tensors = extract_cuda_tensors(value, f"{path}[{idx}]")
            out.append(child)
            tensors.update(child_tensors)
        return type(obj)(out), tensors
    return obj, {}


def restore_tensors(obj: Any, tensors: dict[str, torch.Tensor]) -> Any:
    if isinstance(obj, dict):
        if "_tensor_placeholder" in obj:
            path = obj["_tensor_placeholder"]
            if path not in tensors:
                raise KeyError(f"missing tensor payload for placeholder {path!r}")
            return tensors[path]
        return {key: restore_tensors(value, tensors) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(restore_tensors(value, tensors) for value in obj)
    return obj


def should_use_direct_cuda_ipc_stream_chunk(
    data: Any, metadata: dict[str, Any] | None
) -> bool:
    if not _contains_cuda_tensor(data):
        return False
    if _contains_cpu_tensor(data) or _contains_cpu_tensor(metadata):
        return False
    inline_size = _inline_cpu_pickle_size(data) + _inline_cpu_pickle_size(metadata)
    return inline_size <= _DIRECT_CUDA_IPC_STREAM_INLINE_BYTES_LIMIT


def payload_has_cuda_tensor(payload: Any) -> bool:
    return _contains_cuda_tensor(payload)


def serialize_direct_cuda_ipc_payload(payload: StagePayload) -> dict[str, Any]:
    if not isinstance(payload, StagePayload):
        raise TypeError(
            f"direct CUDA IPC payload requires StagePayload, got "
            f"{type(payload).__name__}"
        )
    if _contains_cuda_tensor(payload.request) or _contains_cpu_tensor(payload.request):
        raise ValueError("direct CUDA IPC payload does not support request tensors")
    data_without_tensors, tensors = extract_cuda_tensors(payload.data)
    if not tensors:
        raise ValueError("direct CUDA IPC payload requires at least one CUDA tensor")
    header = StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=data_without_tensors,
    )
    header_bytes = pickle.dumps(header)
    return {
        "_type": _DIRECT_CUDA_IPC_PAYLOAD_TYPE,
        "version": 1,
        "header": header_bytes,
        "tensors": [
            {"path": path, "tensor_bytes": _ipc_pickle(tensor)}
            for path, tensor in tensors.items()
        ],
    }


def is_direct_cuda_ipc_payload_ref(value: Any) -> bool:
    return (
        isinstance(value, dict) and value.get("_type") == _DIRECT_CUDA_IPC_PAYLOAD_TYPE
    )


def deserialize_direct_cuda_ipc_payload(data_ref: dict[str, Any]) -> StagePayload:
    if data_ref.get("_type") != _DIRECT_CUDA_IPC_PAYLOAD_TYPE:
        raise ValueError("data_ref is not a direct CUDA IPC payload")
    if data_ref.get("version") != 1:
        raise ValueError(
            f"unsupported direct CUDA IPC payload version {data_ref.get('version')!r}"
        )
    header_bytes = data_ref.get("header")
    if not isinstance(header_bytes, bytes):
        raise TypeError(
            "direct CUDA IPC payload header must be bytes, got "
            f"{type(header_bytes).__name__}"
        )
    header = pickle.loads(header_bytes)
    if not isinstance(header, StagePayload):
        raise TypeError(
            "direct CUDA IPC payload header must decode to StagePayload, got "
            f"{type(header).__name__}"
        )
    raw_tensors = data_ref.get("tensors")
    if not isinstance(raw_tensors, list):
        raise TypeError(
            "direct CUDA IPC payload tensors must be list, got "
            f"{type(raw_tensors).__name__}"
        )
    tensors: dict[str, torch.Tensor] = {}
    for item in raw_tensors:
        if not isinstance(item, dict):
            raise TypeError(
                "direct CUDA IPC payload tensor entry must be dict, got "
                f"{type(item).__name__}"
            )
        path = item.get("path")
        if not isinstance(path, str):
            raise TypeError(
                "direct CUDA IPC payload tensor path must be str, got "
                f"{type(path).__name__}"
            )
        tensor_bytes = item.get("tensor_bytes")
        if not isinstance(tensor_bytes, bytes):
            raise TypeError(
                "direct CUDA IPC payload tensor_bytes must be bytes, got "
                f"{type(tensor_bytes).__name__}"
            )
        tensor = pickle.loads(tensor_bytes)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "direct CUDA IPC payload tensor entry must decode to Tensor, got "
                f"{type(tensor).__name__}"
            )
        if not tensor.is_cuda:
            raise ValueError("direct CUDA IPC payload tensor decoded as non-CUDA")
        if path in tensors:
            raise ValueError(f"duplicate direct CUDA IPC tensor path {path!r}")
        tensors[path] = tensor
    return StagePayload(
        request_id=header.request_id,
        request=header.request,
        data=restore_tensors(header.data, tensors),
    )


def serialize_direct_cuda_ipc_stream_chunk(
    data: Any,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if not should_use_direct_cuda_ipc_stream_chunk(data, metadata):
        raise ValueError("same-GPU CUDA stream chunk is not direct-IPC eligible")
    ref: dict[str, Any] = {
        "_type": _DIRECT_CUDA_IPC_STREAM_CHUNK_TYPE,
        "version": 1,
        "tensor_bytes": _ipc_pickle(data),
    }
    if metadata is not None:
        ref["metadata"] = _serialize_direct_ipc_metadata_value(metadata)
    return ref


def is_direct_cuda_ipc_stream_chunk_ref(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("_type") == _DIRECT_CUDA_IPC_STREAM_CHUNK_TYPE
    )


def deserialize_direct_cuda_ipc_stream_chunk(
    data_ref: dict[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    if data_ref.get("_type") != _DIRECT_CUDA_IPC_STREAM_CHUNK_TYPE:
        raise ValueError("data_ref is not a direct CUDA IPC stream chunk")
    if data_ref.get("version") != 1:
        raise ValueError(
            f"unsupported direct CUDA IPC version {data_ref.get('version')!r}"
        )
    tensor_bytes = data_ref.get("tensor_bytes")
    if not isinstance(tensor_bytes, bytes):
        raise TypeError(
            "direct CUDA IPC data_ref tensor_bytes must be bytes, got "
            f"{type(tensor_bytes).__name__}"
        )
    data = pickle.loads(tensor_bytes)
    raw_metadata = data_ref.get("metadata")
    if raw_metadata is None:
        return data, None
    if not isinstance(raw_metadata, dict):
        raise TypeError(
            "direct CUDA IPC metadata must be dict, got "
            f"{type(raw_metadata).__name__}"
        )
    metadata = deserialize_direct_ipc_metadata(raw_metadata)
    if not isinstance(metadata, dict):
        raise TypeError(
            "direct CUDA IPC decoded metadata must be dict, got "
            f"{type(metadata).__name__}"
        )
    return data, metadata


async def write_payload(
    relay: Relay,
    request_id: str,
    payload: StagePayload,
    *,
    transport: TransportKind,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> tuple[DataRef, Any]:
    data_without_tensors, tensors = extract_tensors(payload.data)
    packed, entries = _pack_tensors(tensors, device=relay_device(relay))
    header = StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=data_without_tensors,
    )
    op = await relay.put_async(
        packed,
        request_id=request_id,
        receiver_id=to_stage,
    )
    data_ref = DataRef(
        version=1,
        object_id=f"{request_id}:payload:{from_stage or ''}:{to_stage or ''}",
        kind=DataKind.STAGE_PAYLOAD,
        transport=transport,
        layout=DataLayout.PACKED_TENSORS,
        buffer=BackendRef.from_relay_info(
            transport=transport,
            relay_info=op.metadata,
        ),
        header=base64.b64encode(pickle.dumps(header)).decode("ascii"),
        tensors=tuple(entries),
    )
    return data_ref, op


async def read_payload(
    relay: Relay,
    request_id: str,
    data_ref: DataRef,
) -> StagePayload:
    if data_ref.kind is not DataKind.STAGE_PAYLOAD:
        raise ValueError(f"expected stage_payload, got {data_ref.kind.value}")
    if data_ref.header is None:
        raise ValueError("stage_payload data_ref is missing header")
    header = pickle.loads(base64.b64decode(data_ref.header))
    transfer_buf = await _read_transfer_buffer(relay, request_id, data_ref)
    tensors = {
        entry.path: _restore_tensor_device(
            transfer_buf[entry.offset : entry.offset + entry.size]
            .view(_torch_dtype(entry.dtype))
            .reshape(entry.shape),
            entry.device,
        )
        for entry in data_ref.tensors
    }
    relay.cleanup(request_id)
    return StagePayload(
        request_id=header.request_id,
        request=header.request,
        data=restore_tensors(header.data, tensors),
    )


async def write_tensor(
    relay: Relay,
    object_id: str,
    tensor: torch.Tensor,
    *,
    transport: TransportKind,
    kind: DataKind = DataKind.STREAM_CHUNK,
    request_id: str | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> tuple[DataRef, Any]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"write_tensor requires torch.Tensor, got {type(tensor).__name__}"
        )
    packed = tensor.contiguous().view(torch.uint8).reshape(-1)
    target_device = torch.device(relay_device(relay))
    if packed.device != target_device:
        packed = packed.to(device=target_device)
    offset = _pad_offset(0, _dtype_alignment(tensor.dtype))
    if offset:
        packed = torch.cat(
            [torch.zeros(offset, dtype=torch.uint8, device=target_device), packed]
        )
    op = await relay.put_async(
        packed,
        request_id=object_id,
        receiver_id=to_stage,
    )
    return (
        DataRef(
            version=1,
            object_id=object_id,
            kind=kind,
            transport=transport,
            layout=DataLayout.RAW_TENSOR,
            buffer=BackendRef.from_relay_info(
                transport=transport,
                relay_info=op.metadata,
            ),
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=str(tensor.dtype),
            offset=offset,
        ),
        op,
    )


async def read_tensor(
    relay: Relay,
    data_ref: DataRef,
) -> torch.Tensor:
    if data_ref.layout is not DataLayout.RAW_TENSOR:
        raise ValueError(f"expected raw_tensor layout, got {data_ref.layout.value}")
    if data_ref.shape is None:
        raise ValueError("raw tensor data_ref is missing shape")
    if data_ref.dtype is None:
        raise ValueError("raw tensor data_ref is missing dtype")
    if data_ref.offset is None:
        raise ValueError("raw tensor data_ref is missing offset")
    transfer_buf = await _read_transfer_buffer(relay, data_ref.object_id, data_ref)
    return (
        transfer_buf[data_ref.offset :]
        .view(_torch_dtype(data_ref.dtype))
        .reshape(data_ref.shape)
    )


async def write_stream_chunk(
    relay: Relay,
    *,
    request_id: str,
    data: torch.Tensor,
    target_stage: str,
    from_stage: str,
    chunk_id: int,
    metadata: dict | None = None,
    transport: TransportKind,
) -> tuple[DataRef, list[Any]]:
    object_id = f"{request_id}:stream:{from_stage}:{target_stage}:{chunk_id}"
    data_ref, op = await write_tensor(
        relay,
        object_id,
        data,
        transport=transport,
        kind=DataKind.STREAM_CHUNK,
        request_id=request_id,
        from_stage=from_stage,
        to_stage=target_stage,
    )
    pending_ops = [op]
    data_ref = await _with_stream_metadata(
        relay,
        data_ref,
        metadata,
        transport,
        pending_ops,
        receiver_id=target_stage,
    )
    return data_ref, pending_ops


async def read_stream_chunk(
    relay: Relay,
    data_ref: DataRef,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    data = await read_tensor(relay, data_ref)
    metadata = dict(data_ref.metadata or {})
    if data_ref.metadata_tensors:
        tensors = {
            ref.path: await read_tensor(relay, ref.ref)
            for ref in data_ref.metadata_tensors
        }
        metadata = restore_tensors(metadata, tensors)
    return data, metadata or None


async def send_stream_signal(
    control_plane: Any,
    *,
    request_id: str,
    target_stage: str,
    target_endpoint: str,
    from_stage: str,
    is_done: bool = False,
    error: str | None = None,
) -> None:
    await control_plane.send_to_stage(
        target_stage,
        target_endpoint,
        DataReadyMessage(
            request_id=request_id,
            from_stage=from_stage,
            to_stage=target_stage,
            data_ref=None,
            is_done=is_done,
            error=error,
        ),
    )


async def _with_stream_metadata(
    relay: Relay,
    data_ref: DataRef,
    metadata: dict | None,
    transport: TransportKind,
    pending_ops: list[Any],
    *,
    receiver_id: str | None = None,
) -> DataRef:
    if metadata is None:
        return data_ref
    metadata_without_tensors, tensors = extract_tensors(metadata)
    tensor_refs = []
    for idx, (path, tensor) in enumerate(tensors.items()):
        ref, op = await write_tensor(
            relay,
            f"{data_ref.object_id}:meta:{idx}",
            tensor,
            transport=transport,
            kind=DataKind.STREAM_METADATA_TENSOR,
            to_stage=receiver_id,
        )
        tensor_refs.append(MetadataTensorRef(path=path, ref=ref))
        pending_ops.append(op)
    return DataRef(
        version=data_ref.version,
        object_id=data_ref.object_id,
        kind=data_ref.kind,
        transport=data_ref.transport,
        layout=data_ref.layout,
        buffer=data_ref.buffer,
        shape=data_ref.shape,
        dtype=data_ref.dtype,
        offset=data_ref.offset,
        metadata=metadata_without_tensors,
        metadata_tensors=tuple(tensor_refs),
    )


def _pack_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    device: str,
) -> tuple[torch.Tensor, list[TensorMeta]]:
    target_device = torch.device(device)
    entries, chunks, offset = [], [], 0
    for path, tensor in tensors.items():
        flat = tensor.contiguous().view(torch.uint8).reshape(-1)
        if flat.device != target_device:
            flat = flat.to(device=target_device)
        padding = _pad_offset(offset, _dtype_alignment(tensor.dtype))
        if padding:
            chunks.append(torch.zeros(padding, dtype=torch.uint8, device=target_device))
            offset += padding
        chunks.append(flat)
        entries.append(
            TensorMeta(
                path=path,
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
                offset=offset,
                size=int(flat.numel()),
            )
        )
        offset += int(flat.numel())
    if not chunks:
        chunks.append(torch.zeros(1, dtype=torch.uint8, device=target_device))
    return torch.cat(chunks), entries


async def _read_transfer_buffer(
    relay: Relay,
    request_id: str,
    data_ref: DataRef,
) -> torch.Tensor:
    buf = torch.zeros(
        data_ref.buffer.length,
        dtype=torch.uint8,
        device=relay_device(relay),
    )
    op = await relay.get_async(
        metadata=data_ref.buffer.info,
        dest_tensor=buf,
        request_id=request_id,
    )
    await op.wait_for_completion()
    return buf


def _dtype_alignment(dtype: torch.dtype) -> int:
    return max(torch.empty((), dtype=dtype).element_size(), 1)


def _pad_offset(offset: int, alignment: int) -> int:
    return (-offset) % alignment


def _torch_dtype(dtype_str: str) -> torch.dtype:
    dtype = _TORCH_DTYPES.get(dtype_str)
    if dtype is None:
        raise ValueError(f"unsupported tensor dtype metadata: {dtype_str!r}")
    return dtype


def _restore_tensor_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    if torch.device(device).type == "cpu":
        return tensor.cpu()
    return tensor


def _contains_cuda_tensor(obj: Any, seen: set[int] | None = None) -> bool:
    if obj is None:
        return False
    seen = set() if seen is None else seen
    obj_id = id(obj)
    if obj_id in seen:
        return False
    seen.add(obj_id)
    if isinstance(obj, torch.Tensor):
        return obj.is_cuda
    if isinstance(obj, dict):
        return any(_contains_cuda_tensor(value, seen) for value in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(_contains_cuda_tensor(value, seen) for value in obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return any(
            _contains_cuda_tensor(getattr(obj, field.name), seen)
            for field in fields(obj)
        )
    return False


def _contains_cpu_tensor(obj: Any, seen: set[int] | None = None) -> bool:
    if obj is None:
        return False
    seen = set() if seen is None else seen
    obj_id = id(obj)
    if obj_id in seen:
        return False
    seen.add(obj_id)
    if isinstance(obj, torch.Tensor):
        return not obj.is_cuda
    if isinstance(obj, dict):
        return any(_contains_cpu_tensor(value, seen) for value in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(_contains_cpu_tensor(value, seen) for value in obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return any(
            _contains_cpu_tensor(getattr(obj, field.name), seen)
            for field in fields(obj)
        )
    return False


def _inline_cpu_pickle_size(obj: Any, seen: set[int] | None = None) -> int:
    if obj is None:
        return 0
    seen = set() if seen is None else seen
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, torch.Tensor):
        return 0 if obj.is_cuda else _DIRECT_CUDA_IPC_STREAM_INLINE_BYTES_LIMIT + 1
    if isinstance(obj, dict):
        return sum(
            _inline_cpu_pickle_size(key, seen) + _inline_cpu_pickle_size(value, seen)
            for key, value in obj.items()
        )
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(_inline_cpu_pickle_size(value, seen) for value in obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return sum(
            _inline_cpu_pickle_size(getattr(obj, field.name), seen)
            for field in fields(obj)
        )
    try:
        return len(pickle.dumps(obj))
    except Exception:
        return _DIRECT_CUDA_IPC_STREAM_INLINE_BYTES_LIMIT + 1


def _ipc_pickle(obj: Any) -> bytes:
    if not _contains_cuda_tensor(obj):
        return pickle.dumps(obj)
    buf = io.BytesIO()
    ForkingPickler(buf, 2).dump(obj)
    return buf.getvalue()


def _serialize_direct_ipc_metadata_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"_ipc_tensor": _ipc_pickle(value)}
    if isinstance(value, dict):
        return {
            key: _serialize_direct_ipc_metadata_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_serialize_direct_ipc_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return {
            "_ipc_tuple": [_serialize_direct_ipc_metadata_value(item) for item in value]
        }
    return value


def deserialize_direct_ipc_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"_ipc_tensor"}:
            tensor_bytes = value["_ipc_tensor"]
            if not isinstance(tensor_bytes, bytes):
                raise TypeError("_ipc_tensor metadata must be bytes")
            return pickle.loads(tensor_bytes)
        if set(value) == {"_ipc_tuple"}:
            items = value["_ipc_tuple"]
            if not isinstance(items, list):
                raise TypeError("_ipc_tuple metadata must be list")
            return tuple(deserialize_direct_ipc_metadata(item) for item in items)
        return {
            key: deserialize_direct_ipc_metadata(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [deserialize_direct_ipc_metadata(item) for item in value]
    return value
