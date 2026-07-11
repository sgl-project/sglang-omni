# SPDX-License-Identifier: Apache-2.0
"""Relay IO utilities for inter-stage data transfer.

Handles payload serialization (tensor extraction/restoration), relay read/write,
streaming chunk transfer, and NIXL credit deadlock avoidance.

Extracted from worker/data_plane.py and worker/runtime.py.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import pickle
from multiprocessing.reduction import ForkingPickler
from typing import Any
from uuid import uuid4

import torch

from sglang_omni.pipeline.tensor_ref import (
    TensorRef,
    TensorRefPolicy,
    is_tensor_ref_dict,
)
from sglang_omni.proto import DataReadyMessage, StagePayload
from sglang_omni.relay.base import Relay

logger = logging.getLogger(__name__)


def _dtype_alignment(dtype: torch.dtype) -> int:
    return max(torch.empty((), dtype=dtype).element_size(), 1)


def _pad_offset(offset: int, alignment: int) -> int:
    return (-offset) % alignment


def _dtype_from_str(dtype_str: str) -> torch.dtype:
    return getattr(torch, dtype_str.replace("torch.", ""))


# ---------------------------------------------------------------------------
# Tensor extraction / restoration (recursive, nested dicts/lists)
# ---------------------------------------------------------------------------


def extract_tensors(obj: Any, path: str = "") -> tuple[Any, dict[str, torch.Tensor]]:
    """Recursively extract tensors from nested structure, replacing with placeholders."""
    tensors = {}

    if isinstance(obj, torch.Tensor):
        placeholder = {
            "_tensor_placeholder": path,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }
        tensors[path] = obj
        return placeholder, tensors

    elif isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            new_value, sub_tensors = extract_tensors(value, new_path)
            new_dict[key] = new_value
            tensors.update(sub_tensors)
        return new_dict, tensors

    elif isinstance(obj, (list, tuple)):
        new_list = []
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]"
            new_item, sub_tensors = extract_tensors(item, new_path)
            new_list.append(new_item)
            tensors.update(sub_tensors)
        return (type(obj)(new_list), tensors)

    else:
        return obj, tensors


def restore_tensors(obj: Any, tensor_dict: dict[str, torch.Tensor]) -> Any:
    """Recursively restore tensors from placeholders."""
    if isinstance(obj, dict):
        if "_tensor_placeholder" in obj:
            path = obj["_tensor_placeholder"]
            return tensor_dict.get(path)
        else:
            return {
                key: restore_tensors(value, tensor_dict) for key, value in obj.items()
            }
    elif isinstance(obj, (list, tuple)):
        return type(obj)(restore_tensors(item, tensor_dict) for item in obj)
    else:
        return obj


_BACKGROUND_REF_TASKS: set[asyncio.Task] = set()
_LOGGED_REF_EDGES: set[tuple[str, str]] = set()


def collect_tensor_refs(obj: Any, seen: set[int] | None = None) -> list[TensorRef]:
    """Collect unresolved TensorRef leaves from a nested payload."""
    if obj is None:
        return []
    seen = set() if seen is None else seen
    obj_id = id(obj)
    if obj_id in seen:
        return []
    seen.add(obj_id)

    if is_tensor_ref_dict(obj):
        return [TensorRef.from_dict(obj)]
    if isinstance(obj, dict):
        refs: list[TensorRef] = []
        for value in obj.values():
            refs.extend(collect_tensor_refs(value, seen))
        return refs
    if isinstance(obj, (list, tuple, set, frozenset)):
        refs = []
        for value in obj:
            refs.extend(collect_tensor_refs(value, seen))
        return refs
    return []


def _release_shm_transfer(relay_info: dict[str, Any]) -> bool:
    transfer_info = relay_info.get("transfer_info", {})
    shm_name = (
        transfer_info.get("shm_name") if isinstance(transfer_info, dict) else None
    )
    if not isinstance(shm_name, str) or not shm_name:
        return False

    from multiprocessing import shared_memory as _shm

    try:
        shm = _shm.SharedMemory(name=shm_name)
    except FileNotFoundError:
        return False
    try:
        shm.unlink()
        return True
    except FileNotFoundError:
        return False
    finally:
        shm.close()


def release_blob(relay: Relay, key: str, metadata: dict[str, Any]) -> bool:
    """Best-effort release for a raw relay blob that will never be read."""
    release_fn = getattr(relay, "release_blob", None)
    if release_fn is not None:
        release_fn(key, metadata)
        return True

    relay_info = metadata.get("relay_info", {})
    if isinstance(relay_info, dict) and _release_shm_transfer(relay_info):
        return True

    storage = getattr(relay, "storage", None)
    if isinstance(storage, dict) and key in storage:
        storage.pop(key, None)
        return True

    return False


def release_tensor_refs(relay: Relay, refs: list[TensorRef]) -> int:
    """Release unresolved TensorRef blobs for requests that abort before consume."""
    released = 0
    seen: set[str] = set()
    for ref in refs:
        if ref.blob_key in seen:
            continue
        seen.add(ref.blob_key)
        if release_blob(relay, ref.blob_key, ref.blob_metadata):
            released += 1
    return released


def release_tensor_ref_blobs_from_metadata(
    relay: Relay, metadata: dict[str, Any]
) -> int:
    if not isinstance(metadata, dict):
        return 0
    raw_refs = metadata.get("tensor_ref_blobs", [])
    if not isinstance(raw_refs, list):
        return 0
    refs = [TensorRef.from_dict(item) for item in raw_refs if is_tensor_ref_dict(item)]
    return release_tensor_refs(relay, refs)


def _track_background_op(op: Any, ref_id: str) -> None:
    async def _wait() -> None:
        try:
            await op.wait_for_completion()
        except Exception:
            logger.warning(
                "tensor_ref relay op failed or timed out for %s", ref_id, exc_info=True
            )

    task = asyncio.create_task(_wait())
    _BACKGROUND_REF_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_REF_TASKS.discard)


async def publish_tensor_ref(
    relay: Relay,
    *,
    request_id: str,
    tensor: torch.Tensor,
    path: str,
    from_stage: str,
    to_stage: str,
    consumer_stage: str,
) -> TensorRef:
    """Externalize ``tensor`` to the relay and return a small reference to it.

    The blob write is awaited (the bytes must be copied before this hop's
    small envelope is sent), but its relay op is tracked in the background —
    the consumer stage, not this hop, is the one that actually reads it.
    """
    edge = (from_stage, to_stage)
    if edge not in _LOGGED_REF_EDGES:
        _LOGGED_REF_EDGES.add(edge)
        logger.info(
            f"tensor_ref active: externalizing {path} "
            f"({tensor.numel() * tensor.element_size() / 1024**2:.1f} MiB) "
            f"on edge {from_stage}->{to_stage}, consumer={consumer_stage}"
        )
    blob_key = f"{request_id}:tensor_ref:{from_stage}:{to_stage}:{uuid4().hex}:{path}"
    blob_metadata, op = await write_blob(relay, blob_key, tensor)
    ref = TensorRef(
        ref_id=blob_key,
        request_id=request_id,
        producer_stage=from_stage,
        consumer_stage=consumer_stage,
        path=path,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        nbytes=tensor.numel() * tensor.element_size(),
        blob_key=blob_key,
        blob_metadata=blob_metadata,
    )
    _track_background_op(op, ref.ref_id)
    return ref


async def extract_tensors_for_payload(
    obj: Any,
    *,
    relay: Relay,
    request_id: str,
    from_stage: str | None,
    to_stage: str | None,
    tensor_ref_policy: TensorRefPolicy,
    stats: dict[str, int],
    path: str = "",
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Like ``extract_tensors``, but allowlisted large tensors are published
    as ``TensorRef`` dicts instead of being flattened into the relay buffer.
    Leaves that are already a ``TensorRef`` dict are passed through as-is.
    """
    tensors: dict[str, torch.Tensor] = {}

    if is_tensor_ref_dict(obj):
        return obj, tensors

    if isinstance(obj, torch.Tensor):
        if tensor_ref_policy.should_externalize(path, obj):
            ref = await publish_tensor_ref(
                relay,
                request_id=request_id,
                tensor=obj,
                path=path,
                from_stage=from_stage or "",
                to_stage=to_stage or "",
                consumer_stage=tensor_ref_policy.consumer_stage,
            )
            stats["ref_count"] += 1
            stats["ref_bytes"] += ref.nbytes
            return ref.to_dict(), tensors

        stats["inline_tensor_bytes"] += obj.numel() * obj.element_size()
        placeholder = {
            "_tensor_placeholder": path,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }
        tensors[path] = obj
        return placeholder, tensors

    elif isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            new_value, sub_tensors = await extract_tensors_for_payload(
                value,
                relay=relay,
                request_id=request_id,
                from_stage=from_stage,
                to_stage=to_stage,
                tensor_ref_policy=tensor_ref_policy,
                stats=stats,
                path=new_path,
            )
            new_dict[key] = new_value
            tensors.update(sub_tensors)
        return new_dict, tensors

    elif isinstance(obj, (list, tuple)):
        new_list = []
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]"
            new_item, sub_tensors = await extract_tensors_for_payload(
                item,
                relay=relay,
                request_id=request_id,
                from_stage=from_stage,
                to_stage=to_stage,
                tensor_ref_policy=tensor_ref_policy,
                stats=stats,
                path=new_path,
            )
            new_list.append(new_item)
            tensors.update(sub_tensors)
        return (type(obj)(new_list), tensors)

    else:
        return obj, tensors


async def read_tensor_ref(relay: Relay, ref: TensorRef) -> torch.Tensor:
    tensor = await read_blob(relay, ref.blob_key, ref.blob_metadata)
    if tuple(tensor.shape) != ref.shape:
        raise RuntimeError(
            f"tensor_ref {ref.ref_id} shape mismatch: expected {ref.shape}, "
            f"got {tuple(tensor.shape)}"
        )
    expected_dtype = _dtype_from_str(ref.dtype)
    if tensor.dtype != expected_dtype:
        raise RuntimeError(
            f"tensor_ref {ref.ref_id} dtype mismatch: expected {ref.dtype}, "
            f"got {tensor.dtype}"
        )
    return tensor


async def materialize_tensor_refs(
    relay: Relay,
    obj: Any,
    *,
    current_stage: str,
    materialize_all: bool = False,
) -> Any:
    """Recursively resolve ``TensorRef`` leaves whose ``consumer_stage``
    matches ``current_stage`` (or all of them, if ``materialize_all``).
    Refs belonging to a different consumer stage pass through unresolved.
    """
    if is_tensor_ref_dict(obj):
        ref = TensorRef.from_dict(obj)
        if materialize_all or ref.consumer_stage == current_stage:
            return await read_tensor_ref(relay, ref)
        return obj

    # note (luojiaxuan): Rebuild containers only when a descendant ref resolves;
    # ref-free and non-consumer payloads keep object identity
    # to skip per-request container churn.
    if isinstance(obj, dict):
        new_dict = {}
        changed = False
        for key, value in obj.items():
            new_value = await materialize_tensor_refs(
                relay,
                value,
                current_stage=current_stage,
                materialize_all=materialize_all,
            )
            changed = changed or new_value is not value
            new_dict[key] = new_value
        return new_dict if changed else obj

    if isinstance(obj, (list, tuple)):
        new_items = []
        changed = False
        for item in obj:
            new_item = await materialize_tensor_refs(
                relay,
                item,
                current_stage=current_stage,
                materialize_all=materialize_all,
            )
            changed = changed or new_item is not item
            new_items.append(new_item)
        return type(obj)(new_items) if changed else obj

    return obj


async def materialize_payload_tensor_refs(
    relay: Relay,
    payload: StagePayload,
    *,
    current_stage: str,
) -> StagePayload:
    new_data = await materialize_tensor_refs(
        relay, payload.data, current_stage=current_stage
    )
    if new_data is payload.data:
        return payload
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=new_data,
    )


# ---------------------------------------------------------------------------
# Payload read/write (full StagePayload via relay)
# ---------------------------------------------------------------------------


async def write_payload(
    relay: Relay,
    request_id: str,
    payload: StagePayload,
    *,
    from_stage: str | None = None,
    to_stage: str | None = None,
    tensor_ref_policy: TensorRefPolicy | None = None,
) -> tuple[dict[str, Any], Any]:
    """Write a StagePayload to relay. Returns (control_plane_metadata, relay_op)."""
    device = getattr(relay, "device", "cpu")
    transport_device = torch.device(device)

    stats: dict[str, int] | None = None
    if tensor_ref_policy is not None:
        stats = {"ref_count": 0, "ref_bytes": 0, "inline_tensor_bytes": 0}
        modified_data, tensor_dict = await extract_tensors_for_payload(
            payload.data,
            relay=relay,
            request_id=request_id,
            from_stage=from_stage,
            to_stage=to_stage,
            tensor_ref_policy=tensor_ref_policy,
            stats=stats,
        )
    else:
        modified_data, tensor_dict = extract_tensors(payload.data)

    payload_no_tensors = StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=modified_data,
    )
    metadata_bytes = pickle.dumps(payload_no_tensors)

    if tensor_dict:
        tensor_buffers = []
        tensor_info = []
        offset = 0
        for path, tensor in tensor_dict.items():
            flat = tensor.contiguous().view(torch.uint8).reshape(-1)
            if flat.device != transport_device:
                flat = flat.to(device=transport_device)
            padding = _pad_offset(offset, _dtype_alignment(tensor.dtype))
            if padding:
                tensor_buffers.append(
                    torch.zeros(padding, dtype=torch.uint8, device=transport_device)
                )
                offset += padding
            tensor_buffers.append(flat)
            tensor_info.append(
                {
                    "path": path,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "offset": offset,
                    "size": flat.numel(),
                }
            )
            offset += flat.numel()
        all_tensors = torch.cat(tensor_buffers)
    else:
        all_tensors = torch.zeros(1, dtype=torch.uint8, device=device)
        tensor_info = []

    op = await relay.put_async(all_tensors, request_id=request_id)

    metadata: dict[str, Any] = {
        "relay_info": op.metadata,
        "payload_pickle": base64.b64encode(metadata_bytes).decode("ascii"),
        "tensor_info": tensor_info,
    }
    tensor_ref_blobs = collect_tensor_refs(modified_data)
    if tensor_ref_blobs:
        metadata["tensor_ref_blobs"] = [ref.to_dict() for ref in tensor_ref_blobs]
    if stats is not None:
        metadata["tensor_ref_stats"] = stats
    return metadata, op


async def read_payload(
    relay: Relay,
    request_id: str,
    metadata: dict[str, Any],
) -> StagePayload:
    """Read a StagePayload from relay using control_plane metadata."""
    device = getattr(relay, "device", "cpu")

    payload_bytes = base64.b64decode(metadata["payload_pickle"])
    payload_no_tensors = pickle.loads(payload_bytes)

    relay_info = metadata["relay_info"]
    tensor_info = metadata.get("tensor_info", [])
    tensor_dict = {}

    data_size = relay_info["transfer_info"]["size"]
    recv_tensor = torch.zeros(data_size, dtype=torch.uint8, device=device)
    op = await relay.get_async(
        metadata=relay_info, dest_tensor=recv_tensor, request_id=request_id
    )
    await op.wait_for_completion()

    if tensor_info:
        for info in tensor_info:
            path = info["path"]
            shape = info["shape"]
            dtype_str = info["dtype"]
            offset = info["offset"]
            size = info["size"]
            tensor_bytes = recv_tensor[offset : offset + size]
            dtype = _dtype_from_str(dtype_str)
            tensor = tensor_bytes.view(dtype).reshape(shape)
            tensor_dict[path] = tensor

    restored_data = restore_tensors(payload_no_tensors.data, tensor_dict)
    payload = StagePayload(
        request_id=payload_no_tensors.request_id,
        request=payload_no_tensors.request,
        data=restored_data,
    )
    relay.cleanup(request_id)
    return payload


# ---------------------------------------------------------------------------
# Blob read/write (raw tensor via relay, for streaming chunks)
# ---------------------------------------------------------------------------


async def write_blob(
    relay: Relay,
    key: str,
    tensor: torch.Tensor,
) -> tuple[dict[str, Any], Any]:
    """Write a raw tensor to relay. Returns (metadata, relay_op)."""
    flat = tensor.contiguous().view(torch.uint8).reshape(-1)
    transport_device = torch.device(getattr(relay, "device", "cpu"))
    if flat.device != transport_device:
        flat = flat.to(device=transport_device)
    padding = _pad_offset(0, _dtype_alignment(tensor.dtype))
    if padding:
        flat = torch.cat(
            [
                torch.zeros(padding, dtype=torch.uint8, device=transport_device),
                flat,
            ]
        )
    op = await relay.put_async(flat, request_id=key)
    metadata = {
        "relay_info": op.metadata,
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_offset": padding,
    }
    return metadata, op


async def read_blob(
    relay: Relay,
    key: str,
    metadata: dict[str, Any],
) -> torch.Tensor:
    """Read a raw tensor from relay."""
    device = getattr(relay, "device", "cpu")
    relay_info = metadata["relay_info"]
    shape = metadata["tensor_shape"]
    dtype_str = metadata["tensor_dtype"]
    offset = int(metadata.get("tensor_offset", 0))

    data_size = relay_info["transfer_info"]["size"]
    recv_buf = torch.zeros(data_size, dtype=torch.uint8, device=device)
    op = await relay.get_async(
        metadata=relay_info, dest_tensor=recv_buf, request_id=key
    )
    await op.wait_for_completion()

    dtype = _dtype_from_str(dtype_str)
    return recv_buf[offset:].view(dtype).reshape(shape)


# ---------------------------------------------------------------------------
# Stream chunk send
# ---------------------------------------------------------------------------

_IPC_INLINE_CPU_BYTES_LIMIT = 64 * 1024


def _is_cuda_tensor(obj: Any) -> bool:
    return isinstance(obj, torch.Tensor) and obj.is_cuda


def _contains_cuda_tensor(obj: Any) -> bool:
    if _is_cuda_tensor(obj):
        return True
    if isinstance(obj, torch.Tensor):
        return False
    if isinstance(obj, dict):
        return any(_contains_cuda_tensor(value) for value in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(_contains_cuda_tensor(value) for value in obj)
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
        return not _is_cuda_tensor(obj)
    if isinstance(obj, dict):
        return any(_contains_cpu_tensor(value, seen) for value in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(_contains_cpu_tensor(value, seen) for value in obj)
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
        return 0
    if isinstance(obj, dict):
        return sum(
            _inline_cpu_pickle_size(key, seen) + _inline_cpu_pickle_size(value, seen)
            for key, value in obj.items()
        )
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(_inline_cpu_pickle_size(value, seen) for value in obj)

    try:
        return len(pickle.dumps(obj))
    except Exception:
        return _IPC_INLINE_CPU_BYTES_LIMIT + 1


def _should_use_cuda_ipc_stream_chunk(data: Any, metadata: dict | None) -> bool:
    if not _contains_cuda_tensor(data):
        return False
    if _contains_cpu_tensor(data) or _contains_cpu_tensor(metadata):
        return False
    inline_size = _inline_cpu_pickle_size(data) + _inline_cpu_pickle_size(metadata)
    return inline_size <= _IPC_INLINE_CPU_BYTES_LIMIT


def ipc_pickle(obj: Any) -> bytes:
    """Serialize via ForkingPickler only when CUDA IPC tensor handles are needed."""
    if not _contains_cuda_tensor(obj):
        return pickle.dumps(obj)
    buf = io.BytesIO()
    ForkingPickler(buf, 2).dump(obj)
    return buf.getvalue()


def _serialize_ipc_metadata_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"_ipc_tensor": ipc_pickle(value)}
    if isinstance(value, dict):
        return {key: _serialize_ipc_metadata_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_ipc_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return {"_ipc_tuple": [_serialize_ipc_metadata_value(item) for item in value]}
    return value


def serialize_ipc_chunk(
    data: Any,
    metadata: dict | None,
) -> dict[str, Any]:
    ipc_metadata: dict[str, Any] = {"_ipc": True}
    ipc_metadata["tensor_bytes"] = ipc_pickle(data)

    if metadata:
        ipc_metadata["metadata"] = _serialize_ipc_metadata_value(metadata)

    return ipc_metadata


def deserialize_ipc_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"_ipc_tensor"}:
            return pickle.loads(value["_ipc_tensor"])
        if set(value) == {"_ipc_tuple"}:
            return tuple(deserialize_ipc_metadata(item) for item in value["_ipc_tuple"])
        return {key: deserialize_ipc_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [deserialize_ipc_metadata(item) for item in value]
    return value


async def send_stream_chunk(
    relay: Relay,
    control_plane: Any,
    *,
    request_id: str,
    data: Any,
    target_stage: str,
    target_endpoint: str,
    from_stage: str,
    chunk_id: int,
    metadata: dict | None = None,
    same_gpu_targets: set[str] | None = None,
) -> None:
    """Send a streaming chunk to a downstream stage."""
    # Keep CUDA IPC limited to CUDA-dominant chunks with no CPU tensors and only
    # small inline Python metadata; otherwise the relay path keeps CPU-heavy
    # pieces out of the IPC control-plane pickle.
    if (
        same_gpu_targets
        and target_stage in same_gpu_targets
        and _should_use_cuda_ipc_stream_chunk(data, metadata)
    ):
        msg = DataReadyMessage(
            request_id=request_id,
            from_stage=from_stage,
            to_stage=target_stage,
            shm_metadata=serialize_ipc_chunk(data, metadata),
            chunk_id=chunk_id,
        )
        await control_plane.send_to_stage(target_stage, target_endpoint, msg)
        return

    if (
        same_gpu_targets
        and target_stage in same_gpu_targets
        and _contains_cuda_tensor(data)
        and not isinstance(data, torch.Tensor)
    ):
        raise ValueError(
            "CUDA IPC stream chunks with mixed object graphs must not carry "
            "CPU-heavy data through the control plane; use tensor data with "
            "relay-backed metadata instead"
        )

    blob_key = f"{request_id}:stream:{from_stage}:{target_stage}:{chunk_id}"

    pending_ops = []
    relay_metadata, op = await write_blob(relay, blob_key, data)
    pending_ops.append(op)

    if metadata:
        cleaned_meta, tensor_dict = extract_tensors(metadata)
        relay_metadata["chunk_metadata"] = cleaned_meta
        if tensor_dict:
            metadata_refs: dict[str, Any] = {}
            for meta_idx, (tkey, tensor) in enumerate(tensor_dict.items()):
                meta_blob_key = f"{blob_key}:meta:{meta_idx}"
                meta_relay_info, meta_op = await write_blob(
                    relay, meta_blob_key, tensor
                )
                pending_ops.append(meta_op)
                metadata_refs[tkey] = {
                    "blob_key": meta_blob_key,
                    "relay_metadata": meta_relay_info,
                }
            relay_metadata["chunk_metadata_tensors"] = metadata_refs

    # Send control message FIRST — receiver starts reading immediately.
    # NIXL credit deadlock avoidance: if we wait_for_completion before notifying,
    # the receiver never starts reading, never triggers RDMA notification, deadlock.
    msg = DataReadyMessage(
        request_id=request_id,
        from_stage=from_stage,
        to_stage=target_stage,
        shm_metadata=relay_metadata,
        chunk_id=chunk_id,
    )
    await control_plane.send_to_stage(target_stage, target_endpoint, msg)

    for pending_op in pending_ops:
        await pending_op.wait_for_completion()


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
    """Send stream done/error signal to downstream stage."""
    msg = DataReadyMessage(
        request_id=request_id,
        from_stage=from_stage,
        to_stage=target_stage,
        shm_metadata={},
        is_done=is_done,
        error=error,
    )
    await control_plane.send_to_stage(target_stage, target_endpoint, msg)
