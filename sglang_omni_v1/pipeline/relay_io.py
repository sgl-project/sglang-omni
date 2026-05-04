# SPDX-License-Identifier: Apache-2.0
"""Relay IO utilities for inter-stage data transfer.

Handles payload serialization (tensor extraction/restoration), relay read/write,
streaming chunk transfer (same-GPU CUDA IPC and cross-GPU relay), and
NIXL credit deadlock avoidance.

Extracted from worker/data_plane.py and worker/runtime.py.
"""
from __future__ import annotations

import base64
import io
import logging
import pickle
from multiprocessing.reduction import ForkingPickler
from typing import Any

import torch

from sglang_omni_v1.proto import DataReadyMessage, StagePayload
from sglang_omni_v1.relay.base import Relay

logger = logging.getLogger(__name__)


def _dtype_alignment(dtype: torch.dtype) -> int:
    return max(torch.empty((), dtype=dtype).element_size(), 1)


def _pad_offset(offset: int, alignment: int) -> int:
    return (-offset) % alignment


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


# ---------------------------------------------------------------------------
# Payload read/write (full StagePayload via relay)
# ---------------------------------------------------------------------------


async def write_payload(
    relay: Relay,
    request_id: str,
    payload: StagePayload,
) -> tuple[dict[str, Any], Any]:
    """Write a StagePayload to relay. Returns (control_plane_metadata, relay_op)."""
    device = getattr(relay, "device", "cpu")
    transport_device = torch.device(device)

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

    return {
        "relay_info": op.metadata,
        "payload_pickle": base64.b64encode(metadata_bytes).decode("ascii"),
        "tensor_info": tensor_info,
    }, op


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
            dtype = getattr(torch, dtype_str.replace("torch.", ""))
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

    dtype = getattr(torch, dtype_str.replace("torch.", ""))
    return recv_buf[offset:].view(dtype).reshape(shape)


# ---------------------------------------------------------------------------
# Stream chunk send (handles same-GPU IPC vs cross-GPU relay)
# ---------------------------------------------------------------------------


def ipc_pickle(obj: Any) -> bytes:
    """Serialize via ForkingPickler (CUDA IPC for GPU tensors, zero data copy)."""
    buf = io.BytesIO()
    ForkingPickler(buf, 2).dump(obj)
    return buf.getvalue()


def serialize_ipc_chunk(
    data: Any,
    metadata: dict | None,
) -> dict[str, Any]:
    """Build IPC metadata for a same-GPU stream chunk."""
    ipc_metadata: dict[str, Any] = {"_ipc": True}
    ipc_metadata["tensor_bytes"] = ipc_pickle(data)

    if metadata:
        serialized_meta: dict[str, Any] = {}
        for mkey, value in metadata.items():
            if isinstance(value, torch.Tensor):
                serialized_meta[mkey] = {"_ipc_tensor": ipc_pickle(value)}
            else:
                serialized_meta[mkey] = value
        ipc_metadata["metadata"] = serialized_meta

    return ipc_metadata


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
    """Send a streaming chunk to a downstream stage.

    Handles:
    - Same-GPU: CUDA IPC (zero copy)
    - Cross-GPU: relay write + NIXL credit deadlock avoidance
    """
    # ── Same-GPU CUDA IPC ──
    if same_gpu_targets and target_stage in same_gpu_targets:
        ipc_meta = serialize_ipc_chunk(data, metadata)
        ipc_meta["chunk_id"] = chunk_id
        msg = DataReadyMessage(
            request_id=request_id,
            from_stage=from_stage,
            to_stage=target_stage,
            shm_metadata=ipc_meta,
            chunk_id=chunk_id,
        )
        await control_plane.send_to_stage(target_stage, target_endpoint, msg)
        return

    # ── Cross-GPU: relay ──
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
