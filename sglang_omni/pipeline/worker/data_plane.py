# SPDX-License-Identifier: Apache-2.0
"""Data plane adapter for relay IO and payload serialization."""

from __future__ import annotations

import pickle
import struct
from typing import Any

import numpy as np
import torch

from sglang_omni.proto import StagePayload
from sglang_omni.relay.base import Relay


def _extract_tensors(obj: Any, path: str = "") -> tuple[Any, dict[str, torch.Tensor]]:
    """Recursively extract tensors from nested structure, replacing them with placeholders.

    Returns:
        (modified_obj, tensor_dict) where tensor_dict maps path -> tensor
    """
    tensors = {}

    if isinstance(obj, torch.Tensor):
        # Replace tensor with placeholder
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
            new_value, sub_tensors = _extract_tensors(value, new_path)
            new_dict[key] = new_value
            tensors.update(sub_tensors)
        return new_dict, tensors

    elif isinstance(obj, (list, tuple)):
        new_list = []
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]"
            new_item, sub_tensors = _extract_tensors(item, new_path)
            new_list.append(new_item)
            tensors.update(sub_tensors)
        return (type(obj)(new_list), tensors)

    else:
        return obj, tensors


def _restore_tensors(obj: Any, tensor_dict: dict[str, torch.Tensor]) -> Any:
    """Recursively restore tensors from placeholders."""
    if isinstance(obj, dict):
        if "_tensor_placeholder" in obj:
            # This is a tensor placeholder
            path = obj["_tensor_placeholder"]
            return tensor_dict.get(path)
        else:
            return {
                key: _restore_tensors(value, tensor_dict) for key, value in obj.items()
            }

    elif isinstance(obj, (list, tuple)):
        return type(obj)(_restore_tensors(item, tensor_dict) for item in obj)

    else:
        return obj


class DataPlaneAdapter:
    """Serialize StagePayloads and transfer them via a Relay.

    Optimized for GPU tensors: extracts tensors and transfers them directly
    via relay (keeping them on GPU), while metadata is serialized via pickle.
    """

    def __init__(self, relay: Relay):
        self._relay = relay

    async def write_payload(
        self,
        request_id: str,
        payload: StagePayload,
    ) -> tuple[dict[str, Any], Any]:
        device = self._relay.device if hasattr(self._relay, "device") else "cpu"

        # Extract tensors from payload.data
        modified_data, tensor_dict = _extract_tensors(payload.data)

        # Create a payload copy with tensors replaced by placeholders
        payload_no_tensors = StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=modified_data,
        )

        # Serialize metadata (without tensors)
        metadata_bytes = pickle.dumps(payload_no_tensors)
        metadata_size = len(metadata_bytes)

        # Concatenate all tensors into a single flat buffer
        if tensor_dict:
            tensor_buffers = []
            tensor_info = []
            offset = 0

            for path, tensor in tensor_dict.items():
                # Flatten tensor to bytes
                flat = tensor.contiguous().view(torch.uint8).reshape(-1)
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

            # Concatenate all tensors
            if tensor_buffers[0].is_cuda:
                all_tensors = torch.cat(tensor_buffers)
            else:
                all_tensors = torch.cat(tensor_buffers)
        else:
            all_tensors = torch.tensor([], dtype=torch.uint8, device=device)
            tensor_info = []

        # Pack everything: [metadata_size (8 bytes)] [metadata_bytes] [tensor_bytes]
        metadata_size_bytes = struct.pack("<Q", metadata_size)
        metadata_tensor = torch.tensor(
            np.frombuffer(metadata_size_bytes + metadata_bytes, dtype=np.uint8),
            dtype=torch.uint8,
            device=device,
        )

        # Combine metadata and tensors
        combined = torch.cat([metadata_tensor, all_tensors])

        # Transfer via relay
        op = await self._relay.put_async(combined, request_id=request_id)
        metadata = op.metadata

        # Add tensor info to metadata
        metadata["tensor_info"] = tensor_info

        return metadata, op

    async def read_payload(
        self,
        request_id: str,
        metadata: dict[str, Any],
    ) -> StagePayload:
        data_size = metadata["transfer_info"]["size"]
        device = self._relay.device if hasattr(self._relay, "device") else "cpu"

        # Receive combined data
        recv_tensor = torch.zeros(data_size, dtype=torch.uint8, device=device)
        op = await self._relay.get_async(
            metadata=metadata, dest_tensor=recv_tensor, request_id=request_id
        )
        await op.wait_for_completion()

        # Parse metadata size (first 8 bytes)
        if recv_tensor.is_cuda:
            metadata_size_bytes = recv_tensor[:8].cpu().numpy().tobytes()
        else:
            metadata_size_bytes = recv_tensor[:8].numpy().tobytes()

        metadata_size = struct.unpack("<Q", metadata_size_bytes)[0]

        # Extract metadata bytes
        metadata_end = 8 + metadata_size
        if recv_tensor.is_cuda:
            metadata_bytes = recv_tensor[8:metadata_end].cpu().numpy().tobytes()
        else:
            metadata_bytes = recv_tensor[8:metadata_end].numpy().tobytes()

        # Deserialize payload (without tensors)
        payload_no_tensors = pickle.loads(metadata_bytes)

        # Extract tensor data
        tensor_info = metadata.get("tensor_info", [])
        tensor_dict = {}

        if tensor_info:
            tensor_data = recv_tensor[metadata_end:]

            for info in tensor_info:
                path = info["path"]
                shape = info["shape"]
                dtype_str = info["dtype"]
                offset = info["offset"]
                size = info["size"]

                # Extract tensor bytes
                tensor_bytes = tensor_data[offset : offset + size]

                # Parse dtype
                dtype = getattr(torch, dtype_str.replace("torch.", ""))

                # Reconstruct tensor (keep on original device)
                tensor = tensor_bytes.view(dtype).reshape(shape)
                tensor_dict[path] = tensor

        # Restore tensors into payload
        restored_data = _restore_tensors(payload_no_tensors.data, tensor_dict)
        payload = StagePayload(
            request_id=payload_no_tensors.request_id,
            request=payload_no_tensors.request,
            data=restored_data,
        )

        self._relay.cleanup(request_id)

        if not isinstance(payload, StagePayload):
            raise TypeError(f"Expected StagePayload, got {type(payload)}")
        return payload

    def cleanup(self, request_id: str) -> None:
        self._relay.cleanup(request_id)
