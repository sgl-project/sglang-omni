# SPDX-License-Identifier: Apache-2.0
"""Data plane adapter for relay IO and payload serialization."""

from __future__ import annotations

import base64
import pickle
from typing import Any

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
        transport_device = torch.device(device)

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

        # Concatenate all tensors into a single flat buffer
        if tensor_dict:
            tensor_buffers = []
            tensor_info = []
            offset = 0

            for path, tensor in tensor_dict.items():
                # Flatten tensor to bytes
                flat = tensor.contiguous().view(torch.uint8).reshape(-1)
                if flat.device != transport_device:
                    flat = flat.to(device=transport_device)
                alignment = max(tensor.element_size(), 1)
                padding = (-offset) % alignment
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

            # Concatenate all tensors
            all_tensors = torch.cat(tensor_buffers)
        else:
            # Relay still expects a payload to transfer; use a 1-byte placeholder.
            all_tensors = torch.zeros(1, dtype=torch.uint8, device=device)
            tensor_info = []

        # Transfer via relay
        op = await self._relay.put_async(all_tensors, request_id=request_id)

        # Send metadata via control plane
        return {
            "relay_info": op.metadata,
            "payload_pickle": base64.b64encode(metadata_bytes).decode("ascii"),
            "tensor_info": tensor_info,
        }, op

    async def read_payload(
        self,
        request_id: str,
        metadata: dict[str, Any],
    ) -> StagePayload:
        device = self._relay.device if hasattr(self._relay, "device") else "cpu"

        # Deserialize payload (without tensors) from control-plane metadata
        payload_bytes = base64.b64decode(metadata["payload_pickle"])
        payload_no_tensors = pickle.loads(payload_bytes)

        relay_info = metadata["relay_info"]
        tensor_info = metadata.get("tensor_info", [])
        tensor_dict = {}

        # Receive tensor bytes via relay (even if empty) to complete transfer.
        data_size = relay_info["transfer_info"]["size"]
        relay_device = torch.device(device)
        recv_tensor = torch.zeros(data_size, dtype=torch.uint8, device=relay_device)
        op = await self._relay.get_async(
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

                # Extract tensor bytes
                tensor_bytes = recv_tensor[offset : offset + size]

                # Parse dtype
                dtype = getattr(torch, dtype_str.replace("torch.", ""))

                # Keep a view into the relay receive buffer instead of cloning;
                # write_payload pads offsets so dtype views are aligned.
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

    async def write_blob(
        self,
        key: str,
        tensor: torch.Tensor,
    ) -> tuple[dict[str, Any], Any]:
        """Write a raw tensor via relay (no StagePayload wrapping).

        Returns (metadata_dict, relay_operation).
        metadata_dict is sent via control plane; receiver uses it in read_blob.
        """
        flat = tensor.contiguous().view(torch.uint8).reshape(-1)
        op = await self._relay.put_async(flat, request_id=key)
        metadata = {
            "relay_info": op.metadata,
            "tensor_shape": list(tensor.shape),
            "tensor_dtype": str(tensor.dtype),
        }
        return metadata, op

    async def read_blob(
        self,
        key: str,
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        """Read a raw tensor from relay using metadata from write_blob."""
        device = self._relay.device if hasattr(self._relay, "device") else "cpu"
        relay_info = metadata["relay_info"]
        shape = metadata["tensor_shape"]
        dtype_str = metadata["tensor_dtype"]

        data_size = relay_info["transfer_info"]["size"]
        recv_buf = torch.zeros(data_size, dtype=torch.uint8, device=device)
        op = await self._relay.get_async(
            metadata=relay_info, dest_tensor=recv_buf, request_id=key
        )
        await op.wait_for_completion()

        dtype = getattr(torch, dtype_str.replace("torch.", ""))
        tensor = recv_buf.view(dtype).reshape(shape)
        return tensor

    def cleanup(self, request_id: str) -> None:
        self._relay.cleanup(request_id)
