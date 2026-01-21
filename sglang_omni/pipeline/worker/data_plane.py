# SPDX-License-Identifier: Apache-2.0
"""Data plane adapter for relay IO and payload serialization."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import torch

from sglang_omni.proto import StagePayload
from sglang_omni.relay.base import Relay


class DataPlaneAdapter:
    """Serialize StagePayloads and transfer them via a Relay."""

    def __init__(self, relay: Relay):
        self._relay = relay

    async def write_payload(
        self,
        request_id: str,
        payload: StagePayload,
    ) -> tuple[dict[str, Any], Any]:
        serialized = pickle.dumps(payload)
        data_np = np.frombuffer(serialized, dtype=np.uint8).copy()
        device = self._relay.device if hasattr(self._relay, "device") else "cpu"
        tensor = torch.tensor(data_np, dtype=torch.uint8, device=device)

        op = await self._relay.put_async(tensor, request_id=request_id)
        metadata = op.metadata
        return metadata, op

    async def read_payload(
        self,
        request_id: str,
        metadata: dict[str, Any],
    ) -> StagePayload:
        data_size = metadata["transfer_info"]["size"]
        device = self._relay.device if hasattr(self._relay, "device") else "cpu"
        recv_tensor = torch.zeros(data_size, dtype=torch.uint8, device=device)
        op = await self._relay.get_async(
            metadata=metadata, dest_tensor=recv_tensor, request_id=request_id
        )
        await op.wait_for_completion()

        if recv_tensor.is_cuda:
            buffer_bytes = recv_tensor.cpu().numpy().tobytes()
        else:
            buffer_bytes = recv_tensor.numpy().tobytes()

        payload = pickle.loads(buffer_bytes)
        self._relay.cleanup(request_id)

        if not isinstance(payload, StagePayload):
            raise TypeError(f"Expected StagePayload, got {type(payload)}")
        return payload

    def cleanup(self, request_id: str) -> None:
        self._relay.cleanup(request_id)
