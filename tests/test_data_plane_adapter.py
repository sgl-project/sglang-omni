# SPDX-License-Identifier: Apache-2.0
"""Tests for DataPlaneAdapter metadata/tensor separation."""

import pytest
import torch

from sglang_omni.pipeline.worker.data_plane import DataPlaneAdapter
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.relay.shm import ShmRelay


@pytest.mark.asyncio
async def test_data_plane_roundtrip_with_tensor():
    sender = ShmRelay("sender", device="cpu")
    receiver = ShmRelay("receiver", device="cpu")
    sender_adapter = DataPlaneAdapter(sender)
    receiver_adapter = DataPlaneAdapter(receiver)

    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs={"text": "hello"}),
        data={
            "tensor": torch.arange(8, dtype=torch.float32),
            "meta": {"flag": True},
        },
    )

    metadata, put_op = await sender_adapter.write_payload("req-1", payload)
    assert "relay_info" in metadata
    assert "payload_pickle" in metadata
    assert "tensor_info" in metadata

    received = await receiver_adapter.read_payload("req-1", metadata)
    await put_op.wait_for_completion()

    assert received.request_id == payload.request_id
    assert received.request.inputs == payload.request.inputs
    assert received.data["meta"] == payload.data["meta"]
    assert torch.equal(received.data["tensor"], payload.data["tensor"])


@pytest.mark.asyncio
async def test_data_plane_roundtrip_without_tensor():
    sender = ShmRelay("sender", device="cpu")
    receiver = ShmRelay("receiver", device="cpu")
    sender_adapter = DataPlaneAdapter(sender)
    receiver_adapter = DataPlaneAdapter(receiver)

    payload = StagePayload(
        request_id="req-2",
        request=OmniRequest(inputs={"text": "no_tensor"}),
        data={"meta": {"count": 3}},
    )

    metadata, put_op = await sender_adapter.write_payload("req-2", payload)
    received = await receiver_adapter.read_payload("req-2", metadata)
    await put_op.wait_for_completion()

    assert received.request_id == payload.request_id
    assert received.request.inputs == payload.request.inputs
    assert received.data == payload.data
