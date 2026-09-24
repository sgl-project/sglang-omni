# SPDX-License-Identifier: Apache-2.0
"""Continuation metadata over real packed SHM and direct-IPC framing."""
from __future__ import annotations

import asyncio
import base64
import json
import pickle
import threading

import pytest
import torch

from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import DataRef, TransportKind
from sglang_omni.pipeline.umm import UMMController, UMMDecision, route_umm
from sglang_omni.proto import (
    DataAckMessage,
    DataReadyMessage,
    OmniRequest,
    StagePayload,
)
from sglang_omni.proto.continuation import ContinuationToken
from sglang_omni.relay.shm import ShmRelay
from tests.unit_test.fixtures.pipeline_fakes import RecordingStageControlPlane
from tests.unit_test.pipeline.helpers import make_stage


def _payload(continuation, data):
    value = StagePayload(
        "req",
        OmniRequest("inspect and revise", {"seed": 0}, {"trace": "original"}),
        data,
        continuation,
    )
    value.arrival_id = object()
    value.prefetched_chunks = [threading.Lock()]
    value.prefetched_stream_done = True
    value.stream_queue = threading.Lock()
    return value


def _assert_wire_fields(actual, original):
    assert actual.request_id == original.request_id
    assert actual.request == original.request
    assert actual.continuation == original.continuation
    assert actual.arrival_id is None
    assert actual.prefetched_chunks == []
    assert actual.prefetched_stream_done is False
    assert not hasattr(actual, "stream_queue")


def _add_stale_local_state(header):
    # A receiver must also discard old scheduler state from an incoming header.
    header.arrival_id = "foreign-arrival"
    header.prefetched_chunks = ["foreign-chunk"]
    header.prefetched_stream_done = True
    header.stream_queue = "foreign-queue"


class _FakeCudaTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True

    def __reduce_ex__(self, protocol):
        return (_fake_cuda_tensor, ())


def _fake_cuda_tensor():
    return torch.Tensor._make_subclass(_FakeCudaTensor, torch.arange(3), False)


@pytest.mark.parametrize(
    "continuation",
    [None, ContinuationToken("session", 1, "generation", "guard")],
)
def test_direct_ipc_metadata_preserves_wire_fields_and_resets_local_state(
    monkeypatch, continuation
):
    # Only CUDA storage export is mocked. This qualifies metadata framing,
    # not real CUDA IPC storage transfer.
    monkeypatch.setattr(stage_io, "ipc_pickle", pickle.dumps)
    original = _payload(continuation, {"tensor": _fake_cuda_tensor()})
    ref = stage_io.serialize_direct_cuda_ipc_payload(original)
    header = pickle.loads(ref["header"])
    _assert_wire_fields(header, original)
    _add_stale_local_state(header)
    ref["header"] = pickle.dumps(header)
    restored = stage_io.deserialize_direct_cuda_ipc_payload(ref)
    _assert_wire_fields(restored, original)
    assert isinstance(restored.data["tensor"], _FakeCudaTensor)
    assert restored.data["tensor"].tolist() == [0, 1, 2]


class _Adapter:
    def start(self, request):
        return []

    def reasoner_request(self, history, request, *, remaining_generation_turns):
        return OmniRequest(inputs=history)

    def interpret_reasoner(self, result):
        return UMMDecision(**result)

    def generation_request(self, decision, request):
        return OmniRequest(inputs=decision.generation)

    def incorporate_media(self, history, decision, result):
        return [*history, result]

    def media_segments(self, result):
        return [{"kind": "image", "data": result}]


@pytest.mark.asyncio
async def test_actual_stage_routing_preserves_two_umm_cycles_over_packed_shm():
    stages = {}
    transfers = []
    controller = UMMController(_Adapter())

    class Bridge(RecordingStageControlPlane):
        async def send_to_stage(self, target, endpoint, msg):
            await super().send_to_stage(target, endpoint, msg)
            raw = json.loads(json.dumps(msg.to_dict()))
            if isinstance(msg, DataAckMessage):
                stages[target].comm.ack_transfer(DataAckMessage.from_dict(raw))
                return
            assert isinstance(msg, DataReadyMessage)
            ready = DataReadyMessage.from_dict(raw)
            ref = DataRef.from_dict(ready.data_ref)
            assert ref.transport is TransportKind.SHM
            header = pickle.loads(base64.b64decode(ref.header))
            assert header.continuation is not None
            assert header.arrival_id is None
            transfers.append((msg.from_stage, target, header.continuation))
            await stages[target].on_data_ready(ready)

    endpoints = {
        name: "inproc://" + name for name in ("orchestrator", "reasoner", "generation")
    }
    for name in endpoints:
        stages[name] = make_stage(
            name=name,
            get_next=(
                route_umm if name == "orchestrator" else lambda *_: "orchestrator"
            ),
            endpoints=endpoints,
            control_plane=Bridge(),
            relay=ShmRelay(name, device="cpu", credits=1),
            scheduler=controller if name == "orchestrator" else None,
            is_terminal=name == "orchestrator",
        )

    async def route(name, value):
        await asyncio.wait_for(stages[name].route_result("req", value), timeout=2)
        pending = [
            item.task
            for obj in stages.values()
            for item in obj.comm.pending.values()
            if item.task is not None
        ]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending), timeout=2)

    def received(name):
        item = stages[name].scheduler.inbox.get_nowait().data
        assert item.continuation is not None
        assert item.arrival_id is stages[name].request_arrivals["req"]
        return item

    try:
        initial = StagePayload("req", OmniRequest("draw then revise"), None)
        await stages["orchestrator"].receive_payload_from_stage(
            "req", "coordinator", initial
        )
        item = controller.inbox.get_nowait().data
        await route("orchestrator", controller.advance(item))
        for turn in range(2):
            item = received("reasoner")
            assert item.continuation.turn_index == turn
            item.data = {
                "kind": "generate",
                "text": f"turn{turn}",
                "generation": {},
            }
            await route("reasoner", item)
            item = received("orchestrator")
            await route("orchestrator", controller.advance(item))
            item = received("generation")
            assert item.continuation.phase == "generation"
            item.data = {"url": f"inline-{turn}"}
            await route("generation", item)
            item = received("orchestrator")
            await route("orchestrator", controller.advance(item))
        item = received("reasoner")
        item.data = {"kind": "final", "text": "done"}
        await route("reasoner", item)
        item = received("orchestrator")
        await route("orchestrator", controller.advance(item))
        assert controller.active_sessions == 0
        completions = stages["orchestrator"].control_plane.completions
        assert len(completions) == 1
        assert completions[0].success
        assert completions[0].result["text"] == "turn0turn1done"
        assert len(completions[0].result["segments"]) == 5
        assert len(transfers) == 10
        assert all(not obj.comm.pending for obj in stages.values())
    finally:
        controller.stop()
        tasks = [
            task for obj in stages.values() for task in obj.comm.send_workers.values()
        ]
        for obj in stages.values():
            await obj.comm.close()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
