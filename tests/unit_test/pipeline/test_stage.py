# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import torch

from sglang_omni_v1.pipeline import relay_io
from sglang_omni_v1.pipeline.stage.input import AggregatedInput
from sglang_omni_v1.pipeline.stage.stream_queue import StreamItem, StreamQueue
from sglang_omni_v1.proto import DataReadyMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    EventLog,
    FakeRelay,
    FakeScheduler,
    RecordingStageControlPlane,
    collect_event_names,
    make_noop_projector,
    make_result_message,
    make_stage_payload,
    make_stream_message,
    make_tensor_payload,
    tensor_equal,
)
from tests.unit_test.pipeline.helpers import make_stage


def test_aggregated_input_waits_per_request_without_cross_talk() -> None:
    """Preserves per-request fan-in isolation when requests interleave."""
    handler = AggregatedInput(
        {"preprocess", "image"},
        lambda payloads: make_stage_payload(data={"sources": sorted(payloads)}),
    )

    assert handler.receive("req-1", "preprocess", make_stage_payload()) is None
    assert handler.receive("req-2", "preprocess", make_stage_payload()) is None
    req2 = handler.receive("req-2", "image", make_stage_payload())
    req1 = handler.receive("req-1", "image", make_stage_payload())

    assert req2.data == {"sources": ["image", "preprocess"]}
    assert req1.data == {"sources": ["image", "preprocess"]}


def test_stage_routes_results_streams_and_clears_abort_state() -> None:
    """Preserves result routing, stream forwarding, and abort cleanup."""

    async def _run() -> None:
        relay = FakeRelay()
        scheduler = FakeScheduler()
        control_plane = RecordingStageControlPlane()
        stage_obj = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "decode",
            endpoints={"decode": "inproc://decode", "talker": "inproc://talker"},
            project_payload={"decode": make_noop_projector("decode-only")},
            stream_targets=["talker"],
            relay=relay,
            scheduler=scheduler,
            control_plane=control_plane,
        )
        stage_obj._active_requests.add("req-1")
        scheduler.outbox.put(make_stream_message("req-1", data=torch.tensor([7])))
        scheduler.outbox.put(make_result_message("req-1", data={"answer": 1}))

        await stage_obj._drain_outbox()

        decode_msg = next(
            msg for target, _, msg in control_plane.sent_to_stage if target == "decode"
        )
        restored = await relay_io.read_payload(relay, "req-1", decode_msg.shm_metadata)
        assert restored.data == {"marker": "decode-only", "data": {"answer": 1}}
        stream_msg = next(
            msg
            for target, _, msg in control_plane.sent_to_stage
            if target == "talker" and msg.chunk_id == 0
        )
        assert stream_msg.chunk_id == 0

        stage_obj._stream_queue = StreamQueue()
        stage_obj._stream_queue.open("req-1")
        stage_obj._pending_stream_data["req-1"] = [
            StreamItem(0, torch.tensor([1]), "t")
        ]
        stage_obj._on_abort("req-1")

        assert "req-1" in stage_obj._aborted
        assert relay.cleaned[-1] == "req-1"
        assert scheduler.aborted == ["req-1"]
        assert "req-1" not in stage_obj._pending_stream_data

    asyncio.run(_run())


def test_relay_payload_and_cross_gpu_stream_contracts() -> None:
    """Preserves tensor payload round-trips and stream control-before-wait ordering."""

    async def _run() -> None:
        relay = FakeRelay()
        payload = make_tensor_payload()
        metadata, op = await relay_io.write_payload(relay, payload.request_id, payload)
        await op.wait_for_completion()
        restored = await relay_io.read_payload(relay, payload.request_id, metadata)
        assert tensor_equal(restored.data, payload.data)

        log = EventLog()
        stream_relay = FakeRelay(log=log)
        control_plane = RecordingStageControlPlane()
        control_plane.log = log
        await relay_io.send_stream_chunk(
            stream_relay,
            control_plane,
            request_id="req-1",
            data=torch.tensor([1, 2, 3]),
            target_stage="talker",
            target_endpoint="inproc://talker",
            from_stage="thinker",
            chunk_id=0,
            metadata={"token_id": 1, "hidden": torch.tensor([4])},
        )

        names = collect_event_names(log)
        assert names.index("stage_cp_send_to_stage") < names.index("op_wait")
        msg = control_plane.sent_to_stage[0][2]
        assert msg.shm_metadata["chunk_metadata"]["token_id"] == 1
        assert "hidden" in msg.shm_metadata["chunk_metadata_tensors"]

    asyncio.run(_run())


def test_stage_relay_read_failure_completes_with_error() -> None:
    """Preserves failure reporting when a stage cannot read its relay payload."""

    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        stage_obj = make_stage(relay=relay, control_plane=control_plane)
        payload = make_stage_payload(request_id="req-1")
        metadata, _ = await relay_io.write_payload(relay, "req-1", payload)
        relay.fail_get = RuntimeError("read failed")

        await stage_obj._on_data_ready(
            DataReadyMessage("req-1", "upstream", "stage", metadata)
        )

        assert control_plane.completions[0].success is False
        assert "relay read failed" in control_plane.completions[0].error
        assert relay.cleaned[-1] == "req-1"

    asyncio.run(_run())
