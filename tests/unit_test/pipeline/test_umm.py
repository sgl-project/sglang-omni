# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import copy
import logging
import threading
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, _construct_stage
from sglang_omni.pipeline.umm import UMMController, UMMDecision, UMMLimits, route_umm
from sglang_omni.proto.continuation import ContinuationToken
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
from tests.unit_test.fixtures.pipeline_fakes import fake_factory_path
from tests.unit_test.pipeline.helpers import make_stage, stage


class Adapter:
    def start(self, request):
        return [{"role": "user", "content": request.inputs}]

    def reasoner_request(self, history, request):
        return OmniRequest(inputs={"messages": copy.deepcopy(history)}, params={})

    def interpret_reasoner(self, result):
        return UMMDecision(**result)

    def generation_request(self, decision, request):
        return OmniRequest(inputs=decision.generation, params={})

    def incorporate_media(self, history, decision, result):
        return [
            *history,
            {"role": "assistant", "content": decision.generation},
            {"role": "user", "content": result},
        ]

    def media_segments(self, result):
        return [{"kind": "image", "data": result}]


def request(request_id="req", *, stream=False):
    req = OmniRequest(inputs="draw, inspect, and revise", params={"stream": stream})
    return StagePayload(request_id=request_id, request=req, data=req.inputs)


def reply(payload, data):
    restored = StagePayload.from_dict(payload.to_dict())
    restored.data = data
    return restored


def generate(payload, text=""):
    return reply(
        payload,
        {"kind": "generate", "text": text, "generation": {"prompt": "a tree"}},
    )


def test_two_model_adapter_cycles_and_ordered_segments():
    controller = UMMController(Adapter())
    current = controller._advance(request(stream=True))
    session = current.continuation.session_id
    for turn in range(2):
        assert current.continuation.phase == "reasoner"
        assert current.continuation.turn_index == turn
        current = controller._advance(generate(current, f"render {turn}"))
        assert current.continuation.phase == "generation"
        current = controller._advance(reply(current, {"url": f"inline-{turn}"}))
        assert (
            current.request.inputs["messages"][-1]["content"]["url"] == f"inline-{turn}"
        )
    final = controller._advance(reply(current, {"kind": "final", "text": "finished"}))
    assert route_umm("req", final) is None
    assert controller.active_sessions == 0
    assert final.data["text"] == "render 0render 1finished"
    segments = final.data["segments"]
    assert [s["segment_index"] for s in segments] == list(range(5))
    assert [s["kind"] for s in segments] == ["text", "image", "text", "image", "text"]
    assert {s["session_id"] for s in segments} == {session}
    streamed = [controller.outbox.get_nowait().data for _ in range(5)]
    assert streamed == segments
    assert controller.outbox.empty()


def test_stale_and_duplicate_results_never_advance_or_corrupt_current_turn():
    controller = UMMController(Adapter())
    first = controller._advance(request())
    second = controller._advance(generate(first))
    stale = controller._advance(generate(first))
    controller._emit_result("req", stale, controller.outbox)
    discarded = controller.outbox.get_nowait()
    assert discarded.type == "discard"
    assert discarded.metadata["keep_active"] is True
    forged = reply(second, {"url": "not accepted"})
    forged.continuation = replace(second.continuation, nonce="wrong")
    controller._advance(forged)
    assert controller._sessions["req"].expected == second.continuation
    resumed = controller._advance(reply(second, {"url": "accepted"}))
    final = controller._advance(reply(resumed, {"kind": "final", "text": "done"}))
    late = controller._advance(reply(second, {"url": "late"}))
    controller._emit_result("req", late, controller.outbox)
    assert controller.outbox.get_nowait().metadata["keep_active"] is False
    assert len(final.data["segments"]) == 2
    assert controller.active_sessions == 0


def test_abort_and_stop_release_all_sessions_and_late_results():
    controller = UMMController(Adapter())
    first = controller._advance(request("first"))
    controller._advance(request("second"))
    controller.abort("first")
    assert controller.active_sessions == 1
    dropped = controller._advance(generate(first))
    assert dropped.keep_active is False
    controller.stop()
    controller.stop()
    assert controller.active_sessions == 0


def test_timeout_while_waiting_for_native_stage_emits_error_and_recovers():
    clock = [0.0]
    controller = UMMController(
        Adapter(), limits=UMMLimits(timeout_s=1), clock=lambda: clock[0]
    )
    late = controller._advance(request())
    clock[0] = 1.0
    controller._expire()
    error = controller.outbox.get_nowait()
    assert error.type == "error"
    assert isinstance(error.data, TimeoutError)
    assert controller.active_sessions == 0
    assert controller._advance(generate(late)).keep_active is False
    assert controller._advance(request("healthy")).continuation.phase == "reasoner"


def test_capacity_rejection_preserves_existing_session():
    controller = UMMController(Adapter(), limits=UMMLimits(max_sessions=1))
    original = controller._advance(request())
    with pytest.raises(QueueFullError) as rejected:
        controller._advance(request("excess"))
    assert isinstance(QueueFullError.from_message(str(rejected.value)), QueueFullError)
    assert controller.active_sessions == 1
    assert controller._sessions["req"].expected == original.continuation
    controller._advance(reply(original, {"kind": "final", "text": "done"}))
    assert controller.active_sessions == 0
    assert controller._advance(request("healthy")).continuation.phase == "reasoner"


def test_turn_bound_still_allows_final_reasoning_after_last_generation():
    controller = UMMController(Adapter(), limits=UMMLimits(max_turns=1))
    reasoner = controller._advance(request())
    generation = controller._advance(generate(reasoner))
    last_reasoner = controller._advance(reply(generation, {"url": "media"}))
    final = controller._advance(reply(last_reasoner, {"kind": "final", "text": "done"}))
    assert final.continuation is None

    reasoner = controller._advance(request())
    generation = controller._advance(generate(reasoner))
    last_reasoner = controller._advance(reply(generation, {"url": "media"}))
    with pytest.raises(ValueError, match="max_turns"):
        controller._advance(generate(last_reasoner, "must not stream"))
    assert controller.active_sessions == 0


def test_segment_budget_rejects_entire_transition_before_stream_publication():
    controller = UMMController(Adapter(), limits=UMMLimits(max_segments=1))
    reasoner = controller._advance(request(stream=True))
    generation = controller._advance(generate(reasoner, "first"))
    assert controller.outbox.get_nowait().type == "stream"
    with pytest.raises(ValueError, match="max_segments"):
        controller._advance(reply(generation, {"url": "media"}))
    assert controller.outbox.empty()
    assert controller.active_sessions == 0


def test_retained_byte_budget_covers_initial_input_and_generated_media():
    small = UMMController(Adapter(), limits=UMMLimits(max_context_bytes=10))
    with pytest.raises(ValueError, match="max_context_bytes"):
        small._advance(request())
    assert small.active_sessions == 0

    controller = UMMController(Adapter(), limits=UMMLimits(max_context_bytes=2048))
    reasoner = controller._advance(request())
    generation = controller._advance(generate(reasoner))
    with pytest.raises(ValueError, match="max_context_bytes"):
        controller._advance(reply(generation, {"url": "x" * 4096}))
    assert controller.active_sessions == 0


def test_adapter_failure_isolated_and_next_request_recovers():
    controller = UMMController(Adapter())
    reasoner = controller._advance(request())
    with pytest.raises(ValueError, match="Invalid structured"):
        controller._advance(reply(reasoner, {"kind": "unknown"}))
    assert controller.active_sessions == 0
    assert controller._advance(request("healthy")).continuation.phase == "reasoner"


def test_existing_scheduler_thread_runs_controller_and_reaps_waiting_deadline():
    controller = UMMController(Adapter(), limits=UMMLimits(timeout_s=0.1))
    worker = threading.Thread(target=controller.start)
    worker.start()
    try:
        payload = request()
        controller.inbox.put(IncomingMessage("req", "new_request", payload))
        assert controller.outbox.get(timeout=2).type == "result"
        error = controller.outbox.get(timeout=2)
        assert error.type == "error"
        assert "deadline" in str(error.data)
        assert controller.active_sessions == 0
    finally:
        controller.stop()
        worker.join(timeout=2)
    assert not worker.is_alive()


def test_conditional_terminal_schema_and_route_target_validation():
    config = PipelineConfig(
        model_path="model",
        stages=[
            stage(
                "orchestrator",
                terminal=True,
                next=["reasoner", "generation"],
                route_fn="sglang_omni.pipeline.umm.route_umm",
            ),
            stage("reasoner", next="orchestrator"),
            stage("generation", next="orchestrator"),
        ],
    )
    assert config.terminal_stages == ["orchestrator"]
    spec = StageLaunchConfig(
        stage_name="orchestrator",
        factory=fake_factory_path("make_scheduler"),
        next_stages=["reasoner", "generation"],
        route_fn="sglang_omni.pipeline.umm.route_umm",
        is_terminal=True,
        recv_endpoint="inproc://orchestrator",
        coordinator_endpoint="inproc://coordinator",
        abort_endpoint="inproc://abort",
        stage_endpoints={
            "reasoner": "inproc://reasoner",
            "generation": "inproc://generation",
        },
        comm_config={"slot_size_mb": 1},
    )
    obj = _construct_stage(spec, logging.getLogger(__name__))
    payload = request()
    assert obj.get_next("req", payload) is None
    payload.continuation = ContinuationToken("s", 0, "generation", "n")
    assert obj.get_next("req", payload) == "generation"
    obj = _construct_stage(
        replace(spec, next_stages=["reasoner"]), logging.getLogger(__name__)
    )
    with pytest.raises(ValueError, match="outside the static topology"):
        obj.get_next("req", payload)
    nonterminal = _construct_stage(
        replace(spec, is_terminal=False), logging.getLogger(__name__)
    )
    with pytest.raises(ValueError, match="returned no targets"):
        nonterminal.get_next("req", request())
    with pytest.raises(ValueError, match="identity mismatch"):
        route_umm("different", payload)


@pytest.mark.asyncio
async def test_stream_order_and_ownership_survive_controller_reentry_until_completion():
    obj = make_stage(name="orchestrator", is_terminal=True, get_next=route_umm)
    obj.control_plane.send_stream = AsyncMock()
    obj._send_to_stage = AsyncMock()
    payload = request()
    obj._active_requests.add("req")
    await obj._execute(payload)
    for index in range(2):
        await obj._send_stream_to_coordinator("req", {"text": str(index)})
        payload.continuation = ContinuationToken("s", index, "reasoner", str(index))
        await obj._route_result("req", payload)
        assert "req" in obj._active_requests
        payload = reply(payload, {})
        await obj._receive_payload_from_stage("req", "reasoner", payload)
    await obj._send_stream_to_coordinator("req", {"text": "done"})
    assert [
        call.args[0].chunk_id for call in obj.control_plane.send_stream.call_args_list
    ] == [0, 1, 2]
    payload.continuation = None
    await obj._route_result("req", payload)
    assert "req" not in obj._active_requests
    assert "req" not in obj._request_arrivals
    assert not obj._stream_chunk_counters


@pytest.mark.asyncio
async def test_fast_cycle_cannot_clear_new_arrival_during_send():
    obj = make_stage(name="reasoner", get_next=lambda *_: "orchestrator")
    first = request()
    obj._active_requests.add("req")
    await obj._execute(first)
    old_arrival = obj._request_arrivals["req"]

    async def reenter(*args, **kwargs):
        await obj._receive_payload_from_stage("req", "orchestrator", request())

    obj._send_to_stage = reenter
    await obj._route_result("req", first)
    assert "req" in obj._active_requests
    assert obj._request_arrivals["req"] is not old_arrival
    assert obj.scheduler.inbox.qsize() == 2


@pytest.mark.asyncio
async def test_old_discard_cannot_clear_new_arrival_but_retired_discard_releases_state():
    obj = make_stage(name="orchestrator", is_terminal=True)
    obj._active_requests.add("req")
    old = request()
    await obj._execute(old)
    fresh = request()
    await obj._execute(fresh)
    obj.scheduler.outbox.put(
        OutgoingMessage(
            "req",
            "discard",
            metadata={"arrival_id": old.arrival_id, "keep_active": False},
        )
    )
    await obj._drain_outbox_external()
    assert "req" in obj._active_requests
    obj.scheduler.outbox.put(
        OutgoingMessage(
            "req",
            "discard",
            metadata={"arrival_id": fresh.arrival_id, "keep_active": False},
        )
    )
    await obj._drain_outbox_external()
    assert "req" not in obj._active_requests
    assert not obj._request_arrivals


@pytest.mark.asyncio
async def test_waiting_controller_deadline_reaches_stage_failure_and_abort_cleans_streams():
    controller = UMMController(Adapter())
    obj = make_stage(
        name="orchestrator",
        is_terminal=True,
        get_next=route_umm,
        scheduler=controller,
    )
    obj._send_to_stage = AsyncMock()
    obj.control_plane.send_stream = AsyncMock()
    initial = request()
    obj._active_requests.add("req")
    await obj._execute(initial)
    result = controller._advance(initial)
    await obj._route_result("req", result)
    assert "req" in obj._active_requests
    controller.outbox.put(OutgoingMessage("req", "error", TimeoutError("expired")))
    await obj._drain_outbox_external()
    assert len(obj.control_plane.completions) == 1
    assert obj.control_plane.completions[0].success is False
    obj._on_abort("req")
    await obj._send_stream_to_coordinator("req", {"text": "late"})
    assert obj.control_plane.send_stream.await_count == 0
    assert controller.active_sessions == 0
    assert not obj._request_arrivals


@pytest.mark.asyncio
async def test_cleanup_uses_producing_arrival_when_new_turn_is_already_queued():
    obj = make_stage(name="reasoner", get_next=lambda *_: "orchestrator")
    obj._send_to_stage = AsyncMock()
    old = request()
    obj._active_requests.add("req")
    await obj._execute(old)
    newer = request()
    await obj._execute(newer)
    await obj._route_result("req", old)
    assert "req" in obj._active_requests
    assert obj._request_arrivals["req"] is newer.arrival_id


def test_controller_output_preserves_the_arrival_that_produced_it():
    controller = UMMController(Adapter())
    initial = request()
    initial.arrival_id = object()
    reasoner = controller._advance(initial)
    assert reasoner.arrival_id is initial.arrival_id
    result = reply(reasoner, {"kind": "final", "text": "done"})
    result.arrival_id = object()
    final = controller._advance(result)
    assert final.arrival_id is result.arrival_id


@pytest.mark.asyncio
async def test_legacy_reconstructed_payload_without_arrival_still_cleans_up():
    obj = make_stage()
    obj._active_requests.add("req")
    await obj._execute(request())
    await obj._route_result("req", request())
    assert not obj._active_requests
    assert not obj._request_arrivals


@pytest.mark.parametrize(
    "boundary",
    [
        "start",
        "reasoner_request",
        "generation_request",
        "interpret_reasoner",
        "incorporate_media",
    ],
)
def test_adapter_work_cannot_admit_a_transition_after_its_deadline(
    boundary, monkeypatch
):
    clock = [0.0]
    adapter = Adapter()
    original = getattr(adapter, boundary)

    def slow(*args):
        value = original(*args)
        clock[0] = 2.0
        return value

    monkeypatch.setattr(adapter, boundary, slow)
    controller = UMMController(
        adapter, limits=UMMLimits(timeout_s=1), clock=lambda: clock[0]
    )
    with pytest.raises(TimeoutError, match="deadline"):
        current = controller._advance(request(stream=True))
        if boundary == "interpret_reasoner":
            controller._advance(reply(current, {"kind": "final", "text": "late text"}))
        else:
            current = controller._advance(generate(current))
            controller._advance(reply(current, {"url": "late media"}))
    assert controller.active_sessions == 0
    assert controller.outbox.empty()


@pytest.mark.asyncio
async def test_expired_queued_continuation_is_not_forwarded_and_fails_once():
    clock = [0.0]
    controller = UMMController(
        Adapter(), limits=UMMLimits(timeout_s=1), clock=lambda: clock[0]
    )
    obj = make_stage(
        name="orchestrator",
        scheduler=controller,
        is_terminal=True,
        get_next=route_umm,
    )
    obj._send_to_stage = AsyncMock()
    payload = request()
    obj._active_requests.add("req")
    await obj._execute(payload)
    result = controller._advance(payload)
    controller._emit_result("req", result, controller.outbox)
    clock[0] = 2.0
    controller._expire()
    await obj._drain_outbox_external()
    assert obj._send_to_stage.await_count == 0
    assert len(obj.control_plane.completions) == 1
    assert obj.control_plane.completions[0].success is False
    assert controller.active_sessions == 0


def test_route_validation_expires_once_but_delivers_accepted_final_snapshots():
    clock = [0.0]
    controller = UMMController(
        Adapter(), limits=UMMLimits(timeout_s=1), clock=lambda: clock[0]
    )
    result = controller._advance(request())
    clock[0] = 2.0
    assert controller.validate_result(result) is False
    assert controller.validate_result(result) is False
    controller._expire()
    assert controller.outbox.qsize() == 1
    assert controller.outbox.get_nowait().type == "error"
    current = controller._advance(request("fresh"))
    final = controller._advance(reply(current, {"kind": "final", "text": "done"}))
    clock[0] = 4.0
    assert controller.validate_result(final) is True


@pytest.mark.asyncio
async def test_readmitted_retired_id_fails_at_entry_and_fresh_identity_recovers():
    from sglang_omni.pipeline.coordinator import Coordinator
    from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane

    coordinator = Coordinator(
        "inproc://complete",
        "inproc://abort",
        entry_stage="orchestrator",
        terminal_stages=["orchestrator"],
    )
    control = RecordingCoordinatorControlPlane()
    coordinator.control_plane = control
    coordinator.register_stage("orchestrator", "inproc://orchestrator")
    obj = make_stage(name="orchestrator", is_terminal=True)

    first = asyncio.create_task(coordinator.submit("req", "initial"))
    await asyncio.sleep(0)
    await obj._on_submit(control.submitted[-1][2])
    await coordinator.abort("req")
    obj._on_abort("req")
    with pytest.raises(asyncio.CancelledError):
        await first
    queued = obj.scheduler.inbox.qsize()

    repeated = asyncio.create_task(coordinator.submit("req", "reused"))
    await asyncio.sleep(0)
    await obj._on_submit(control.submitted[-1][2])
    assert obj.scheduler.inbox.qsize() == queued
    await coordinator._handle_completion(obj.control_plane.completions[-1])
    with pytest.raises(RuntimeError, match="fresh request ID"):
        await asyncio.wait_for(repeated, timeout=1)
    completions = len(obj.control_plane.completions)
    await obj._receive_payload_from_stage("req", "reasoner", request())
    assert len(obj.control_plane.completions) == completions
    assert obj.scheduler.inbox.qsize() == queued

    fresh = asyncio.create_task(coordinator.submit("fresh", "healthy"))
    await asyncio.sleep(0)
    message = control.submitted[-1][2]
    await obj._on_submit(message)
    message.data.data = {"text": "healthy"}
    await obj._route_result("fresh", message.data)
    await coordinator._handle_completion(obj.control_plane.completions[-1])
    assert await asyncio.wait_for(fresh, timeout=1) == {"text": "healthy"}
    assert not obj._active_requests
    assert not obj._request_arrivals
