# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavRunResult,
    GraphKey,
)
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
    Code2WavStreamState,
    batched_graph_keys,
    serial_threshold_graph_keys,
    serial_window_frames,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.fixtures.qwen_fakes import FakeCode2WavModel


class FakeGraphRunner:
    def __init__(self, model, keys) -> None:
        self.model = model
        self.keys = set(keys)
        self.calls: list[tuple[tuple[int, ...], bool, str]] = []

    def available_batch_sizes(self, frames: int) -> tuple[int, ...]:
        return tuple(
            sorted(
                {key.batch_size for key in self.keys if key.frames == int(frames)},
                reverse=True,
            )
        )

    def run(self, codes: torch.Tensor, *, eligible: bool) -> Code2WavRunResult:
        key = GraphKey(batch_size=int(codes.shape[0]), frames=int(codes.shape[-1]))
        if eligible and key in self.keys:
            mode, reason = "cuda_graph", None
        elif eligible:
            mode, reason = "eager", "key_miss"
        else:
            mode, reason = "eager", "ineligible"
        self.calls.append((tuple(codes.shape), eligible, mode))
        return Code2WavRunResult(self.model(codes), mode, key, reason)

    def stats(self) -> dict:
        return {
            "enabled": True,
            "disable_reason": None,
            "graph_contract": {"keys": len(self.keys)},
        }


def make_batching_scheduler(**kwargs) -> Code2WavScheduler:
    return Code2WavScheduler(
        FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
        enable_batching=True,
        **kwargs,
    )


def make_chunk_aligned_scheduler(**kwargs) -> Code2WavScheduler:
    model = FakeCode2WavModel(total_upsample=2)
    runner = FakeGraphRunner(model, batched_graph_keys(2, 1, 8))
    return Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
        enable_batching=True,
        enable_cuda_graph=True,
        cuda_graph_runner=runner,
        **kwargs,
    )


def chunk(request_id: str) -> IncomingMessage:
    return IncomingMessage(request_id=request_id, type="stream_chunk", data=None)


def stream_item(code: int, *, stream: bool = True) -> StreamItem:
    return StreamItem(
        0, torch.tensor([code, code * 10]), "talker", metadata={"stream": stream}
    )


def stream_chunk(request_id: str, code: int) -> IncomingMessage:
    return IncomingMessage(
        request_id=request_id,
        type="stream_chunk",
        data=stream_item(code),
    )


def start_scheduler(scheduler: Code2WavScheduler) -> threading.Thread:
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    return thread


def stop_scheduler(scheduler: Code2WavScheduler, thread: threading.Thread) -> None:
    scheduler.stop()
    thread.join(timeout=1)
    assert not thread.is_alive()


def next_stream(scheduler: Code2WavScheduler, request_id: str, *, timeout: float):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"timed out waiting for stream from {request_id}")
        message = scheduler.outbox.get(timeout=remaining)
        if message.type == "stream" and message.request_id == request_id:
            return message


def feed_batch(
    scheduler: Code2WavScheduler,
    entries: list[tuple[str, int]],
    *,
    stream_flags: dict[str, bool] | None = None,
) -> None:
    items = []
    for rid, code in entries:
        stream = True if stream_flags is None else stream_flags[rid]
        items.append((rid, stream_item(code, stream=stream)))
    scheduler.on_stream_chunk_batch(items)


def drain_outbox(scheduler: Code2WavScheduler) -> list:
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    return messages


def test_collector_collects_only_already_queued_chunks() -> None:
    scheduler = make_batching_scheduler()
    scheduler.inbox.put(chunk("req-2"))
    batch = scheduler.collect_stream_chunk_batch(chunk("req-1"))
    assert [m.request_id for m in batch] == ["req-1", "req-2"]


def test_collector_no_wait_when_nothing_due() -> None:
    scheduler = make_batching_scheduler()
    assert scheduler.batch_deadline() is None
    batch = scheduler.collect_stream_chunk_batch(chunk("req-1"))
    assert [m.request_id for m in batch] == ["req-1"]


def test_collector_pushback_non_chunk() -> None:
    scheduler = make_batching_scheduler()
    done = IncomingMessage(request_id="req-1", type="stream_done", data=None)
    scheduler.inbox.put(done)
    batch = scheduler.collect_stream_chunk_batch(chunk("req-1"))
    assert [m.request_id for m in batch] == ["req-1"]
    assert scheduler.pending_messages[0] is done


def test_scheduler_loop_wakes_at_batch_deadline() -> None:
    scheduler = make_batching_scheduler(max_batch_wait_ms=50, batch_floor=2)
    thread = start_scheduler(scheduler)
    try:
        scheduler.inbox.put(stream_chunk("req-1", 1))
        scheduler.inbox.put(stream_chunk("req-1", 2))
        next_stream(scheduler, "req-1", timeout=0.5)

        started = time.monotonic()
        scheduler.inbox.put(stream_chunk("req-1", 3))
        scheduler.inbox.put(stream_chunk("req-1", 4))
        next_stream(scheduler, "req-1", timeout=0.5)
        elapsed = time.monotonic() - started

        assert 0.025 <= elapsed < 0.2
    finally:
        stop_scheduler(scheduler, thread)


def test_old_deadline_does_not_delay_new_first_window() -> None:
    scheduler = make_batching_scheduler(max_batch_wait_ms=300, batch_floor=2)
    thread = start_scheduler(scheduler)
    try:
        scheduler.inbox.put(stream_chunk("req-a", 1))
        scheduler.inbox.put(stream_chunk("req-a", 2))
        next_stream(scheduler, "req-a", timeout=0.5)

        scheduler.inbox.put(stream_chunk("req-a", 3))
        scheduler.inbox.put(stream_chunk("req-a", 4))
        time.sleep(0.02)

        started = time.monotonic()
        scheduler.inbox.put(stream_chunk("req-b", 5))
        scheduler.inbox.put(stream_chunk("req-b", 6))
        next_stream(scheduler, "req-b", timeout=0.5)
        elapsed = time.monotonic() - started

        assert elapsed < 0.15
    finally:
        stop_scheduler(scheduler, thread)


def test_decompose_batch() -> None:
    assert Code2WavScheduler.decompose_batch(1) == [1]
    assert Code2WavScheduler.decompose_batch(3) == [2, 1]
    assert Code2WavScheduler.decompose_batch(5) == [4, 1]
    assert Code2WavScheduler.decompose_batch(6) == [4, 2]
    assert Code2WavScheduler.decompose_batch(7) == [4, 2, 1]
    assert Code2WavScheduler.decompose_batch(8) == [8]


def test_decompose_batch_against_published_sizes() -> None:
    decompose = Code2WavScheduler.decompose_batch
    assert decompose(7, (4, 2, 1)) == [4, 2, 1]
    assert decompose(8, (4, 1)) == [4, 4]
    assert decompose(7, (4,)) == [4, 3]
    assert decompose(5, (8,)) == [5]
    assert decompose(3, ()) == [3]


class AvailabilityRunner:
    def __init__(self, sizes: tuple[int, ...]) -> None:
        self.sizes = sizes
        self.queries: list[int] = []

    def available_batch_sizes(self, frames: int) -> tuple[int, ...]:
        self.queries.append(frames)
        return self.sizes


def states_with_ready(count: int, ready: int) -> list[tuple[str, Code2WavStreamState]]:
    participants = []
    for i in range(count):
        state = Code2WavStreamState()
        state.chunks = [torch.tensor([0, 0]) for _ in range(ready)]
        participants.append((f"r{i}", state))
    return participants


def test_eager_step_plan_is_one_forward() -> None:
    scheduler = make_batching_scheduler()
    assert scheduler.cuda_graph_runner is None
    participants = [(f"r{i}", Code2WavStreamState()) for i in range(7)]
    assert scheduler.build_step_plan(participants) == [7]


def test_step_plan_follows_runner_availability_for_the_window() -> None:
    runner = AvailabilityRunner((4, 2, 1))
    scheduler = make_batching_scheduler(
        enable_cuda_graph=True,
        cuda_graph_runner=runner,
    )
    participants = states_with_ready(7, ready=6)
    assert scheduler.build_step_plan(participants) == [4, 2, 1]
    # Note (ruoyu): the plan must query the chunk-capped window, not the raw
    # backlog depth — an uncapped query would miss the captured key set.
    assert runner.queries[-1] == 2


def test_step_plan_without_published_graphs_stays_one_eager_forward() -> None:
    runner = AvailabilityRunner(())
    scheduler = make_batching_scheduler(
        enable_cuda_graph=True,
        cuda_graph_runner=runner,
    )
    participants = states_with_ready(7, ready=6)
    assert scheduler.build_step_plan(participants) == [7]


def test_five_streams_take_one_forward_not_two() -> None:
    scheduler = make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    rids = [f"req-{i}" for i in range(5)]
    feed_batch(scheduler, [(rid, 1) for rid in rids])
    feed_batch(scheduler, [(rid, 2) for rid in rids])
    assert scheduler.model.calls == [(5, 2, 2)]
    for rid in rids:
        assert scheduler.stream_states[rid].emitted == 2


def test_first_window_fires_immediately() -> None:
    scheduler = make_batching_scheduler(max_batch_wait_ms=1000, batch_floor=4)
    feed_batch(scheduler, [("req-1", 1), ("req-1", 2)])
    messages = drain_outbox(scheduler)
    assert [m.type for m in messages] == ["stream"]
    assert scheduler.model.calls == [(1, 2, 2)]


def test_floor_fires_without_deadline() -> None:
    scheduler = make_batching_scheduler(max_batch_wait_ms=1000, batch_floor=2)
    feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    feed_batch(scheduler, [("req-b", 3), ("req-b", 4)])
    drain_outbox(scheduler)
    feed_batch(scheduler, [("req-a", 5), ("req-a", 6), ("req-b", 7), ("req-b", 8)])
    messages = drain_outbox(scheduler)
    assert sorted(m.request_id for m in messages) == ["req-a", "req-b"]
    assert scheduler.model.calls == [(1, 2, 2), (1, 2, 2), (2, 2, 3)]


def test_deadline_fires_single() -> None:
    scheduler = make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    feed_batch(scheduler, [("req-1", 1), ("req-1", 2)])
    drain_outbox(scheduler)
    feed_batch(scheduler, [("req-1", 3), ("req-1", 4)])
    messages = drain_outbox(scheduler)
    assert [m.request_id for m in messages] == ["req-1"]
    assert scheduler.model.calls == [(1, 2, 2), (1, 2, 3)]


def test_bucket_isolation() -> None:
    scheduler = make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    feed_batch(scheduler, [("req-b", 3), ("req-b", 4)])
    drain_outbox(scheduler)
    feed_batch(
        scheduler,
        [
            ("req-a", 5),
            ("req-a", 6),
            ("req-b", 7),
            ("req-b", 8),
            ("req-b", 9),
            ("req-b", 10),
        ],
    )
    steady_calls = scheduler.model.calls[2:]
    assert all(call[0] == 1 for call in steady_calls)
    assert sorted(steady_calls) == [(1, 2, 3), (1, 2, 5)]
    assert scheduler.stream_states["req-a"].emitted == 4
    assert scheduler.stream_states["req-b"].emitted == 6


def test_step_cursor_uses_captured_window_end() -> None:
    scheduler = make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    feed_batch(scheduler, [("req-1", 1), ("req-1", 2)])
    drain_outbox(scheduler)

    state = scheduler.stream_states["req-1"]
    real_forward = scheduler.forward_codes

    def forward_then_ingest(codes, **kwargs):
        result = real_forward(codes, **kwargs)
        state.chunks.append(torch.tensor([9, 90]))
        return result

    scheduler.forward_codes = forward_then_ingest
    feed_batch(scheduler, [("req-1", 3), ("req-1", 4)])

    assert state.emitted == 4
    assert len(state.chunks) == 5
    assert scheduler.ready(state) == 1


def test_bitwise_equivalence() -> None:
    schedule = {
        "req-1": [1, 2, 3, 4, 5, 6],
        "req-2": [7, 8, 9, 10, 11, 12],
        "req-3": [13, 14, 15, 16, 17, 18],
    }

    control = Code2WavScheduler(
        FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
    )
    for rid, codes in schedule.items():
        for code in codes:
            control.handle_stream_chunk(rid, stream_item(code))

    batched = make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    for round_start in range(0, 6, 2):
        entries = []
        for rid, codes in schedule.items():
            entries.append((rid, codes[round_start]))
            entries.append((rid, codes[round_start + 1]))
        feed_batch(batched, entries)

    assert any(call[0] > 1 for call in batched.model.calls)
    for rid in schedule:
        control_state = control.stream_states[rid]
        batched_state = batched.stream_states[rid]
        assert batched_state.emitted == 6
        assert np.array_equal(
            np.concatenate(control_state.audio_parts),
            np.concatenate(batched_state.audio_parts),
        )


def test_mixed_stream_enabled() -> None:
    scheduler = make_batching_scheduler()
    feed_batch(
        scheduler,
        [("req-a", 1), ("req-a", 2), ("req-b", 3), ("req-b", 4)],
        stream_flags={"req-a": True, "req-b": False},
    )
    messages = drain_outbox(scheduler)
    assert [(m.type, m.request_id) for m in messages] == [("stream", "req-a")]
    assert scheduler.model.calls == [(2, 2, 2)]
    for rid in ("req-a", "req-b"):
        state = scheduler.stream_states[rid]
        assert state.emitted == 2
        assert len(state.audio_parts) == 1


def test_step_failure_isolates_participants() -> None:
    scheduler = make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    feed_batch(scheduler, [("req-b", 3), ("req-b", 4)])
    drain_outbox(scheduler)

    real_forward = scheduler.forward_codes

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    scheduler.forward_codes = boom
    feed_batch(
        scheduler,
        [
            ("req-a", 5),
            ("req-a", 6),
            ("req-b", 7),
            ("req-b", 8),
            ("req-c", 9),
        ],
    )
    assert "req-a" not in scheduler.stream_states
    assert "req-b" not in scheduler.stream_states
    assert scheduler.is_aborted("req-a") and scheduler.is_aborted("req-b")
    assert "req-c" in scheduler.stream_states

    scheduler.forward_codes = real_forward
    drain_outbox(scheduler)
    feed_batch(scheduler, [("req-c", 10)])
    messages = drain_outbox(scheduler)
    assert [(m.type, m.request_id) for m in messages] == [("stream", "req-c")]
    assert scheduler.stream_states["req-c"].emitted == 2


def test_step_failure_after_success_keeps_decoded_sub_batches() -> None:
    scheduler = make_chunk_aligned_scheduler(max_batch_wait_ms=0, batch_floor=2)
    cleaned: list[str] = []
    scheduler.cleanup_aborted_request = cleaned.append

    real_forward = scheduler.forward_codes
    forwards = 0

    def fail_on_second_sub_batch(codes, **kwargs):
        nonlocal forwards
        forwards += 1
        if forwards == 2:
            raise RuntimeError("boom")
        return real_forward(codes, **kwargs)

    scheduler.forward_codes = fail_on_second_sub_batch
    feed_batch(
        scheduler,
        [(rid, code) for rid in ("req-a", "req-b", "req-c") for code in (1, 2)],
    )

    # Note (ruoyu): plan [2, 1] — the size-2 sub-batch decoded before the
    # size-1 one failed, so its audio must survive the failure.
    messages = drain_outbox(scheduler)
    assert [m.request_id for m in messages if m.type == "stream"] == [
        "req-a",
        "req-b",
    ]
    assert [m.request_id for m in messages if m.type == "error"] == ["req-c"]
    for rid in ("req-a", "req-b"):
        assert scheduler.stream_states[rid].emitted == 2
        assert not scheduler.is_aborted(rid)
    assert "req-c" not in scheduler.stream_states
    assert scheduler.is_aborted("req-c")
    assert cleaned == ["req-c"]
    assert scheduler.pending_step_failures == []

    scheduler.forward_codes = real_forward
    feed_batch(scheduler, [("req-a", 3), ("req-a", 4)])
    assert [(m.type, m.request_id) for m in drain_outbox(scheduler)] == [
        ("stream", "req-a")
    ]
    assert scheduler.stream_states["req-a"].emitted == 4


def test_one_participation_per_pump() -> None:
    scheduler = make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    feed_batch(scheduler, [("req-1", 1), ("req-1", 2)])
    drain_outbox(scheduler)

    selections: list[list[str]] = []
    original_select = scheduler.select_step_participants

    def recording_select():
        participants = original_select()
        if participants:
            selections.append([rid for rid, _ in participants])
        return participants

    scheduler.select_step_participants = recording_select
    feed_batch(scheduler, [("req-1", 3), ("req-1", 4), ("req-1", 5), ("req-1", 6)])
    assert selections == [["req-1"]]
    state = scheduler.stream_states["req-1"]
    assert state.emitted == 6
    assert len(state.chunks) - state.emitted == 0


def test_factory_flags_reach_scheduler(monkeypatch) -> None:
    import sglang_omni.models.qwen3_omni.components.code2wav_scheduler as mod

    monkeypatch.setattr(
        mod,
        "load_code2wav_model",
        lambda path, *, device, dtype: FakeCode2WavModel(total_upsample=2),
    )
    scheduler = mod.create_code2wav_scheduler(
        "fake-path",
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        enable_batching=True,
        max_batch_wait_ms=250,
        batch_floor=3,
        batch_ceiling=4,
    )
    assert scheduler.enable_batching is True
    assert scheduler.max_batch_wait_s == 0.25
    assert scheduler.batch_floor == 3
    assert scheduler.batch_ceiling == 4
    assert scheduler.can_batch_stream_chunks is True


def test_forward_codes_eager() -> None:
    scheduler = make_batching_scheduler()
    codes = torch.zeros(1, 2, 2, dtype=torch.long)
    _, meta = scheduler.forward_codes(codes)
    assert meta == {
        "execution_mode": "eager",
        "graph_key": None,
        "fallback_reason": None,
    }


def test_batch_events_emitted(monkeypatch) -> None:
    import sglang_omni.models.qwen3_omni.components.code2wav_scheduler as mod

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        mod,
        "_emit_event",
        lambda **kw: events.append((kw["event_name"], kw["metadata"])),
    )

    class ActiveRecorder:
        def is_active(self) -> bool:
            return True

    monkeypatch.setattr(mod, "_get_recorder", lambda: ActiveRecorder())

    scheduler = make_batching_scheduler(max_batch_wait_ms=1000, batch_floor=2)
    feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    feed_batch(scheduler, [("req-b", 3), ("req-b", 4)])
    drain_outbox(scheduler)
    events.clear()

    feed_batch(scheduler, [("req-a", 5), ("req-a", 6), ("req-b", 7), ("req-b", 8)])

    batch_events = [
        (name, meta)
        for name, meta in events
        if name in ("code2wav_batch_start", "code2wav_batch_end")
    ]
    assert [name for name, _ in batch_events] == [
        "code2wav_batch_start",
        "code2wav_batch_end",
    ]
    start_meta = batch_events[0][1]
    end_meta = batch_events[1][1]
    assert start_meta["batch_id"] == end_meta["batch_id"]
    assert start_meta["participant_request_ids"] == ["req-a", "req-b"]
    assert end_meta["participant_request_ids"] == ["req-a", "req-b"]
    assert start_meta["first_audio_request_ids"] == []
    assert start_meta["fire_reason"] == "floor"
    assert start_meta["batch_size"] == 2
    assert start_meta["subbatch_decomposition"] == [2]
    assert start_meta["bucket"] == [1, 3]
    assert start_meta["due_bucket_count"] == 1
    assert end_meta["audio_samples"] > 0
    assert end_meta["execution_mode"] == "eager"
    assert end_meta["sub_batch_execution"] == [
        {
            "batch_size": 2,
            "execution_mode": "eager",
            "graph_key": None,
            "fallback_reason": None,
        }
    ]


def test_first_window_ingest_events_are_bounded_and_exclude_eos(monkeypatch) -> None:
    events = []
    recorder = SimpleNamespace(is_active=lambda: True, active_run_id=lambda: "run-a")
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.components.code2wav_scheduler"
        "._get_event_recorder",
        lambda: recorder,
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.components.code2wav_scheduler._emit_event",
        lambda **kw: events.append(kw),
    )
    scheduler = make_batching_scheduler(initial_codec_chunk_frames=2)
    state = Code2WavStreamState()
    for code in (2150, 1, 2, 3, 4):
        scheduler.ingest("req-a", state, torch.tensor([code, code * 10]))

    assert [event["event_name"] for event in events] == [
        "code2wav_first_ingest",
        "code2wav_first_window_ready",
    ]
    first, ready = events
    assert first["request_id"] == ready["request_id"] == "req-a"
    assert first["timestamp_ns"] > 0
    assert first["metadata"]["accepted_frames"] == 0
    assert ready["metadata"]["messages"] == 3
    assert ready["metadata"]["accepted_frames"] == 2
    assert ready["metadata"]["ready_frames"] == 2
    assert ready["metadata"]["threshold_frames"] == 2
    assert ready["metadata"]["eos_checks"] == 3
    assert ready["metadata"]["eos_check_host_ns"] >= 0
    assert ready["metadata"]["ingest_host_ns"] >= ready["metadata"]["eos_check_host_ns"]
    assert [row[0].item() for row in state.chunks] == [1, 2, 3, 4]


def test_coalesced_first_window_profile_resets_on_new_run(monkeypatch) -> None:
    events = []
    current = SimpleNamespace(run_id="run-a")
    recorder = SimpleNamespace(
        is_active=lambda: True, active_run_id=lambda: current.run_id
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.components.code2wav_scheduler"
        "._get_event_recorder",
        lambda: recorder,
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.components.code2wav_scheduler._emit_event",
        lambda **kw: events.append(kw),
    )
    scheduler = make_batching_scheduler(initial_codec_chunk_frames=2)
    state = Code2WavStreamState()
    scheduler.ingest("req-a", state, torch.tensor([[1, 10], [2, 20]]))
    assert events[-1]["metadata"]["messages"] == 1
    assert events[-1]["metadata"]["accepted_frames"] == 2
    assert events[-1]["metadata"]["eos_checks"] == 0

    current.run_id = "run-b"
    scheduler.ingest("req-a", state, torch.tensor([[3, 30]]))
    assert len(events) == 4
    assert events[-1]["metadata"]["started_with_frames"] == 2
    assert events[-1]["metadata"]["messages"] == 1
    assert state.checked == 3


def test_ingest_without_recorder_does_not_read_profile_clocks(monkeypatch) -> None:
    def unexpected():
        raise AssertionError("inactive profiling must not read a profile clock")

    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.components.code2wav_scheduler"
        "._get_event_recorder",
        lambda: SimpleNamespace(is_active=lambda: False),
    )
    scheduler = make_batching_scheduler()
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.components.code2wav_scheduler.time",
        SimpleNamespace(time_ns=unexpected, perf_counter_ns=unexpected),
    )
    state = Code2WavStreamState()
    scheduler.ingest("req-a", state, torch.tensor([1, 10]))
    scheduler.ingest("req-a", state, torch.tensor([2150, 0]))
    assert len(state.chunks) == 1
    assert (
        state._critical_ingest_profile is None
    )  # noqa: leading-underscore  # production name


def test_batching_and_cuda_graph_coexist() -> None:
    scheduler = make_chunk_aligned_scheduler()
    assert scheduler.chunk_aligned_dispatch is True
    assert scheduler.cuda_graph_runner is not None
    legacy = make_batching_scheduler()
    assert legacy.chunk_aligned_dispatch is False


def ready_participants(n: int) -> list[tuple[str, Code2WavStreamState]]:
    participants = []
    for i in range(n):
        state = Code2WavStreamState()
        state.chunks = [torch.tensor([i, i * 10]), torch.tensor([i, i * 10])]
        participants.append((f"r{i}", state))
    return participants


def test_chunk_aligned_step_plan_decomposes() -> None:
    scheduler = make_chunk_aligned_scheduler()
    assert scheduler.build_step_plan(ready_participants(7)) == [4, 2, 1]


def test_chunk_aligned_backlog_drains_in_uniform_graph_windows() -> None:
    scheduler = make_chunk_aligned_scheduler(max_batch_wait_ms=0, batch_floor=2)
    feed_batch(scheduler, [("req-1", code) for code in (1, 2, 3, 4, 5, 6)])
    assert scheduler.model.calls == [(1, 2, 2), (1, 2, 3), (1, 2, 3)]
    assert scheduler.stream_states["req-1"].emitted == 6
    runner = scheduler.cuda_graph_runner
    assert [mode for _, _, mode in runner.calls] == ["cuda_graph"] * 3


def test_chunk_aligned_buckets_merge_mixed_backlogs() -> None:
    scheduler = make_chunk_aligned_scheduler(max_batch_wait_ms=0, batch_floor=2)
    feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    feed_batch(scheduler, [("req-b", 3), ("req-b", 4)])
    drain_outbox(scheduler)
    feed_batch(
        scheduler,
        [
            ("req-a", 5),
            ("req-a", 6),
            ("req-b", 7),
            ("req-b", 8),
            ("req-b", 9),
            ("req-b", 10),
        ],
    )
    # Note (ruoyu): legacy buckets isolate ready=2 from ready=4 (see
    # test_bucket_isolation); chunk-aligned buckets collapse to
    # (context, context+chunk) and merge them.
    assert scheduler.model.calls[2:] == [(2, 2, 3), (1, 2, 3)]
    assert scheduler.stream_states["req-a"].emitted == 4
    assert scheduler.stream_states["req-b"].emitted == 6
    runner = scheduler.cuda_graph_runner
    assert [(call[1], call[2]) for call in runner.calls[-2:]] == [
        (True, "cuda_graph"),
        (True, "cuda_graph"),
    ]


def test_serial_graph_keys_follow_initial_chunk_offset() -> None:
    assert serial_window_frames(10, 25) == (10, 20, 30, 35)
    assert serial_window_frames(10, 25, 2) == (2, 12, 22, 32, 35)
    assert {key.frames for key in batched_graph_keys(10, 25, 8, 2)} == {
        2,
        12,
        22,
        32,
        35,
    }


def test_large_batch_classes_stay_on_the_early_windows() -> None:
    by_frames: dict[int, set[int]] = {}
    for key in batched_graph_keys(10, 25, 16, 2):
        by_frames.setdefault(key.frames, set()).add(key.batch_size)
    assert by_frames[2] == {1, 2, 4, 8, 16}
    assert by_frames[12] == {1, 2, 4, 8, 16}
    assert by_frames[22] == {1, 2, 4, 8}
    assert by_frames[35] == {1, 2, 4, 8}
    assert {key.batch_size for key in batched_graph_keys(10, 25, 8, 2)} == {1, 2, 4, 8}


def test_pinned_slot_pool_covers_a_coalesced_step() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=10,
        left_context_size=25,
        sample_rate=24000,
        enable_batching=True,
        batch_ceiling=16,
    )
    assert scheduler.max_pinned_slots == scheduler.MAX_PINNED_SLOTS + 16


def test_bucket_batch_ceiling_is_per_window() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=10,
        left_context_size=25,
        sample_rate=24000,
        enable_batching=True,
        enable_cuda_graph=True,
        initial_codec_chunk_frames=2,
        batch_ceiling=16,
        cuda_graph_runner=FakeGraphRunner(model, batched_graph_keys(10, 25, 16, 2)),
    )
    assert scheduler.bucket_batch_ceiling(2) == 16
    assert scheduler.bucket_batch_ceiling(12) == 16
    assert scheduler.bucket_batch_ceiling(35) == 8
    assert scheduler.bucket_batch_ceiling(7) == 8


def test_bucket_batch_ceiling_honours_a_lower_configured_ceiling() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=10,
        left_context_size=25,
        sample_rate=24000,
        enable_batching=True,
        enable_cuda_graph=True,
        initial_codec_chunk_frames=2,
        batch_ceiling=4,
        cuda_graph_runner=FakeGraphRunner(model, batched_graph_keys(10, 25, 4, 2)),
    )
    assert scheduler.bucket_batch_ceiling(2) == 4
    assert scheduler.bucket_batch_ceiling(35) == 4


def test_batched_graph_keys_cover_decompose_sizes() -> None:
    keys = batched_graph_keys(2, 1, 8)
    assert set(keys) == {
        GraphKey(batch_size=size, frames=frames)
        for size in (1, 2, 4, 8)
        for frames in (2, 3)
    }
    capped = batched_graph_keys(2, 1, 4)
    assert {key.batch_size for key in capped} == {1, 2, 4}


def test_factory_builds_batched_keys_with_batching(monkeypatch) -> None:
    import sglang_omni.models.qwen3_omni.components.code2wav_scheduler as mod

    def fake_load(path, *, device, dtype):
        model = FakeCode2WavModel(total_upsample=2)
        model.config = SimpleNamespace(num_quantizers=2)
        return model

    monkeypatch.setattr(mod, "load_code2wav_model", fake_load)
    captured: dict = {}

    class FakeRunnerCls:
        @classmethod
        def build(cls, model, **kwargs):
            captured.update(kwargs)
            return FakeGraphRunner(model, kwargs["graph_keys"])

    monkeypatch.setattr(mod, "Code2WavCudaGraphRunner", FakeRunnerCls)
    scheduler = mod.create_code2wav_scheduler(
        "fake-path",
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        enable_batching=True,
        enable_cuda_graph=True,
        batch_ceiling=4,
        total_gpu_memory_fraction=0.05,
    )
    keys = captured["graph_keys"]
    assert GraphKey(batch_size=1, frames=3) in keys
    assert GraphKey(batch_size=2, frames=3) in keys
    assert GraphKey(batch_size=4, frames=2) in keys
    assert all(key.batch_size <= 4 for key in keys)
    assert scheduler.chunk_aligned_dispatch is True


def test_serial_only_runner_splits_groups_into_safe_b1_replays() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    runner = FakeGraphRunner(model, serial_threshold_graph_keys(2, 1))
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
        enable_batching=True,
        enable_cuda_graph=True,
        cuda_graph_runner=runner,
    )
    # Note (ruoyu): a serial-only runner may have dropped batched graphs after
    # their eager warmup OOMed, so retrying the group as one eager forward is
    # unsafe even when it benchmarks faster in the non-OOM case.
    assert scheduler.chunk_aligned_dispatch is True
    assert scheduler.build_step_plan(ready_participants(7)) == [1] * 7


def test_runtime_disable_stops_chunk_aligned_dispatch() -> None:
    scheduler = make_chunk_aligned_scheduler()
    assert scheduler.chunk_aligned_dispatch is True
    scheduler.cuda_graph_runner.keys = set()
    assert scheduler.chunk_aligned_dispatch is False
    participants = [(f"r{i}", Code2WavStreamState()) for i in range(3)]
    assert scheduler.build_step_plan(participants) == [3]


def test_chunk_aligned_groups_replay_batched_graphs() -> None:
    scheduler = make_chunk_aligned_scheduler(max_batch_wait_ms=0, batch_floor=2)
    feed_batch(
        scheduler,
        [(rid, code) for rid in ("req-a", "req-b") for code in (1, 2, 3, 4)],
    )
    runner = scheduler.cuda_graph_runner
    batched_calls = [call for call in runner.calls if call[0][0] > 1]
    assert batched_calls
    assert all(mode == "cuda_graph" for _, _, mode in batched_calls)


def test_chunk_aligned_waveforms_match_serial_reference() -> None:
    schedule = {
        "req-1": [1, 2, 3, 4, 5, 6],
        "req-2": [7, 8, 9, 10, 11, 12],
    }

    control = Code2WavScheduler(
        FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
    )
    for rid, codes in schedule.items():
        for code in codes:
            control.handle_stream_chunk(rid, stream_item(code))

    quantized = make_chunk_aligned_scheduler(max_batch_wait_ms=0, batch_floor=2)
    feed_batch(
        quantized,
        [(rid, code) for rid, codes in schedule.items() for code in codes],
    )

    assert any(call[0] > 1 for call in quantized.model.calls)
    for rid in schedule:
        assert quantized.stream_states[rid].emitted == 6
        assert np.array_equal(
            np.concatenate(control.stream_states[rid].audio_parts),
            np.concatenate(quantized.stream_states[rid].audio_parts),
        )


def test_qwen_code2wav_run_step_emits_full_chunk_despite_output_deficit() -> None:
    scheduler = Code2WavScheduler(
        FakeCode2WavModel(total_upsample=2, output_deficit=1),
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
        enable_batching=True,
    )
    state = Code2WavStreamState(stream_enabled=True)
    state.chunks = [torch.tensor([c, c * 10]) for c in (1, 2, 3)]
    state.emitted = 1
    state.audio_parts = [np.zeros(1, dtype=np.float32)]
    scheduler.stream_states["req-1"] = state

    decoded = scheduler.run_step([("req-1", state)], [1])
    assert decoded["req-1"].shape == (4,)
    assert state.emitted == 3


class StubGraphRunner:
    """Replays through the eager model but reports cuda_graph execution, and
    misses (eager fallback) for batch sizes it does not publish."""

    def __init__(self, model, sizes: tuple[int, ...]) -> None:
        self.model = model
        self.sizes = sizes
        self.run_calls: list[tuple[tuple[int, ...], bool]] = []

    def available_batch_sizes(self, frames: int) -> tuple[int, ...]:
        del frames
        return self.sizes

    def run(self, codes: torch.Tensor, *, eligible: bool = True) -> Code2WavRunResult:
        self.run_calls.append((tuple(codes.shape), eligible))
        key = GraphKey(batch_size=int(codes.shape[0]), frames=int(codes.shape[2]))
        hit = eligible and key.batch_size in self.sizes
        return Code2WavRunResult(
            output=self.model(codes),
            execution_mode="cuda_graph" if hit else "eager",
            key=key,
            fallback_reason=None if hit else "key_miss",
        )


def make_graph_batching_scheduler(
    sizes: tuple[int, ...], **kwargs
) -> tuple[Code2WavScheduler, StubGraphRunner]:
    model = FakeCode2WavModel(total_upsample=2)
    runner = StubGraphRunner(model, sizes)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
        enable_batching=True,
        enable_cuda_graph=True,
        cuda_graph_runner=runner,
        **kwargs,
    )
    return scheduler, runner


def test_batched_step_replays_one_graph_when_size_is_published() -> None:
    scheduler, runner = make_graph_batching_scheduler((8, 4, 2, 1))
    feed_batch(
        scheduler,
        [("req-a", 1), ("req-a", 2), ("req-b", 3), ("req-b", 4)],
    )

    assert runner.run_calls == [((2, 2, 2), True)]
    messages = drain_outbox(scheduler)
    assert sorted(m.request_id for m in messages) == ["req-a", "req-b"]


def test_batched_step_replays_b1_graphs_without_batched_sizes() -> None:
    # Note (ruoyu): a serial-only runner can mean batched eager warmup OOMed,
    # so the plan must stay within its published B1 capacity.
    scheduler, runner = make_graph_batching_scheduler((1,))
    feed_batch(
        scheduler,
        [("req-a", 1), ("req-a", 2), ("req-b", 3), ("req-b", 4)],
    )

    assert runner.run_calls == [((1, 2, 2), True), ((1, 2, 2), True)]
    messages = drain_outbox(scheduler)
    assert sorted(m.request_id for m in messages) == ["req-a", "req-b"]


def test_batch_end_event_reports_mixed_sub_batch_execution(monkeypatch) -> None:
    import sglang_omni.models.qwen3_omni.components.code2wav_scheduler as mod

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        mod,
        "_emit_event",
        lambda **kw: events.append((kw["event_name"], kw["metadata"])),
    )

    class ActiveRecorder:
        def is_active(self) -> bool:
            return True

    monkeypatch.setattr(mod, "_get_recorder", lambda: ActiveRecorder())

    scheduler, runner = make_graph_batching_scheduler((2,))
    feed_batch(
        scheduler,
        [
            ("req-a", 1),
            ("req-a", 2),
            ("req-b", 3),
            ("req-b", 4),
            ("req-c", 5),
            ("req-c", 6),
        ],
    )

    assert runner.run_calls == [((2, 2, 2), True), ((1, 2, 2), True)]
    end_meta = next(meta for name, meta in events if name == "code2wav_batch_end")
    assert end_meta["execution_mode"] == "mixed"
    assert end_meta["sub_batch_execution"] == [
        {
            "batch_size": 2,
            "execution_mode": "cuda_graph",
            "graph_key": {"batch_size": 2, "frames": 2},
            "fallback_reason": None,
        },
        {
            "batch_size": 1,
            "execution_mode": "eager",
            "graph_key": {"batch_size": 1, "frames": 2},
            "fallback_reason": "key_miss",
        },
    ]
    assert end_meta["subbatch_decomposition"] == [2, 1]


def test_initial_codec_chunk_frames_fires_first_window_early() -> None:
    scheduler = make_batching_scheduler(
        max_batch_wait_ms=0, batch_floor=2, initial_codec_chunk_frames=1
    )
    feed_batch(scheduler, [("req-1", 1)])
    messages = drain_outbox(scheduler)
    assert [m.type for m in messages] == ["stream"]
    assert scheduler.model.calls == [(1, 2, 1)]
    assert scheduler.stream_states["req-1"].emitted == 1
    feed_batch(scheduler, [("req-1", 2)])
    assert drain_outbox(scheduler) == []
    feed_batch(scheduler, [("req-1", 3)])
    assert [m.type for m in drain_outbox(scheduler)] == ["stream"]
    assert scheduler.stream_states["req-1"].emitted == 3


def test_initial_codec_chunk_frames_zero_keeps_steady_threshold() -> None:
    scheduler = make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    feed_batch(scheduler, [("req-1", 1)])
    assert drain_outbox(scheduler) == []
    feed_batch(scheduler, [("req-1", 2)])
    assert [m.type for m in drain_outbox(scheduler)] == ["stream"]


def make_done(request_id: str) -> IncomingMessage:
    return IncomingMessage(request_id=request_id, type="stream_done", data=None)


def test_next_message_ingests_first_chunks_before_other_messages() -> None:
    scheduler = make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    drain_outbox(scheduler)
    scheduler.inbox.put(make_done("req-a"))
    scheduler.inbox.put(stream_chunk("req-b", 5))
    scheduler.inbox.put(stream_chunk("req-b", 6))

    msg = scheduler.next_message()

    assert msg is not None and msg.type == "stream_done"
    assert msg.request_id == "req-a"
    state = scheduler.stream_states["req-b"]
    assert state.emitted == 2
    assert len(state.audio_parts) == 1


def test_next_message_keeps_steady_chunks_in_fifo_order() -> None:
    scheduler = make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    drain_outbox(scheduler)
    scheduler.inbox.put(make_done("req-a"))
    scheduler.inbox.put(stream_chunk("req-a", 3))
    scheduler.inbox.put(stream_chunk("req-a", 4))

    msg = scheduler.next_message()

    assert msg is not None and msg.type == "stream_done"
    assert len(scheduler.stream_states["req-a"].chunks) == 2

    batches: list[int] = []
    original = scheduler.on_stream_chunk_batch

    def recording(items):
        batches.append(len(items))
        return original(items)

    scheduler.on_stream_chunk_batch = recording
    assert scheduler.next_message() is None
    assert batches == [2]
    assert scheduler.stream_states["req-a"].emitted == 4
