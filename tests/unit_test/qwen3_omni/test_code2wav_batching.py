# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
    Code2WavStreamState,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.messages import IncomingMessage
from tests.unit_test.fixtures.qwen_fakes import FakeCode2WavModel


def _make_batching_scheduler(**kwargs) -> Code2WavScheduler:
    return Code2WavScheduler(
        FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
        enable_batching=True,
        **kwargs,
    )


def _chunk(request_id: str) -> IncomingMessage:
    return IncomingMessage(request_id=request_id, type="stream_chunk", data=None)


def _stream_item(code: int, *, stream: bool = True) -> StreamItem:
    return StreamItem(
        0, torch.tensor([code, code * 10]), "talker", metadata={"stream": stream}
    )


def _stream_chunk(request_id: str, code: int) -> IncomingMessage:
    return IncomingMessage(
        request_id=request_id,
        type="stream_chunk",
        data=_stream_item(code),
    )


def _start_scheduler(scheduler: Code2WavScheduler) -> threading.Thread:
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    return thread


def _stop_scheduler(scheduler: Code2WavScheduler, thread: threading.Thread) -> None:
    scheduler.stop()
    thread.join(timeout=1)
    assert not thread.is_alive()


def _next_stream(scheduler: Code2WavScheduler, request_id: str, *, timeout: float):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"timed out waiting for stream from {request_id}")
        message = scheduler.outbox.get(timeout=remaining)
        if message.type == "stream" and message.request_id == request_id:
            return message


def _feed_batch(
    scheduler: Code2WavScheduler,
    entries: list[tuple[str, int]],
    *,
    stream_flags: dict[str, bool] | None = None,
) -> None:
    items = []
    for rid, code in entries:
        stream = True if stream_flags is None else stream_flags[rid]
        items.append((rid, _stream_item(code, stream=stream)))
    scheduler.on_stream_chunk_batch(items)


def _drain_outbox(scheduler: Code2WavScheduler) -> list:
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    return messages


def test_collector_collects_only_already_queued_chunks() -> None:
    scheduler = _make_batching_scheduler()
    scheduler.inbox.put(_chunk("req-2"))
    batch = scheduler._collect_stream_chunk_batch(_chunk("req-1"))
    assert [m.request_id for m in batch] == ["req-1", "req-2"]


def test_collector_no_wait_when_nothing_due() -> None:
    scheduler = _make_batching_scheduler()
    assert scheduler._batch_deadline() is None
    batch = scheduler._collect_stream_chunk_batch(_chunk("req-1"))
    assert [m.request_id for m in batch] == ["req-1"]


def test_collector_pushback_non_chunk() -> None:
    scheduler = _make_batching_scheduler()
    done = IncomingMessage(request_id="req-1", type="stream_done", data=None)
    scheduler.inbox.put(done)
    batch = scheduler._collect_stream_chunk_batch(_chunk("req-1"))
    assert [m.request_id for m in batch] == ["req-1"]
    assert scheduler._pending_messages[0] is done


def test_scheduler_loop_wakes_at_batch_deadline() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=50, batch_floor=2)
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(_stream_chunk("req-1", 1))
        scheduler.inbox.put(_stream_chunk("req-1", 2))
        _next_stream(scheduler, "req-1", timeout=0.5)

        started = time.monotonic()
        scheduler.inbox.put(_stream_chunk("req-1", 3))
        scheduler.inbox.put(_stream_chunk("req-1", 4))
        _next_stream(scheduler, "req-1", timeout=0.5)
        elapsed = time.monotonic() - started

        assert 0.025 <= elapsed < 0.2
    finally:
        _stop_scheduler(scheduler, thread)


def test_old_deadline_does_not_delay_new_first_window() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=300, batch_floor=2)
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(_stream_chunk("req-a", 1))
        scheduler.inbox.put(_stream_chunk("req-a", 2))
        _next_stream(scheduler, "req-a", timeout=0.5)

        scheduler.inbox.put(_stream_chunk("req-a", 3))
        scheduler.inbox.put(_stream_chunk("req-a", 4))
        time.sleep(0.02)

        started = time.monotonic()
        scheduler.inbox.put(_stream_chunk("req-b", 5))
        scheduler.inbox.put(_stream_chunk("req-b", 6))
        _next_stream(scheduler, "req-b", timeout=0.5)
        elapsed = time.monotonic() - started

        assert elapsed < 0.15
    finally:
        _stop_scheduler(scheduler, thread)


def test_decompose_batch() -> None:
    assert Code2WavScheduler._decompose_batch(1) == [1]
    assert Code2WavScheduler._decompose_batch(3) == [2, 1]
    assert Code2WavScheduler._decompose_batch(5) == [4, 1]
    assert Code2WavScheduler._decompose_batch(6) == [4, 2]
    assert Code2WavScheduler._decompose_batch(7) == [4, 2, 1]
    assert Code2WavScheduler._decompose_batch(8) == [8]


def test_eager_step_plan_is_one_forward() -> None:
    scheduler = _make_batching_scheduler()
    assert scheduler._cuda_graph_runner is None
    participants = [(f"r{i}", Code2WavStreamState()) for i in range(7)]
    assert scheduler.build_step_plan(participants) == [7]


def test_five_streams_take_one_forward_not_two() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    rids = [f"req-{i}" for i in range(5)]
    _feed_batch(scheduler, [(rid, 1) for rid in rids])
    _feed_batch(scheduler, [(rid, 2) for rid in rids])
    assert scheduler._model.calls == [(5, 2, 2)]
    for rid in rids:
        assert scheduler._stream_states[rid].emitted == 2


def test_first_window_fires_immediately() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=1000, batch_floor=4)
    _feed_batch(scheduler, [("req-1", 1), ("req-1", 2)])
    messages = _drain_outbox(scheduler)
    assert [m.type for m in messages] == ["stream"]
    assert scheduler._model.calls == [(1, 2, 2)]


def test_floor_fires_without_deadline() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=1000, batch_floor=2)
    _feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    _feed_batch(scheduler, [("req-b", 3), ("req-b", 4)])
    _drain_outbox(scheduler)
    _feed_batch(scheduler, [("req-a", 5), ("req-a", 6), ("req-b", 7), ("req-b", 8)])
    messages = _drain_outbox(scheduler)
    assert sorted(m.request_id for m in messages) == ["req-a", "req-b"]
    assert scheduler._model.calls == [(1, 2, 2), (1, 2, 2), (2, 2, 3)]


def test_deadline_fires_single() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(scheduler, [("req-1", 1), ("req-1", 2)])
    _drain_outbox(scheduler)
    _feed_batch(scheduler, [("req-1", 3), ("req-1", 4)])
    messages = _drain_outbox(scheduler)
    assert [m.request_id for m in messages] == ["req-1"]
    assert scheduler._model.calls == [(1, 2, 2), (1, 2, 3)]


def test_bucket_isolation() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    _feed_batch(scheduler, [("req-b", 3), ("req-b", 4)])
    _drain_outbox(scheduler)
    _feed_batch(
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
    steady_calls = scheduler._model.calls[2:]
    assert all(call[0] == 1 for call in steady_calls)
    assert sorted(steady_calls) == [(1, 2, 3), (1, 2, 5)]
    assert scheduler._stream_states["req-a"].emitted == 4
    assert scheduler._stream_states["req-b"].emitted == 6


def test_step_cursor_uses_captured_window_end() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(scheduler, [("req-1", 1), ("req-1", 2)])
    _drain_outbox(scheduler)

    state = scheduler._stream_states["req-1"]
    real_forward = scheduler._forward_codes

    def _forward_then_ingest(codes, **kwargs):
        result = real_forward(codes, **kwargs)
        state.chunks.append(torch.tensor([9, 90]))
        return result

    scheduler._forward_codes = _forward_then_ingest
    _feed_batch(scheduler, [("req-1", 3), ("req-1", 4)])

    assert state.emitted == 4
    assert len(state.chunks) == 5
    assert scheduler._ready(state) == 1


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
            control._handle_stream_chunk(rid, _stream_item(code))

    batched = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    for round_start in range(0, 6, 2):
        entries = []
        for rid, codes in schedule.items():
            entries.append((rid, codes[round_start]))
            entries.append((rid, codes[round_start + 1]))
        _feed_batch(batched, entries)

    assert any(call[0] > 1 for call in batched._model.calls)
    for rid in schedule:
        control_state = control._stream_states[rid]
        batched_state = batched._stream_states[rid]
        assert batched_state.emitted == 6
        assert np.array_equal(
            np.concatenate(control_state.audio_parts),
            np.concatenate(batched_state.audio_parts),
        )


def test_mixed_stream_enabled() -> None:
    scheduler = _make_batching_scheduler()
    _feed_batch(
        scheduler,
        [("req-a", 1), ("req-a", 2), ("req-b", 3), ("req-b", 4)],
        stream_flags={"req-a": True, "req-b": False},
    )
    messages = _drain_outbox(scheduler)
    assert [(m.type, m.request_id) for m in messages] == [("stream", "req-a")]
    assert scheduler._model.calls == [(2, 2, 2)]
    for rid in ("req-a", "req-b"):
        state = scheduler._stream_states[rid]
        assert state.emitted == 2
        assert len(state.audio_parts) == 1


def test_step_failure_isolates_participants() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    _feed_batch(scheduler, [("req-b", 3), ("req-b", 4)])
    _drain_outbox(scheduler)

    real_forward = scheduler._forward_codes

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    scheduler._forward_codes = _boom
    _feed_batch(
        scheduler,
        [
            ("req-a", 5),
            ("req-a", 6),
            ("req-b", 7),
            ("req-b", 8),
            ("req-c", 9),
        ],
    )
    assert "req-a" not in scheduler._stream_states
    assert "req-b" not in scheduler._stream_states
    assert scheduler._is_aborted("req-a") and scheduler._is_aborted("req-b")
    assert "req-c" in scheduler._stream_states

    scheduler._forward_codes = real_forward
    _drain_outbox(scheduler)
    _feed_batch(scheduler, [("req-c", 10)])
    messages = _drain_outbox(scheduler)
    assert [(m.type, m.request_id) for m in messages] == [("stream", "req-c")]
    assert scheduler._stream_states["req-c"].emitted == 2


def test_one_participation_per_pump() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(scheduler, [("req-1", 1), ("req-1", 2)])
    _drain_outbox(scheduler)

    selections: list[list[str]] = []
    original_select = scheduler.select_step_participants

    def recording_select():
        participants = original_select()
        if participants:
            selections.append([rid for rid, _ in participants])
        return participants

    scheduler.select_step_participants = recording_select
    _feed_batch(scheduler, [("req-1", 3), ("req-1", 4), ("req-1", 5), ("req-1", 6)])
    assert selections == [["req-1"]]
    state = scheduler._stream_states["req-1"]
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
    assert scheduler._enable_batching is True
    assert scheduler._max_batch_wait_s == 0.25
    assert scheduler._batch_floor == 3
    assert scheduler._batch_ceiling == 4
    assert scheduler._can_batch_stream_chunks is True


def test_forward_codes_eager() -> None:
    scheduler = _make_batching_scheduler()
    codes = torch.zeros(1, 2, 2, dtype=torch.long)
    _, meta = scheduler._forward_codes(codes)
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

    class _ActiveRecorder:
        def is_active(self) -> bool:
            return True

    monkeypatch.setattr(mod, "_get_recorder", lambda: _ActiveRecorder())

    scheduler = _make_batching_scheduler(max_batch_wait_ms=1000, batch_floor=2)
    _feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    _feed_batch(scheduler, [("req-b", 3), ("req-b", 4)])
    _drain_outbox(scheduler)
    events.clear()

    _feed_batch(scheduler, [("req-a", 5), ("req-a", 6), ("req-b", 7), ("req-b", 8)])

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
    assert start_meta["fire_reason"] == "floor"
    assert start_meta["batch_size"] == 2
    assert start_meta["subbatch_decomposition"] == [2]
    assert start_meta["bucket"] == [1, 3]
    assert start_meta["due_bucket_count"] == 1
    assert end_meta["audio_samples"] > 0
    assert end_meta["execution_mode"] == "eager"


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
    scheduler._stream_states["req-1"] = state

    decoded = scheduler.run_step([("req-1", state)], [1])
    assert decoded["req-1"].shape == (4,)
    assert state.emitted == 3
