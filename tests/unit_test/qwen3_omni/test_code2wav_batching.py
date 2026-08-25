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
    _batched_graph_keys,
    _serial_threshold_graph_keys,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.messages import IncomingMessage
from tests.unit_test.fixtures.qwen_fakes import FakeCode2WavModel


class _FakeGraphRunner:
    def __init__(self, model, keys) -> None:
        self._model = model
        self._keys = set(keys)
        self.calls: list[tuple[tuple[int, ...], bool, str]] = []

    def available_batch_sizes(self, frames: int) -> tuple[int, ...]:
        return tuple(
            sorted(
                {key.batch_size for key in self._keys if key.frames == int(frames)},
                reverse=True,
            )
        )

    def run(self, codes: torch.Tensor, *, eligible: bool) -> Code2WavRunResult:
        key = GraphKey(batch_size=int(codes.shape[0]), frames=int(codes.shape[-1]))
        if eligible and key in self._keys:
            mode, reason = "cuda_graph", None
        elif eligible:
            mode, reason = "eager", "key_miss"
        else:
            mode, reason = "eager", "ineligible"
        self.calls.append((tuple(codes.shape), eligible, mode))
        return Code2WavRunResult(self._model(codes), mode, key, reason)

    def stats(self) -> dict:
        return {
            "enabled": True,
            "disable_reason": None,
            "graph_contract": {"keys": len(self._keys)},
        }


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


def _make_chunk_aligned_scheduler(**kwargs) -> Code2WavScheduler:
    model = FakeCode2WavModel(total_upsample=2)
    runner = _FakeGraphRunner(model, _batched_graph_keys(2, 1, 8))
    return Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
        enable_batching=True,
        enable_cuda_graph=True,
        _cuda_graph_runner=runner,
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


def test_decompose_batch_against_published_sizes() -> None:
    decompose = Code2WavScheduler._decompose_batch
    assert decompose(7, (4, 2, 1)) == [4, 2, 1]
    assert decompose(8, (4, 1)) == [4, 4]
    assert decompose(7, (4,)) == [4, 3]
    assert decompose(5, (8,)) == [5]
    assert decompose(3, ()) == [3]


class _AvailabilityRunner:
    def __init__(self, sizes: tuple[int, ...]) -> None:
        self.sizes = sizes
        self.queries: list[int] = []

    def available_batch_sizes(self, frames: int) -> tuple[int, ...]:
        self.queries.append(frames)
        return self.sizes


def _states_with_ready(count: int, ready: int) -> list[tuple[str, Code2WavStreamState]]:
    participants = []
    for i in range(count):
        state = Code2WavStreamState()
        state.chunks = [torch.tensor([0, 0]) for _ in range(ready)]
        participants.append((f"r{i}", state))
    return participants


def test_eager_step_plan_is_one_forward() -> None:
    scheduler = _make_batching_scheduler()
    assert scheduler._cuda_graph_runner is None
    participants = [(f"r{i}", Code2WavStreamState()) for i in range(7)]
    assert scheduler.build_step_plan(participants) == [7]


def test_step_plan_follows_runner_availability_for_the_window() -> None:
    runner = _AvailabilityRunner((4, 2, 1))
    scheduler = _make_batching_scheduler(
        enable_cuda_graph=True,
        _cuda_graph_runner=runner,
    )
    participants = _states_with_ready(7, ready=6)
    assert scheduler.build_step_plan(participants) == [4, 2, 1]
    # Note (ruoyu): the plan must query the chunk-capped window, not the raw
    # backlog depth — an uncapped query would miss the captured key set.
    assert runner.queries[-1] == 2


def test_step_plan_without_published_graphs_stays_one_eager_forward() -> None:
    runner = _AvailabilityRunner(())
    scheduler = _make_batching_scheduler(
        enable_cuda_graph=True,
        _cuda_graph_runner=runner,
    )
    participants = _states_with_ready(7, ready=6)
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


def test_step_failure_after_success_keeps_decoded_sub_batches() -> None:
    scheduler = _make_chunk_aligned_scheduler(max_batch_wait_ms=0, batch_floor=2)
    cleaned: list[str] = []
    scheduler._cleanup_aborted_request = cleaned.append

    real_forward = scheduler._forward_codes
    forwards = 0

    def _fail_on_second_sub_batch(codes, **kwargs):
        nonlocal forwards
        forwards += 1
        if forwards == 2:
            raise RuntimeError("boom")
        return real_forward(codes, **kwargs)

    scheduler._forward_codes = _fail_on_second_sub_batch
    _feed_batch(
        scheduler,
        [(rid, code) for rid in ("req-a", "req-b", "req-c") for code in (1, 2)],
    )

    # Note (ruoyu): plan [2, 1] — the size-2 sub-batch decoded before the
    # size-1 one failed, so its audio must survive the failure.
    messages = _drain_outbox(scheduler)
    assert [m.request_id for m in messages if m.type == "stream"] == [
        "req-a",
        "req-b",
    ]
    assert [m.request_id for m in messages if m.type == "error"] == ["req-c"]
    for rid in ("req-a", "req-b"):
        assert scheduler._stream_states[rid].emitted == 2
        assert not scheduler._is_aborted(rid)
    assert "req-c" not in scheduler._stream_states
    assert scheduler._is_aborted("req-c")
    assert cleaned == ["req-c"]
    assert scheduler._pending_step_failures == []

    scheduler._forward_codes = real_forward
    _feed_batch(scheduler, [("req-a", 3), ("req-a", 4)])
    assert [(m.type, m.request_id) for m in _drain_outbox(scheduler)] == [
        ("stream", "req-a")
    ]
    assert scheduler._stream_states["req-a"].emitted == 4


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
    assert end_meta["sub_batch_execution"] == [
        {
            "batch_size": 2,
            "execution_mode": "eager",
            "graph_key": None,
            "fallback_reason": None,
        }
    ]


def test_batching_and_cuda_graph_coexist() -> None:
    scheduler = _make_chunk_aligned_scheduler()
    assert scheduler._chunk_aligned_dispatch is True
    assert scheduler._cuda_graph_runner is not None
    legacy = _make_batching_scheduler()
    assert legacy._chunk_aligned_dispatch is False


def _ready_participants(n: int) -> list[tuple[str, Code2WavStreamState]]:
    participants = []
    for i in range(n):
        state = Code2WavStreamState()
        state.chunks = [torch.tensor([i, i * 10]), torch.tensor([i, i * 10])]
        participants.append((f"r{i}", state))
    return participants


def test_chunk_aligned_step_plan_decomposes() -> None:
    scheduler = _make_chunk_aligned_scheduler()
    assert scheduler.build_step_plan(_ready_participants(7)) == [4, 2, 1]


def test_chunk_aligned_backlog_drains_in_uniform_graph_windows() -> None:
    scheduler = _make_chunk_aligned_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(scheduler, [("req-1", code) for code in (1, 2, 3, 4, 5, 6)])
    assert scheduler._model.calls == [(1, 2, 2), (1, 2, 3), (1, 2, 3)]
    assert scheduler._stream_states["req-1"].emitted == 6
    runner = scheduler._cuda_graph_runner
    assert [mode for _, _, mode in runner.calls] == ["cuda_graph"] * 3


def test_chunk_aligned_buckets_merge_mixed_backlogs() -> None:
    scheduler = _make_chunk_aligned_scheduler(max_batch_wait_ms=0, batch_floor=2)
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
    # Note (ruoyu): legacy buckets isolate ready=2 from ready=4 (see
    # test_bucket_isolation); chunk-aligned buckets collapse to
    # (context, context+chunk) and merge them.
    assert scheduler._model.calls[2:] == [(2, 2, 3), (1, 2, 3)]
    assert scheduler._stream_states["req-a"].emitted == 4
    assert scheduler._stream_states["req-b"].emitted == 6
    runner = scheduler._cuda_graph_runner
    assert [(call[1], call[2]) for call in runner.calls[-2:]] == [
        (True, "cuda_graph"),
        (True, "cuda_graph"),
    ]


def test_batched_graph_keys_cover_decompose_sizes() -> None:
    keys = _batched_graph_keys(2, 1, 8)
    assert set(keys) == {
        GraphKey(batch_size=size, frames=frames)
        for size in (1, 2, 4, 8)
        for frames in (2, 3)
    }
    capped = _batched_graph_keys(2, 1, 4)
    assert {key.batch_size for key in capped} == {1, 2, 4}


def test_factory_builds_batched_keys_with_batching(monkeypatch) -> None:
    import sglang_omni.models.qwen3_omni.components.code2wav_scheduler as mod

    def _fake_load(path, *, device, dtype):
        model = FakeCode2WavModel(total_upsample=2)
        model.config = SimpleNamespace(num_quantizers=2)
        return model

    monkeypatch.setattr(mod, "load_code2wav_model", _fake_load)
    captured: dict = {}

    class _FakeRunnerCls:
        @classmethod
        def build(cls, model, **kwargs):
            captured.update(kwargs)
            return _FakeGraphRunner(model, kwargs["graph_keys"])

    monkeypatch.setattr(mod, "Code2WavCudaGraphRunner", _FakeRunnerCls)
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
    assert scheduler._chunk_aligned_dispatch is True


def test_serial_only_runner_splits_groups_into_safe_b1_replays() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    runner = _FakeGraphRunner(model, _serial_threshold_graph_keys(2, 1))
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
        enable_batching=True,
        enable_cuda_graph=True,
        _cuda_graph_runner=runner,
    )
    # Note (ruoyu): a serial-only runner may have dropped batched graphs after
    # their eager warmup OOMed, so retrying the group as one eager forward is
    # unsafe even when it benchmarks faster in the non-OOM case.
    assert scheduler._chunk_aligned_dispatch is True
    assert scheduler.build_step_plan(_ready_participants(7)) == [1] * 7


def test_runtime_disable_stops_chunk_aligned_dispatch() -> None:
    scheduler = _make_chunk_aligned_scheduler()
    assert scheduler._chunk_aligned_dispatch is True
    scheduler._cuda_graph_runner._keys = set()
    assert scheduler._chunk_aligned_dispatch is False
    participants = [(f"r{i}", Code2WavStreamState()) for i in range(3)]
    assert scheduler.build_step_plan(participants) == [3]


def test_chunk_aligned_groups_replay_batched_graphs() -> None:
    scheduler = _make_chunk_aligned_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(
        scheduler,
        [(rid, code) for rid in ("req-a", "req-b") for code in (1, 2, 3, 4)],
    )
    runner = scheduler._cuda_graph_runner
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
            control._handle_stream_chunk(rid, _stream_item(code))

    quantized = _make_chunk_aligned_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(
        quantized,
        [(rid, code) for rid, codes in schedule.items() for code in codes],
    )

    assert any(call[0] > 1 for call in quantized._model.calls)
    for rid in schedule:
        assert quantized._stream_states[rid].emitted == 6
        assert np.array_equal(
            np.concatenate(control._stream_states[rid].audio_parts),
            np.concatenate(quantized._stream_states[rid].audio_parts),
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
    scheduler._stream_states["req-1"] = state

    decoded = scheduler.run_step([("req-1", state)], [1])
    assert decoded["req-1"].shape == (4,)
    assert state.emitted == 3


class _StubGraphRunner:
    """Replays through the eager model but reports cuda_graph execution, and
    misses (eager fallback) for batch sizes it does not publish."""

    def __init__(self, model, sizes: tuple[int, ...]) -> None:
        self._model = model
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
            output=self._model(codes),
            execution_mode="cuda_graph" if hit else "eager",
            key=key,
            fallback_reason=None if hit else "key_miss",
        )


def _make_graph_batching_scheduler(
    sizes: tuple[int, ...], **kwargs
) -> tuple[Code2WavScheduler, _StubGraphRunner]:
    model = FakeCode2WavModel(total_upsample=2)
    runner = _StubGraphRunner(model, sizes)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
        enable_batching=True,
        enable_cuda_graph=True,
        _cuda_graph_runner=runner,
        **kwargs,
    )
    return scheduler, runner


def test_batched_step_replays_one_graph_when_size_is_published() -> None:
    scheduler, runner = _make_graph_batching_scheduler((8, 4, 2, 1))
    _feed_batch(
        scheduler,
        [("req-a", 1), ("req-a", 2), ("req-b", 3), ("req-b", 4)],
    )

    assert runner.run_calls == [((2, 2, 2), True)]
    messages = _drain_outbox(scheduler)
    assert sorted(m.request_id for m in messages) == ["req-a", "req-b"]


def test_batched_step_replays_b1_graphs_without_batched_sizes() -> None:
    # Note (ruoyu): a serial-only runner can mean batched eager warmup OOMed,
    # so the plan must stay within its published B1 capacity.
    scheduler, runner = _make_graph_batching_scheduler((1,))
    _feed_batch(
        scheduler,
        [("req-a", 1), ("req-a", 2), ("req-b", 3), ("req-b", 4)],
    )

    assert runner.run_calls == [((1, 2, 2), True), ((1, 2, 2), True)]
    messages = _drain_outbox(scheduler)
    assert sorted(m.request_id for m in messages) == ["req-a", "req-b"]


def test_batch_end_event_reports_mixed_sub_batch_execution(monkeypatch) -> None:
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

    scheduler, runner = _make_graph_batching_scheduler((2,))
    _feed_batch(
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
    scheduler = _make_batching_scheduler(
        max_batch_wait_ms=0, batch_floor=2, initial_codec_chunk_frames=1
    )
    _feed_batch(scheduler, [("req-1", 1)])
    messages = _drain_outbox(scheduler)
    assert [m.type for m in messages] == ["stream"]
    assert scheduler._model.calls == [(1, 2, 1)]
    assert scheduler._stream_states["req-1"].emitted == 1
    _feed_batch(scheduler, [("req-1", 2)])
    assert _drain_outbox(scheduler) == []
    _feed_batch(scheduler, [("req-1", 3)])
    assert [m.type for m in _drain_outbox(scheduler)] == ["stream"]
    assert scheduler._stream_states["req-1"].emitted == 3


def test_initial_codec_chunk_frames_zero_keeps_steady_threshold() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(scheduler, [("req-1", 1)])
    assert _drain_outbox(scheduler) == []
    _feed_batch(scheduler, [("req-1", 2)])
    assert [m.type for m in _drain_outbox(scheduler)] == ["stream"]


def _done(request_id: str) -> IncomingMessage:
    return IncomingMessage(request_id=request_id, type="stream_done", data=None)


def test_next_message_ingests_first_chunks_before_other_messages() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    _drain_outbox(scheduler)
    scheduler.inbox.put(_done("req-a"))
    scheduler.inbox.put(_stream_chunk("req-b", 5))
    scheduler.inbox.put(_stream_chunk("req-b", 6))

    msg = scheduler._next_message()

    assert msg is not None and msg.type == "stream_done"
    assert msg.request_id == "req-a"
    state = scheduler._stream_states["req-b"]
    assert state.emitted == 2
    assert len(state.audio_parts) == 1


def test_next_message_keeps_steady_chunks_in_fifo_order() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    _drain_outbox(scheduler)
    scheduler.inbox.put(_done("req-a"))
    scheduler.inbox.put(_stream_chunk("req-a", 3))
    scheduler.inbox.put(_stream_chunk("req-a", 4))

    msg = scheduler._next_message()

    assert msg is not None and msg.type == "stream_done"
    assert len(scheduler._stream_states["req-a"].chunks) == 2

    batches: list[int] = []
    original = scheduler.on_stream_chunk_batch

    def _recording(items):
        batches.append(len(items))
        return original(items)

    scheduler.on_stream_chunk_batch = _recording
    assert scheduler._next_message() is None
    assert batches == [2]
    assert scheduler._stream_states["req-a"].emitted == 4
