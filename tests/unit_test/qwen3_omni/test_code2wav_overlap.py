# SPDX-License-Identifier: Apache-2.0
"""Output-overlap (roadmap 4.5) coverage for Code2WavScheduler.

The depth-2 pipeline is CUDA-only in production, so CPU tests force it on and
stub the pinned allocation and CUDA event to reach every branch device-free.
Byte-identity tests always compare against a kill-switch control scheduler fed
the identical stream, so a drift in the shared code shows up as a failure in
both arms rather than a false pass.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import torch

from sglang_omni.models.qwen3_omni.components import code2wav_scheduler
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavRunResult,
    GraphKey,
)
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.messages import IncomingMessage
from tests.unit_test.fixtures.qwen_fakes import FakeCode2WavModel, make_qwen_payload


class _FakeEvent:
    """CPU stand-in for torch.cuda.Event with controllable completion."""

    def __init__(self) -> None:
        self.complete = True
        self.sync_error: BaseException | None = None
        self.query_error: BaseException | None = None
        self.record_error: BaseException | None = None
        self.synchronize_calls = 0
        self.query_calls = 0
        self.record_calls = 0

    def record(self) -> None:
        self.record_calls += 1
        if self.record_error is not None:
            raise self.record_error

    def query(self) -> bool:
        self.query_calls += 1
        if self.query_error is not None:
            raise self.query_error
        return self.complete

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        if self.sync_error is not None:
            raise self.sync_error
        self.complete = True


class _DeviceFakeModel(FakeCode2WavModel):
    """FakeCode2WavModel whose output follows the input device (GPU tests)."""

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(codes.shape))
        samples = int(codes.shape[-1]) * self.total_upsample - self.output_deficit
        base = codes.to(dtype=torch.float32).flatten(1).sum(dim=1).view(-1, 1, 1)
        return (
            torch.arange(samples, dtype=torch.float32, device=codes.device).view(
                1, 1, samples
            )
            + base
        )


def _activate_event_capture(monkeypatch) -> list[dict]:
    events: list[dict] = []

    class _ActiveRecorder:
        @staticmethod
        def is_active() -> bool:
            return True

    monkeypatch.setattr(
        code2wav_scheduler, "_get_event_recorder", lambda: _ActiveRecorder()
    )
    monkeypatch.setattr(
        code2wav_scheduler, "_emit_event", lambda **event: events.append(event)
    )
    return events


def _make_scheduler(
    *,
    overlap: bool,
    model: FakeCode2WavModel | None = None,
    stream_chunk_size: int = 10,
    left_context_size: int = 1,
    cuda_graph_runner=None,
) -> Code2WavScheduler:
    return Code2WavScheduler(
        model or FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
        enable_output_overlap=overlap,
        enable_cuda_graph=cuda_graph_runner is not None,
        _cuda_graph_runner=cuda_graph_runner,
    )


def _force_pipeline(scheduler: Code2WavScheduler, monkeypatch) -> None:
    """Enable the CUDA-only pipeline branch on a CPU scheduler."""
    scheduler._pipeline_active = True
    monkeypatch.setattr(
        scheduler,
        "_alloc_pinned",
        lambda samples: torch.empty(samples, dtype=torch.float32),
    )
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)


def _seed(scheduler: Code2WavScheduler, request_id: str = "req-1") -> None:
    scheduler._stream_payloads[request_id] = make_qwen_payload(request_id=request_id)
    scheduler._get_or_create_stream_state(request_id)


def _chunk(index: int) -> torch.Tensor:
    return torch.tensor([index % 7 + 1, 10])


def _feed(
    scheduler: Code2WavScheduler,
    request_id: str,
    indices: range,
    *,
    stream: bool = True,
) -> None:
    for i in indices:
        scheduler._on_chunk(
            request_id,
            StreamItem(i, _chunk(i), "talker", metadata={"stream": stream}),
        )


def _drain_snapshot(scheduler: Code2WavScheduler) -> list[tuple]:
    messages = [scheduler.outbox.get_nowait() for _ in range(scheduler.outbox.qsize())]
    snapshot: list[tuple] = []
    for message in messages:
        if message.type == "stream":
            snapshot.append(
                (
                    message.request_id,
                    message.type,
                    message.data["audio_waveform"],
                    message.data["sample_rate"],
                    message.metadata,
                )
            )
        else:
            snapshot.append((message.request_id, message.type, message.data.data))
    return snapshot


def _run_stream(
    *, overlap: bool, n_chunks: int, stream: bool = True, monkeypatch=None
) -> list[tuple]:
    scheduler = _make_scheduler(overlap=overlap)
    if overlap:
        _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)
    _feed(scheduler, "req-1", range(n_chunks), stream=stream)
    scheduler._on_done("req-1")
    return _drain_snapshot(scheduler)


@pytest.mark.parametrize(
    ("n_chunks", "expected_types"),
    [
        # Note (edwardzh): boundary-exact end is where a naive impl
        # silently drops the pending window.
        (20, ["stream", "stream", "result"]),
        # Note (edwardzh): pending and tail must not merge.
        (21, ["stream", "stream", "stream", "result"]),
    ],
)
def test_overlap_protocol_bitwise_matches_sync(
    monkeypatch, n_chunks: int, expected_types: list[str]
) -> None:
    sync_snapshot = _run_stream(overlap=False, n_chunks=n_chunks)
    overlap_snapshot = _run_stream(
        overlap=True, n_chunks=n_chunks, monkeypatch=monkeypatch
    )

    assert overlap_snapshot == sync_snapshot
    assert [item[1] for item in overlap_snapshot] == expected_types


def test_overlap_protocol_bitwise_matches_sync_with_threshold_eos(monkeypatch) -> None:
    def _run(*, overlap: bool) -> list[tuple]:
        scheduler = _make_scheduler(overlap=overlap)
        if overlap:
            _force_pipeline(scheduler, monkeypatch)
        _seed(scheduler)
        _feed(scheduler, "req-1", range(9))
        scheduler._on_chunk(
            "req-1",
            StreamItem(
                9,
                torch.tensor([2150, 0]),
                "talker",
                metadata={"stream": True},
            ),
        )
        _feed(scheduler, "req-1", range(9, 20))
        scheduler._on_done("req-1")
        return _drain_snapshot(scheduler)

    assert _run(overlap=True) == _run(overlap=False)


def test_overlap_first_window_sync_second_deferred(monkeypatch) -> None:
    control = _run_stream(overlap=False, n_chunks=30)

    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)

    _feed(scheduler, "req-1", range(10))
    assert scheduler.outbox.qsize() == 1  # first window emits synchronously

    _feed(scheduler, "req-1", range(10, 20))
    assert scheduler.outbox.qsize() == 1  # second window launched, deferred

    _feed(scheduler, "req-1", range(20, 30))
    assert scheduler.outbox.qsize() == 2  # third launch flushed window 2

    scheduler._on_done("req-1")
    snapshot = _drain_snapshot(scheduler)
    assert snapshot == control


def test_overlap_nonstreaming_pending_appends_parts_result_only(monkeypatch) -> None:
    control = _run_stream(overlap=False, n_chunks=21, stream=False)
    overlap = _run_stream(
        overlap=True, n_chunks=21, stream=False, monkeypatch=monkeypatch
    )

    assert overlap == control
    assert [item[1] for item in overlap] == ["result"]
    audio = np.frombuffer(overlap[0][2]["audio_waveform"], dtype=np.float32)
    assert audio.shape == (42,)


def test_overlap_flush_failure_keeps_pending_owned_until_abort(monkeypatch) -> None:
    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)
    _feed(scheduler, "req-1", range(20))

    state = scheduler._stream_states["req-1"]
    pending = state.pending
    assert pending is not None
    event = pending.slot.event
    event.complete = False
    event.sync_error = RuntimeError("D2H synchronization failed")

    with pytest.raises(RuntimeError, match="D2H synchronization failed"):
        scheduler._flush_pending("req-1", state)

    assert state.pending is pending
    assert pending.slot not in scheduler._pinned_free

    scheduler.abort("req-1")

    assert state.pending is None
    assert pending.slot in scheduler._pinned_retired
    assert pending.slot not in scheduler._pinned_free


def test_overlap_abort_retires_inflight_slot_without_synchronizing(monkeypatch) -> None:
    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)
    _feed(scheduler, "req-1", range(20))

    state = scheduler._stream_states["req-1"]
    pending = state.pending
    assert pending is not None
    event = pending.slot.event
    event.complete = False
    event.sync_error = AssertionError("abort must not synchronize")

    scheduler.abort("req-1")

    assert event.synchronize_calls == 0
    assert pending.slot in scheduler._pinned_retired
    assert pending.slot not in scheduler._pinned_free


def test_overlap_acquire_reaps_completed_retired_slot(monkeypatch) -> None:
    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)
    _feed(scheduler, "req-1", range(20))

    pending = scheduler._stream_states["req-1"].pending
    assert pending is not None
    slot = pending.slot
    slot.event.complete = False
    scheduler.abort("req-1")

    slot.event.complete = True
    acquired = scheduler._acquire_slot(slot.buffer.numel())

    assert acquired is slot
    assert scheduler._pinned_retired == []
    assert scheduler._pinned_created == 1


def test_overlap_query_failure_quarantines_slot(monkeypatch, caplog) -> None:
    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)
    _feed(scheduler, "req-1", range(20))

    pending = scheduler._stream_states["req-1"].pending
    assert pending is not None
    slot = pending.slot
    slot.event.query_error = RuntimeError("event query failed")
    scheduler.abort("req-1")

    scheduler._reap_retired_slots()

    assert scheduler._pinned_retired == []
    assert scheduler._pinned_quarantined == [slot]
    assert slot not in scheduler._pinned_free
    assert scheduler._pipeline_active is False
    assert "failed to query a retired D2H copy" in caplog.text


def test_overlap_previous_flush_failure_keeps_both_slots_owned(monkeypatch) -> None:
    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)
    _feed(scheduler, "req-1", range(20))

    state = scheduler._stream_states["req-1"]
    previous = state.pending
    assert previous is not None
    previous.slot.event.complete = False
    previous.slot.event.sync_error = RuntimeError("previous flush failed")

    with pytest.raises(RuntimeError, match="previous flush failed"):
        _feed(scheduler, "req-1", range(20, 30))

    assert state.pending is previous
    assert len(scheduler._pinned_retired) == 1
    current_slot = scheduler._pinned_retired[0]
    assert current_slot is not previous.slot
    assert current_slot.event.record_calls == 1

    scheduler.abort("req-1")
    assert previous.slot in scheduler._pinned_retired
    assert current_slot in scheduler._pinned_retired


def test_overlap_record_failure_quarantines_current_slot(monkeypatch) -> None:
    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)
    _feed(scheduler, "req-1", range(10))

    event = _FakeEvent()
    event.record_error = RuntimeError("event record failed")
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)

    with pytest.raises(RuntimeError, match="event record failed"):
        _feed(scheduler, "req-1", range(10, 20))

    assert len(scheduler._pinned_quarantined) == 1
    assert scheduler._pinned_quarantined[0].event is event
    assert scheduler._pinned_free == []
    assert scheduler._pipeline_active is False


def test_overlap_slot_growth_failure_returns_original_free_slot(monkeypatch) -> None:
    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    slot = scheduler._acquire_slot(2)
    assert slot is not None
    scheduler._release_slot(slot)

    def _fail_alloc(samples: int) -> torch.Tensor:
        raise RuntimeError(f"cannot grow to {samples}")

    monkeypatch.setattr(scheduler, "_alloc_pinned", _fail_alloc)

    with pytest.raises(RuntimeError, match="cannot grow"):
        scheduler._acquire_slot(slot.buffer.numel() + 1)

    assert scheduler._pinned_free == [slot]
    assert scheduler._pinned_created == 1


def test_overlap_flush_synchronizes_before_releasing_slot(monkeypatch) -> None:
    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)
    _feed(scheduler, "req-1", range(20))

    pending = scheduler._stream_states["req-1"].pending
    assert pending is not None
    event = pending.slot.event

    release_slot = scheduler._release_slot

    def _release_after_synchronize(slot) -> None:
        assert slot.event.synchronize_calls == 1
        release_slot(slot)

    monkeypatch.setattr(scheduler, "_release_slot", _release_after_synchronize)

    scheduler._on_done("req-1")

    assert event.synchronize_calls == 1
    assert pending.slot in scheduler._pinned_free


def test_overlap_replay_failure_with_pending_aborts_and_releases(monkeypatch) -> None:
    class _FailOnThirdRunner:
        def __init__(self, model, error: Exception) -> None:
            self.model = model
            self.error = error
            self.runs = 0

        def run(self, codes: torch.Tensor, *, eligible: bool) -> Code2WavRunResult:
            self.runs += 1
            if self.runs == 3:
                raise self.error
            return Code2WavRunResult(
                self.model(codes),
                "cuda_graph",
                GraphKey(1, int(codes.shape[-1])),
                None,
            )

    model = FakeCode2WavModel(total_upsample=2)
    replay_error = RuntimeError("replay exploded")
    runner = _FailOnThirdRunner(model, replay_error)
    scheduler = _make_scheduler(overlap=True, model=model, cuda_graph_runner=runner)
    _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)

    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        for i in range(30):
            scheduler.inbox.put(
                IncomingMessage(
                    request_id="req-1",
                    type="stream_chunk",
                    data=StreamItem(i, _chunk(i), "talker", metadata={"stream": True}),
                )
            )
        messages = []
        while True:
            message = scheduler.outbox.get(timeout=2.0)
            messages.append(message)
            if message.type == "error":
                break
        # Note (wenyao): the reclaim must be observed while the scheduler is
        # still running — stopping first lets the shutdown drain synchronize
        # instead, which would hide a missing reap.
        deadline = time.monotonic() + 2.0
        while not scheduler._pinned_free and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)
    assert not thread.is_alive()

    assert messages[-1].data is replay_error
    assert scheduler._is_aborted("req-1")
    assert "req-1" not in scheduler._stream_states
    # Note (edwardzh): reclaimed via release_stream_resources, which is
    # the only path an aborted request takes.
    # Note (wenyao): two — the second window's copy had drained before the third
    # replay failed, so its audio reaches the client instead of dying with the
    # aborted request.
    assert [message.type for message in messages] == ["stream", "stream", "error"]
    assert scheduler._pinned_retired == []
    assert len(scheduler._pinned_free) == 1
    assert scheduler._pinned_free[0].event.query_calls >= 1


def test_overlap_pool_exhaustion_falls_back_sync_per_window(monkeypatch) -> None:
    control = _make_scheduler(overlap=False)
    _seed(control, "req-a")
    _seed(control, "req-b")

    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    scheduler._MAX_PINNED_SLOTS = 1
    _seed(scheduler, "req-a")
    _seed(scheduler, "req-b")

    for target in (control, scheduler):
        _feed(target, "req-a", range(20))  # window 2 pipelined, holds the slot
        _feed(target, "req-b", range(20))  # window 2 finds no slot: sync path
        _feed(target, "req-a", range(20, 30))  # flush-own-pending reuses slot
        target._on_done("req-a")
        target._on_done("req-b")

    def _by_request(snapshot: list[tuple]) -> dict[str, list[tuple]]:
        grouped: dict[str, list[tuple]] = {}
        for item in snapshot:
            grouped.setdefault(item[0], []).append(item)
        return grouped

    # Note (edwardzh): the pipeline defers req-a past req-b, so only
    # per-request order is contractual, not global order.
    assert _by_request(_drain_snapshot(scheduler)) == _by_request(
        _drain_snapshot(control)
    )
    assert scheduler._pinned_created == 1


def test_eos_lazy_scan_one_scan_per_window_and_tail_stays_stream_done(
    monkeypatch,
) -> None:
    events = _activate_event_capture(monkeypatch)
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = _make_scheduler(overlap=True, model=model)
    _seed(scheduler)
    scans: list[int] = []
    original_scan = scheduler._scan_unchecked

    def _counted_scan(state):
        scans.append(len(state.chunks) - state.checked)
        return original_scan(state)

    monkeypatch.setattr(scheduler, "_scan_unchecked", _counted_scan)

    # Note (edwardzh): raw ready hits the threshold here, so this fails
    # if the scan runs after the gate instead of before it.
    _feed(scheduler, "req-1", range(9))
    scheduler._on_chunk(
        "req-1",
        StreamItem(9, torch.tensor([2150, 0]), "talker", metadata={"stream": True}),
    )
    assert model.calls == []
    assert scans == [10]

    scheduler._on_done("req-1")
    assert model.calls == [(1, 2, 9)]
    decode_start = next(
        event for event in events if event["event_name"] == "code2wav_decode_start"
    )
    assert decode_start["metadata"]["trigger"] == "stream_done"
    assert decode_start["metadata"]["new_frames"] == 9


def test_eos_lazy_scan_batches_one_scan_per_threshold_window(monkeypatch) -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = _make_scheduler(overlap=True, model=model)
    _seed(scheduler)
    scans: list[int] = []
    original_scan = scheduler._scan_unchecked

    def _counted_scan(state):
        scans.append(len(state.chunks) - state.checked)
        return original_scan(state)

    monkeypatch.setattr(scheduler, "_scan_unchecked", _counted_scan)

    _feed(scheduler, "req-1", range(30))
    assert model.calls == [(1, 2, 10), (1, 2, 11), (1, 2, 11)]
    assert scans == [10, 10, 10]

    scheduler._on_done("req-1")
    # Note (edwardzh): stream-done rescans unconditionally.
    assert scans == [10, 10, 10, 0]


def test_overlap_events_order_and_metadata(monkeypatch) -> None:
    events = _activate_event_capture(monkeypatch)
    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)
    _feed(scheduler, "req-1", range(20))
    scheduler._on_done("req-1")

    decode_events = [
        event["event_name"]
        for event in events
        if event["event_name"].startswith("code2wav_decode_")
    ]
    assert decode_events == [
        "code2wav_decode_start",
        "code2wav_decode_end",
        "code2wav_decode_start",
        "code2wav_decode_launched",
        "code2wav_decode_end",
    ]

    first_end, second_end = (
        event for event in events if event["event_name"] == "code2wav_decode_end"
    )
    assert first_end["metadata"]["pipelined"] is False
    assert first_end["metadata"]["d2h_wait_ns"] == 0
    assert first_end["metadata"]["audio_samples"] == 20
    assert second_end["metadata"]["pipelined"] is True
    assert second_end["metadata"]["d2h_wait_ns"] >= 0
    assert second_end["metadata"]["audio_samples"] == 20

    launched = next(
        event for event in events if event["event_name"] == "code2wav_decode_launched"
    )
    assert launched["metadata"] == {
        "execution_mode": "eager",
        "graph_key": None,
        "fallback_reason": None,
        "window_frames": 11,
        "new_frames": 10,
    }

    first_audio_index = next(
        i
        for i, event in enumerate(events)
        if event["event_name"] == "code2wav_first_audio"
    )
    second_start_index = [
        i
        for i, event in enumerate(events)
        if event["event_name"] == "code2wav_decode_start"
    ][1]
    assert first_audio_index < second_start_index  # TTFA event timing unchanged


def test_overlap_drained_window_is_emitted_without_a_further_dispatch(
    monkeypatch,
) -> None:
    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)

    _feed(scheduler, "req-1", range(20))
    assert scheduler._stream_states["req-1"].pending is not None
    assert [item[1] for item in _drain_snapshot(scheduler)] == ["stream"]

    _feed(scheduler, "req-1", range(20, 21))
    assert scheduler._stream_states["req-1"].pending is None
    assert [item[1] for item in _drain_snapshot(scheduler)] == ["stream"]


def test_overlap_sync_scheduler_emits_no_new_event_keys(monkeypatch) -> None:
    events = _activate_event_capture(monkeypatch)
    scheduler = _make_scheduler(overlap=False)
    _seed(scheduler)
    _feed(scheduler, "req-1", range(10))

    decode_end = next(
        event for event in events if event["event_name"] == "code2wav_decode_end"
    )
    assert "pipelined" not in decode_end["metadata"]
    assert "d2h_wait_ns" not in decode_end["metadata"]
    assert not any(
        event["event_name"] == "code2wav_decode_launched" for event in events
    )


def test_overlap_borrowed_output_copied_before_next_replay(monkeypatch) -> None:
    class _BorrowedOutputRunner:
        def __init__(self) -> None:
            self.static_output = torch.zeros((1, 1, 2), dtype=torch.float32)
            self.replays = 0

        def run(self, codes: torch.Tensor, *, eligible: bool) -> Code2WavRunResult:
            assert eligible
            self.replays += 1
            self.static_output.fill_(float(self.replays))
            return Code2WavRunResult(
                self.static_output,
                "cuda_graph",
                GraphKey(1, int(codes.shape[-1])),
                None,
            )

    runner = _BorrowedOutputRunner()
    scheduler = _make_scheduler(
        overlap=True,
        stream_chunk_size=1,
        left_context_size=0,
        cuda_graph_runner=runner,
    )
    _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)
    _feed(scheduler, "req-1", range(3))
    state = scheduler._stream_states["req-1"]
    scheduler._on_done("req-1")

    # Note (edwardzh): replay N+1 overwrites the static buffer before
    # window N flushes, so this fails if the copy is not launch-ordered.
    assert [chunk.tolist() for chunk in state.audio_parts] == [
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_overlap_gpu_real_pinned_event_bitwise() -> None:
    def _run(*, overlap: bool) -> list[tuple]:
        # Note (edwardzh): bare cuda has no index; only the factory
        # normalizes it.
        scheduler = Code2WavScheduler(
            _DeviceFakeModel(total_upsample=2),
            device=f"cuda:{torch.cuda.current_device()}",
            stream_chunk_size=10,
            left_context_size=1,
            enable_output_overlap=overlap,
        )
        assert scheduler._pipeline_active is overlap
        _seed(scheduler)
        _feed(scheduler, "req-1", range(21))
        scheduler._on_done("req-1")
        return _drain_snapshot(scheduler)

    overlap_snapshot = _run(overlap=True)
    sync_snapshot = _run(overlap=False)
    assert overlap_snapshot == sync_snapshot
    assert [item[1] for item in overlap_snapshot] == [
        "stream",
        "stream",
        "stream",
        "result",
    ]
