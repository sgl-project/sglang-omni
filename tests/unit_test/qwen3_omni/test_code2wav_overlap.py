# SPDX-License-Identifier: Apache-2.0
"""Output-overlap (roadmap 4.5) coverage for Code2WavScheduler.

The depth-2 pipeline is CUDA-only in production, so CPU tests force it on and
stub the pinned allocation and CUDA event to reach every branch device-free.
Byte-identity tests always compare against a kill-switch control scheduler fed
the identical stream, so a drift in the shared code shows up as a failure in
both arms rather than a false pass. The ``accelerator`` cases run real pinned
buffers and events: eager/graph parity, in-flight completion queries, abort
recovery, and cross-device use.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import torch

from sglang_omni.models.qwen3_omni.components import code2wav_scheduler
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavCudaGraphRunner,
    Code2WavRunResult,
    GraphKey,
)
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.utils import cuda_staging
from sglang_omni.utils.cuda_staging import PinnedTransferSlot
from tests.unit_test.fixtures.accelerator import require_cuda
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

    def record(self, stream=None) -> None:
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


def _slot_event(slot: PinnedTransferSlot) -> _FakeEvent:
    """Return the slot's lazily created completion event (a _FakeEvent after
    _force_pipeline); tests drive completion and failures through it."""
    return slot._event


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


class _SlowDeviceFakeModel(_DeviceFakeModel):
    """_DeviceFakeModel that queues ~0.5 s of device work first, so the fenced
    D2H copy is observably in flight (GPU tests)."""

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        torch.cuda._sleep(1_000_000_000)
        return super().__call__(codes)


class _DeviceFakeModule(torch.nn.Module):
    """CUDA-graph-capturable twin of _DeviceFakeModel (GPU tests)."""

    total_upsample = 2

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        samples = int(codes.shape[-1]) * self.total_upsample
        base = codes.to(dtype=torch.float32).flatten(1).sum(dim=1).view(-1, 1, 1)
        return (
            torch.arange(samples, dtype=torch.float32, device=codes.device).view(
                1, 1, samples
            )
            + base
        )


class _SlowDeviceFakeModule(_DeviceFakeModule):
    """_DeviceFakeModule whose replay queues ~0.5 s of device work first; the
    spin kernel is captured with the graph."""

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        torch.cuda._sleep(1_000_000_000)
        return super().forward(codes)


def _make_gpu_scheduler(
    *,
    overlap: bool,
    device: torch.device,
    model: FakeCode2WavModel | torch.nn.Module | None = None,
    cuda_graph: bool = False,
    slow: bool = False,
) -> Code2WavScheduler:
    """Real-CUDA scheduler: real pinned buffers, real events, optionally a
    real graph runner over the serving-reachable serial keys."""
    if model is None:
        if cuda_graph:
            module = _SlowDeviceFakeModule() if slow else _DeviceFakeModule()
            model = module.to(device).eval()
        else:
            model = (
                _SlowDeviceFakeModel(total_upsample=2)
                if slow
                else _DeviceFakeModel(total_upsample=2)
            )
    runner = None
    if cuda_graph:
        runner = Code2WavCudaGraphRunner.build(
            model,
            device=device,
            num_quantizers=2,
            total_gpu_memory_fraction=1.0,
            graph_keys=code2wav_scheduler._serial_threshold_graph_keys(10, 1),
        )
        assert runner.stats()["enabled"] is True
    scheduler = Code2WavScheduler(
        model,
        device=str(device),
        stream_chunk_size=10,
        left_context_size=1,
        enable_output_overlap=overlap,
        enable_cuda_graph=cuda_graph,
        _cuda_graph_runner=runner,
    )
    assert scheduler._pipeline_active is overlap
    if overlap:
        # Note (jiannan-17): cudaHostAlloc may synchronize the device, so no
        # pinned allocation may sit between queued device work and the fenced
        # copy the probe tests observe.
        scheduler._release_slot(
            scheduler._acquire_slot(scheduler._default_slot_samples)
        )
    return scheduler


def _stage_chunks(device: torch.device, n_chunks: int) -> list[torch.Tensor]:
    """Frames already on ``device``: ``validate_chunk``'s blocking ``.to()``
    from pageable memory synchronizes the decode stream, which would drain an
    in-flight copy before the probe."""
    chunks = [_chunk(i).to(device) for i in range(n_chunks)]
    torch.cuda.synchronize(device)
    return chunks


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


def _force_pipeline(scheduler: Code2WavScheduler, monkeypatch) -> list:
    """Enable the CUDA-only pipeline branch on a CPU scheduler.

    Returns the list of devices the launch asked ``torch.cuda.current_stream``
    for, one entry per pipelined window.
    """
    scheduler._pipeline_active = True
    monkeypatch.setattr(
        cuda_staging,
        "_allocate_pinned",
        lambda numel, dtype: torch.empty(numel, dtype=dtype),
    )
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    stream_devices: list = []

    def current_stream(device=None):
        stream_devices.append(device)
        return None

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    return stream_devices


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
    chunks: list[torch.Tensor] | None = None,
) -> None:
    for i in indices:
        codes = _chunk(i) if chunks is None else chunks[i]
        scheduler._on_chunk(
            request_id,
            StreamItem(i, codes, "talker", metadata={"stream": stream}),
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
    stream_devices = _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)

    _feed(scheduler, "req-1", range(10))
    assert scheduler.outbox.qsize() == 1  # first window emits synchronously
    assert stream_devices == []

    _feed(scheduler, "req-1", range(10, 20))
    assert scheduler.outbox.qsize() == 1  # second window launched, deferred
    # Note (jiannan-17): the fence is recorded on the scheduler device's
    # stream, not the thread-current device's.
    assert stream_devices == [scheduler._device]

    _feed(scheduler, "req-1", range(20, 30))
    assert scheduler.outbox.qsize() == 2  # third launch flushed window 2
    assert stream_devices == [scheduler._device] * 2

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
    event = _slot_event(pending.slot)
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
    event = _slot_event(pending.slot)
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
    _slot_event(slot).complete = False
    scheduler.abort("req-1")

    _slot_event(slot).complete = True
    acquired = scheduler._acquire_slot(slot.capacity)

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
    _slot_event(slot).query_error = RuntimeError("event query failed")
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
    _slot_event(previous.slot).complete = False
    _slot_event(previous.slot).sync_error = RuntimeError("previous flush failed")

    with pytest.raises(RuntimeError, match="previous flush failed"):
        _feed(scheduler, "req-1", range(20, 30))

    assert state.pending is previous
    assert len(scheduler._pinned_retired) == 1
    current_slot = scheduler._pinned_retired[0]
    assert current_slot is not previous.slot
    assert _slot_event(current_slot).record_calls == 1

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

    assert event.record_calls == 1
    assert scheduler._pinned_created == 1
    assert len(scheduler._pinned_quarantined) == 1
    assert _slot_event(scheduler._pinned_quarantined[0]) is event
    assert scheduler._pinned_retired == []
    assert scheduler._pinned_free == []
    assert scheduler._pipeline_active is False
    assert scheduler._stream_states["req-1"].pending is None
    # The slot must not treat the failed transfer as complete.
    with pytest.raises(RuntimeError, match="not recorded"):
        scheduler._pinned_quarantined[0].query()


def test_overlap_rerecord_failure_on_reused_slot_quarantines_it(monkeypatch) -> None:
    """A free-pool slot whose second record() raises is quarantined and refuses
    completion reads; the other pending window still flushes."""
    control = _run_stream(overlap=False, n_chunks=40)

    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)
    state_lookup = scheduler._stream_states

    _feed(scheduler, "req-1", range(20))  # window 2 pipelined on slot A
    first = state_lookup["req-1"].pending
    assert first is not None
    slot_a = first.slot
    # Keep each copy "in flight" until flushed; the fake completes on synchronize().
    _slot_event(slot_a).complete = False
    _feed(scheduler, "req-1", range(20, 30))  # window 3 on slot B, A flushed
    second = state_lookup["req-1"].pending
    assert second is not None and second.slot is not slot_a
    slot_b = second.slot
    _slot_event(slot_b).complete = False
    assert scheduler._pinned_free == [slot_a]
    assert scheduler._pinned_created == 2
    assert _slot_event(slot_a).record_calls == 1
    assert _slot_event(slot_a).synchronize_calls == 1

    _slot_event(slot_a).record_error = RuntimeError("event record failed")
    with pytest.raises(RuntimeError, match="event record failed"):
        _feed(scheduler, "req-1", range(30, 40))  # window 4 pops A again

    assert _slot_event(slot_a).record_calls == 2
    assert scheduler._pinned_quarantined == [slot_a]
    assert scheduler._pinned_free == []
    assert scheduler._pinned_retired == []
    assert scheduler._pinned_created == 2
    assert scheduler._pipeline_active is False
    assert state_lookup["req-1"].pending is second, "window 3 is still owned"
    # The slot must not treat the failed transfer as complete.
    assert _slot_event(slot_a).complete is True
    with pytest.raises(RuntimeError, match="not recorded"):
        slot_a.query()
    with pytest.raises(RuntimeError, match="not recorded"):
        slot_a.synchronize()

    scheduler._on_done("req-1")
    assert scheduler._pinned_free == [slot_b]
    snapshot = _drain_snapshot(scheduler)
    assert [item[1] for item in snapshot] == ["stream"] * 4 + ["result"]
    assert snapshot == control


def test_overlap_slot_growth_failure_returns_original_free_slot(monkeypatch) -> None:
    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    slot = scheduler._acquire_slot(2)
    assert slot is not None
    scheduler._release_slot(slot)

    def _fail_alloc(numel: int, dtype: torch.dtype) -> torch.Tensor:
        raise RuntimeError(f"cannot grow to {numel}")

    monkeypatch.setattr(cuda_staging, "_allocate_pinned", _fail_alloc)

    with pytest.raises(RuntimeError, match="cannot grow"):
        scheduler._acquire_slot(slot.capacity + 1)

    assert scheduler._pinned_free == [slot]
    assert scheduler._pinned_created == 1


def test_overlap_flush_synchronizes_before_releasing_slot(monkeypatch) -> None:
    scheduler = _make_scheduler(overlap=True)
    _force_pipeline(scheduler, monkeypatch)
    _seed(scheduler)
    _feed(scheduler, "req-1", range(20))

    pending = scheduler._stream_states["req-1"].pending
    assert pending is not None
    event = _slot_event(pending.slot)

    release_slot = scheduler._release_slot

    def _release_after_synchronize(slot) -> None:
        assert _slot_event(slot).synchronize_calls == 1
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
    assert _slot_event(scheduler._pinned_free[0]).query_calls >= 1


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


@pytest.mark.accelerator
@pytest.mark.parametrize("cuda_graph", [False, True], ids=["eager", "cuda_graph"])
@pytest.mark.parametrize(
    ("n_chunks", "expected_types"),
    [
        # Note (edwardzh): boundary-exact end is where a naive impl
        # silently drops the pending window.
        (20, ["stream", "stream", "result"]),
        # Note (edwardzh): pending and tail must not merge.
        (21, ["stream", "stream", "stream", "result"]),
        # Note (jiannan-17): a replay over a pending copy's source; same-stream
        # ordering keeps the bytes identical.
        (30, ["stream", "stream", "stream", "result"]),
        (31, ["stream", "stream", "stream", "stream", "result"]),
    ],
)
def test_overlap_gpu_real_pinned_event_bitwise(
    cuda_graph: bool, n_chunks: int, expected_types: list[str]
) -> None:
    require_cuda()
    # Note (edwardzh): bare cuda has no index; only the factory normalizes it.
    device = torch.device("cuda", torch.cuda.current_device())

    def _run(*, overlap: bool) -> list[tuple]:
        scheduler = _make_gpu_scheduler(
            overlap=overlap, device=device, cuda_graph=cuda_graph
        )
        _seed(scheduler)
        _feed(scheduler, "req-1", range(n_chunks))
        scheduler._on_done("req-1")
        if cuda_graph:
            runtime = scheduler._cuda_graph_runner.stats()["runtime"]
            assert runtime["graph_replays"] == n_chunks // 10
            assert runtime["replay_failures"] == 0
            assert runtime["fallback_counts"] == (
                {"ineligible": 1} if n_chunks % 10 else {}
            ), "only the stream-done tail may run eagerly"
        return _drain_snapshot(scheduler)

    overlap_snapshot = _run(overlap=True)
    sync_snapshot = _run(overlap=False)
    assert overlap_snapshot == sync_snapshot
    assert [item[1] for item in overlap_snapshot] == expected_types


@pytest.mark.accelerator
def test_overlap_gpu_query_is_false_until_inflight_copy_drains() -> None:
    """The per-frame probe is False while the copy is in flight, True once it
    drained, and the next chunk then flushes without a dispatch."""
    require_cuda()
    device = torch.device("cuda", torch.cuda.current_device())
    chunks = _stage_chunks(device, 22)
    control = _make_gpu_scheduler(
        overlap=False, device=device, model=_SlowDeviceFakeModel(total_upsample=2)
    )
    _seed(control)
    _feed(control, "req-1", range(22), chunks=chunks)
    control._on_done("req-1")
    control_snapshot = _drain_snapshot(control)

    scheduler = _make_gpu_scheduler(
        overlap=True, device=device, model=_SlowDeviceFakeModel(total_upsample=2)
    )
    _seed(scheduler)
    _feed(scheduler, "req-1", range(20), chunks=chunks)
    state = scheduler._stream_states["req-1"]
    pending = state.pending
    assert pending is not None
    assert pending.slot.device == device
    assert pending.slot.view(pending.samples).is_pinned()
    assert pending.slot.query() is False
    first_window = _drain_snapshot(scheduler)
    assert [item[1] for item in first_window] == ["stream"]

    _feed(scheduler, "req-1", range(20, 21), chunks=chunks)  # probe: in flight
    assert state.pending is pending
    assert scheduler.outbox.qsize() == 0

    pending.slot.synchronize()
    assert pending.slot.query() is True
    assert state.pending is pending, "observing completion is not a flush"

    _feed(scheduler, "req-1", range(21, 22), chunks=chunks)  # True: flush
    assert state.pending is None
    second_window = _drain_snapshot(scheduler)
    assert [item[1] for item in second_window] == ["stream"]
    assert scheduler._pinned_retired == []
    assert scheduler._pinned_free == [pending.slot]

    scheduler._on_done("req-1")
    tail = _drain_snapshot(scheduler)
    assert [item[1] for item in tail] == ["stream", "result"]
    assert [*first_window, *second_window, *tail] == control_snapshot


@pytest.mark.accelerator
@pytest.mark.parametrize("cuda_graph", [False, True], ids=["eager", "cuda_graph"])
def test_overlap_gpu_abort_midstream_neither_blocks_nor_reuses_inflight_slot(
    cuda_graph: bool,
) -> None:
    """Abort with a copy in flight neither waits nor hands the slot out, and
    the bytes land intact even after a later replay rewrote their source."""
    require_cuda()
    device = torch.device("cuda", torch.cuda.current_device())
    chunks = _stage_chunks(device, 31)
    control = _make_gpu_scheduler(overlap=False, device=device, cuda_graph=cuda_graph)
    _seed(control)
    _feed(control, "req-1", range(20), chunks=chunks)
    control._on_done("req-1")
    expected_window_2 = np.frombuffer(
        _drain_snapshot(control)[1][2], dtype=np.float32
    ).copy()

    scheduler = _make_gpu_scheduler(
        overlap=True, device=device, cuda_graph=cuda_graph, slow=True
    )
    _seed(scheduler)
    _feed(scheduler, "req-1", range(20), chunks=chunks)
    state = scheduler._stream_states["req-1"]
    pending = state.pending
    assert pending is not None
    slot = pending.slot
    assert slot.query() is False

    scheduler.abort("req-1")

    # A synchronizing abort would have drained the copy.
    assert slot.query() is False
    assert state.pending is None
    assert "req-1" not in scheduler._stream_states
    assert scheduler._pinned_retired == [slot]
    assert scheduler._pinned_free == []

    # Note (jiannan-17): a same-key replay rewrites the retired copy's source;
    # same-stream ordering means the copy still lands with the old bytes. This
    # runs before any pinned allocation (cudaHostAlloc may synchronize).
    window = torch.stack(chunks[19:30], dim=0).transpose(0, 1).unsqueeze(0)
    _, execution = scheduler._forward_codes(window, graph_eligible=True)
    assert execution["execution_mode"] == ("cuda_graph" if cuda_graph else "eager")
    assert slot.query() is False, "the replay queues behind the copy, not before"

    # The reap precedes the allocation, so this holds even if cudaHostAlloc
    # synchronizes.
    other = scheduler._acquire_slot(pending.samples)
    assert other is not None and other is not slot
    assert scheduler._pinned_created == 2
    assert scheduler._pinned_retired == [slot]
    scheduler._release_slot(other)

    slot.synchronize()
    assert slot.query() is True
    assert np.array_equal(
        slot.view(pending.samples).numpy(), expected_window_2
    ), "the retired copy landed intact; nothing overwrote the buffer early"

    reaped = scheduler._acquire_slot(pending.samples)
    assert reaped is slot
    assert scheduler._pinned_retired == []
    assert scheduler._pinned_quarantined == []
    assert scheduler._pinned_created == 2
    scheduler._release_slot(reaped)
    assert sorted(map(id, scheduler._pinned_free)) == sorted(map(id, [other, slot]))


@pytest.mark.accelerator
def test_overlap_gpu_slot_on_other_device_than_process_current() -> None:
    """Slots live on ``cuda:1``; warm-up, probes, waits, flushes, reap and
    shutdown drain run from ``cuda:0`` and leave it current. (``decode_delta``
    pins ``cuda:1`` by design.)"""
    require_cuda(min_devices=2)
    previous_device = torch.cuda.current_device()
    try:
        device = torch.device("cuda", 1)
        torch.cuda.set_device(0)
        chunks = _stage_chunks(device, 22)
        assert torch.cuda.current_device() == 0
        control = _make_gpu_scheduler(
            overlap=False, device=device, model=_SlowDeviceFakeModel(total_upsample=2)
        )
        _seed(control)
        _feed(control, "req-1", range(22), chunks=chunks)
        control._on_done("req-1")
        control_snapshot = _drain_snapshot(control)

        # The control's forwards pinned cuda:1.
        torch.cuda.set_device(0)
        scheduler = _make_gpu_scheduler(
            overlap=True, device=device, model=_SlowDeviceFakeModel(total_upsample=2)
        )
        assert torch.cuda.current_device() == 0
        _seed(scheduler)
        _feed(scheduler, "req-1", range(20), chunks=chunks)
        state = scheduler._stream_states["req-1"]
        pending = state.pending
        assert pending is not None
        assert pending.slot.device == device
        first_window = _drain_snapshot(scheduler)
        assert [item[1] for item in first_window] == ["stream"]

        # The forward pinned cuda:1; probe, wait and flush must start from cuda:0.
        torch.cuda.set_device(0)
        _feed(scheduler, "req-1", range(20, 21), chunks=chunks)  # probe: in flight
        assert torch.cuda.current_device() == 0
        assert state.pending is pending
        assert scheduler.outbox.qsize() == 0

        pending.slot.synchronize()
        assert torch.cuda.current_device() == 0
        assert pending.slot.query() is True
        assert torch.cuda.current_device() == 0

        _feed(scheduler, "req-1", range(21, 22), chunks=chunks)  # True: flush
        assert torch.cuda.current_device() == 0
        assert state.pending is None
        second_window = _drain_snapshot(scheduler)
        assert [item[1] for item in second_window] == ["stream"]
        assert scheduler._pinned_free == [pending.slot]

        scheduler._on_done("req-1")
        tail = _drain_snapshot(scheduler)
        assert [item[1] for item in tail] == ["stream", "result"]
        assert [*first_window, *second_window, *tail] == control_snapshot

        # Note (jiannan-17): the reap and the shutdown drain run on whichever
        # thread pumps or stops the stage.
        _seed(scheduler, "req-2")
        _feed(scheduler, "req-2", range(20), chunks=chunks)
        retired = scheduler._stream_states["req-2"].pending
        assert retired is not None
        torch.cuda.set_device(0)
        scheduler.abort("req-2")
        assert scheduler._pinned_retired == [retired.slot]
        scheduler._reap_retired_slots()
        assert torch.cuda.current_device() == 0
        assert scheduler._pinned_retired == [retired.slot], "still in flight"
        scheduler.on_serving_stop()
        assert torch.cuda.current_device() == 0
        assert scheduler._pinned_retired == []
        assert scheduler._pinned_quarantined == []
        assert retired.slot in scheduler._pinned_free
        assert retired.slot.query() is True
    finally:
        torch.cuda.set_device(previous_device)
