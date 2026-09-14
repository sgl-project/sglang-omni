# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pinned staging buffer and transfer slot primitives.

``torch.cuda.Event`` is replaced with a CPU stand-in, and so is pinned
allocation when no CUDA device is present, so the growth, inference-mode,
event-reuse, and error-propagation contracts can be checked without a GPU.
The ``accelerator``-marked cases run the real pinned/event path: an
asynchronous D2H copy observed through ``query()``, and the device guard with
the slot on a device other than the process-current one.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.utils import cuda_staging
from sglang_omni.utils.cuda_staging import GrowablePinnedBuffer, PinnedTransferSlot
from tests.unit_test.fixtures.accelerator import require_cuda


class _FakeEvent:
    def __init__(self) -> None:
        self.recorded_streams: list = []
        self.synchronize_calls = 0
        self.query_calls = 0
        # Note (jiannan-17): CUDA reports an unrecorded event as complete; the
        # stand-in matches.
        self.complete = True
        self.record_error: BaseException | None = None
        self.sync_error: BaseException | None = None
        self.query_error: BaseException | None = None

    def record(self, stream=None) -> None:
        if self.record_error is not None:
            raise self.record_error
        self.recorded_streams.append(stream)

    def query(self) -> bool:
        self.query_calls += 1
        if self.query_error is not None:
            raise self.query_error
        return self.complete

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        if self.sync_error is not None:
            raise self.sync_error


def _install_fake_events(
    monkeypatch, *, configure: Callable[[_FakeEvent], None] | None = None
) -> list[_FakeEvent]:
    created: list[_FakeEvent] = []

    def factory():
        event = _FakeEvent()
        if configure is not None:
            configure(event)
        created.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", factory)
    return created


def _install_fake_pinned_alloc(
    monkeypatch, *, fail_after: int | None = None
) -> list[tuple[int, torch.dtype]]:
    calls: list[tuple[int, torch.dtype]] = []

    def allocate(numel, dtype):
        if fail_after is not None and len(calls) >= fail_after:
            raise RuntimeError("pinned allocation failed")
        calls.append((numel, dtype))
        return torch.empty(numel, dtype=dtype)

    monkeypatch.setattr(cuda_staging, "_allocate_pinned", allocate)
    return calls


def test_growable_pinned_buffer_allocates_outside_inference_mode(monkeypatch):
    """A buffer grown under inference mode is still an ordinary tensor."""
    real_empty = torch.empty
    if not torch.cuda.is_available():
        # Pinned allocation needs a CUDA context; keep the real wrapper and
        # only drop the pin request.
        def cpu_empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        monkeypatch.setattr(torch, "empty", cpu_empty)

    buffer = GrowablePinnedBuffer(torch.float32)
    with torch.inference_mode():
        buffer.ensure_capacity(4)
        buffer.view(4).fill_(1.0)
    view = buffer.view(4)
    assert not view.is_inference()
    view.fill_(2.0)
    assert not view.clone().is_inference()
    assert torch.equal(view, torch.full((4,), 2.0))
    if torch.cuda.is_available():
        assert view.is_pinned()


def test_growable_pinned_buffer_grows_exactly_and_keeps_storage_on_failure(
    monkeypatch,
):
    calls = _install_fake_pinned_alloc(monkeypatch, fail_after=2)
    buffer = GrowablePinnedBuffer(torch.long)
    assert buffer.capacity == 0
    assert buffer.view(0).numel() == 0
    with pytest.raises(ValueError):
        buffer.view(1)
    assert calls == []

    buffer.ensure_capacity(4)
    buffer.view(4).fill_(7)
    buffer.ensure_capacity(3)
    assert calls == [(4, torch.long)], "smaller requests must not allocate"
    buffer.ensure_capacity(5)
    assert calls[-1] == (5, torch.long), "growth is exact, not geometric"
    assert buffer.capacity == 5
    buffer.view(5).fill_(1)
    storage_ptr = buffer.view(5).data_ptr()

    with pytest.raises(RuntimeError, match="pinned allocation failed"):
        buffer.ensure_capacity(8)
    assert buffer.capacity == 5
    assert buffer.view(5).data_ptr() == storage_ptr
    assert torch.equal(buffer.view(5), torch.ones(5, dtype=torch.long))
    with pytest.raises(ValueError):
        buffer.view(6)


def test_pinned_transfer_slot_reuses_one_event(monkeypatch):
    created = _install_fake_events(monkeypatch)
    _install_fake_pinned_alloc(monkeypatch)
    slot = PinnedTransferSlot("cpu", torch.float32, initial_capacity=8)
    stream = object()

    assert slot.capacity == 8
    for _ in range(3):
        slot.record(stream)
        slot.synchronize()
    slot.ensure_capacity(16)
    slot.record(stream)
    slot.synchronize()

    assert len(created) == 1, "the slot must reuse its event across transfers"
    assert created[0].recorded_streams == [stream] * 4
    assert created[0].synchronize_calls == 4
    assert slot.view(16).numel() == 16


def test_pinned_transfer_slot_query_probes_completion_without_blocking(monkeypatch):
    created = _install_fake_events(monkeypatch)
    _install_fake_pinned_alloc(monkeypatch)

    slot = PinnedTransferSlot("cpu", torch.float32)
    with pytest.raises(RuntimeError, match="not recorded"):
        slot.query()

    slot.record(object())
    created[0].complete = False
    assert slot.query() is False
    created[0].complete = True
    assert slot.query() is True
    assert created[0].synchronize_calls == 0, "query must not block on the event"

    query_error = RuntimeError("query failed")
    created[0].query_error = query_error
    with pytest.raises(RuntimeError) as query_info:
        slot.query()
    assert query_info.value is query_error
    # A failed probe leaves the record in place (owner policy).
    created[0].query_error = None
    assert slot.query() is True


def test_pinned_transfer_slot_first_record_failure_rejects_completion_reads(
    monkeypatch,
):
    """A slot whose only ``record()`` raised has no transfer to report on."""
    record_error = RuntimeError("event record failed")

    def _fail_record(event: _FakeEvent) -> None:
        event.record_error = record_error

    created = _install_fake_events(monkeypatch, configure=_fail_record)
    _install_fake_pinned_alloc(monkeypatch)
    slot = PinnedTransferSlot("cpu", torch.float32)

    with pytest.raises(RuntimeError) as record_info:
        slot.record(object())
    assert record_info.value is record_error
    assert len(created) == 1
    assert created[0].complete is True

    # The slot must not treat the failed transfer as complete.
    with pytest.raises(RuntimeError, match="not recorded"):
        slot.query()
    with pytest.raises(RuntimeError, match="not recorded"):
        slot.synchronize()
    assert (created[0].query_calls, created[0].synchronize_calls) == (0, 0)

    created[0].record_error = None
    stream = object()
    slot.record(stream)
    assert len(created) == 1
    assert created[0].recorded_streams == [stream]
    assert slot.query() is True
    slot.synchronize()
    assert created[0].synchronize_calls == 1


def test_pinned_transfer_slot_event_construction_failure_rejects_completion_reads(
    monkeypatch,
):
    """A failed ``torch.cuda.Event()`` leaves no event; the retry creates it."""
    created = _install_fake_events(monkeypatch)
    _install_fake_pinned_alloc(monkeypatch)
    slot = PinnedTransferSlot("cpu", torch.float32)
    init_error = RuntimeError("event init failed")
    factory = torch.cuda.Event

    def exploding_once():
        monkeypatch.setattr(torch.cuda, "Event", factory)
        raise init_error

    monkeypatch.setattr(torch.cuda, "Event", exploding_once)
    with pytest.raises(RuntimeError) as init_info:
        slot.record(object())
    assert init_info.value is init_error
    assert created == []
    with pytest.raises(RuntimeError, match="not recorded"):
        slot.query()
    with pytest.raises(RuntimeError, match="not recorded"):
        slot.synchronize()

    stream = object()
    slot.record(stream)
    assert len(created) == 1
    assert created[0].recorded_streams == [stream]
    assert slot.query() is True
    slot.synchronize()


def test_pinned_transfer_slot_failed_rerecord_hides_previous_completion(
    monkeypatch,
):
    """A failed re-record must not expose the previous transfer's completion."""
    created = _install_fake_events(monkeypatch)
    _install_fake_pinned_alloc(monkeypatch)
    slot = PinnedTransferSlot("cpu", torch.float32)

    first_stream = object()
    slot.record(first_stream)
    assert slot.query() is True
    slot.synchronize()
    assert created[0].synchronize_calls == 1

    record_error = RuntimeError("event record failed")
    created[0].record_error = record_error
    with pytest.raises(RuntimeError) as record_info:
        slot.record(object())
    assert record_info.value is record_error
    assert created[0].recorded_streams == [first_stream]
    assert created[0].complete is True

    # The slot must not treat the failed transfer as complete.
    with pytest.raises(RuntimeError, match="not recorded"):
        slot.query()
    with pytest.raises(RuntimeError, match="not recorded"):
        slot.synchronize()
    assert (created[0].query_calls, created[0].synchronize_calls) == (1, 1)

    created[0].record_error = None
    second_stream = object()
    slot.record(second_stream)
    assert created[0].recorded_streams == [first_stream, second_stream]
    assert slot.query() is True
    slot.synchronize()
    assert created[0].synchronize_calls == 2


def test_pinned_transfer_slot_propagates_errors_and_rejects_foreign_stream(
    monkeypatch,
):
    created = _install_fake_events(monkeypatch)
    _install_fake_pinned_alloc(monkeypatch)

    slot = PinnedTransferSlot("cpu", torch.float32)
    with pytest.raises(RuntimeError, match="not recorded"):
        slot.synchronize()

    slot.record(object())
    sync_error = RuntimeError("sync failed")
    created[0].sync_error = sync_error
    with pytest.raises(RuntimeError) as sync_info:
        slot.synchronize()
    assert sync_info.value is sync_error
    # A failed wait leaves the record in place (owner policy).
    created[0].sync_error = None
    slot.synchronize()
    assert created[0].synchronize_calls == 2

    guards: list[torch.device] = []

    @contextlib.contextmanager
    def fake_device_guard(device):
        guards.append(torch.device(device))
        yield

    monkeypatch.setattr(torch.cuda, "device", fake_device_guard)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    cuda_slot = PinnedTransferSlot("cuda", torch.float32)
    assert cuda_slot.device == torch.device("cuda", 0)
    with pytest.raises(ValueError):
        cuda_slot.record(SimpleNamespace(device=torch.device("cuda", 1)))
    assert len(created) == 1, "a rejected stream must not create an event"
    with pytest.raises(RuntimeError, match="not recorded"):
        cuda_slot.query()
    cuda_slot.record(SimpleNamespace(device=torch.device("cuda:0")))
    cuda_slot.query()
    cuda_slot.synchronize()
    assert len(created) == 2
    assert guards == [torch.device("cuda", 0)] * 3

    # A rejected stream is a failed record.
    with pytest.raises(ValueError):
        cuda_slot.record(SimpleNamespace(device=torch.device("cuda", 1)))
    with pytest.raises(RuntimeError, match="not recorded"):
        cuda_slot.query()
    with pytest.raises(RuntimeError, match="not recorded"):
        cuda_slot.synchronize()
    assert (created[1].query_calls, created[1].synchronize_calls) == (1, 1)
    assert guards == [torch.device("cuda", 0)] * 3


@pytest.mark.accelerator
def test_pinned_transfer_slot_real_cuda_query_tracks_async_copy() -> None:
    """``query()`` is False while the fenced D2H copy is in flight, then True."""
    require_cuda()
    device = torch.device("cuda", torch.cuda.current_device())
    # Note (jiannan-17): the contract's premise, checked against the driver:
    # an unrecorded event reports complete.
    unrecorded = torch.cuda.Event()
    assert unrecorded.query() is True
    unrecorded.synchronize()
    numel = 1 << 16
    slot = PinnedTransferSlot(device, torch.float32, initial_capacity=numel)
    assert slot.view(numel).is_pinned()
    with pytest.raises(RuntimeError, match="not recorded"):
        slot.query()

    source = torch.arange(numel, dtype=torch.float32, device=device)
    expected = source.cpu()
    stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)
    with torch.cuda.stream(stream):
        # Note (jiannan-17): ~0.5 s of queued device work keeps the copy in
        # flight; nothing between here and the probe touches CUDA.
        torch.cuda._sleep(1_000_000_000)
        slot.view(numel).copy_(source, non_blocking=True)
    slot.record(stream)

    assert slot.query() is False
    slot.synchronize()
    assert slot.query() is True
    assert torch.equal(slot.view(numel), expected)

    # The same event and buffer serve the next transfer.
    with torch.cuda.stream(stream):
        source.mul_(2)
        slot.view(numel).copy_(source, non_blocking=True)
    slot.record(stream)
    slot.synchronize()
    assert slot.query() is True
    assert torch.equal(slot.view(numel), expected * 2)


@pytest.mark.accelerator
def test_pinned_transfer_slot_real_cuda_guards_slot_on_other_device() -> None:
    """Record/query/synchronize work with the slot and stream on ``cuda:1``
    while the process-current device is ``cuda:0``, and leave it there."""
    require_cuda(min_devices=2)
    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        slot_device = torch.device("cuda", 1)
        numel = 1 << 16
        slot = PinnedTransferSlot(slot_device, torch.float32, initial_capacity=numel)
        assert slot.device == slot_device
        stream = torch.cuda.Stream(device=slot_device)
        source = torch.arange(numel, dtype=torch.float32, device=slot_device)
        expected = source.cpu()
        torch.cuda.synchronize(slot_device)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000_000)
            slot.view(numel).copy_(source, non_blocking=True)
        # Note (jiannan-17): the stream context restores cuda:0, so every slot
        # call below starts on the other device.
        assert torch.cuda.current_device() == 0

        slot.record(stream)
        assert torch.cuda.current_device() == 0
        assert slot.query() is False
        assert torch.cuda.current_device() == 0
        slot.synchronize()
        assert torch.cuda.current_device() == 0
        assert slot.query() is True
        assert torch.cuda.current_device() == 0
        assert torch.equal(slot.view(numel), expected)

        with pytest.raises(ValueError, match="cannot record"):
            slot.record(torch.cuda.current_stream(torch.device("cuda", 0)))
        assert torch.cuda.current_device() == 0
        with pytest.raises(RuntimeError, match="not recorded"):
            slot.query()
    finally:
        torch.cuda.set_device(previous_device)
