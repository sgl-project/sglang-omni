# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pinned staging buffer and transfer slot primitives.

``torch.cuda.Event`` is replaced with a CPU stand-in, and so is pinned
allocation when no CUDA device is present, so the growth, inference-mode,
event-reuse, and error-propagation contracts can be checked without a GPU.
The real pinned/event path is exercised by the Qwen3-TTS CUDA tests.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.utils import cuda_staging
from sglang_omni.utils.cuda_staging import GrowablePinnedBuffer, PinnedTransferSlot


class _FakeEvent:
    def __init__(self) -> None:
        self.recorded_streams: list = []
        self.synchronize_calls = 0
        self.record_error: BaseException | None = None
        self.sync_error: BaseException | None = None

    def record(self, stream=None) -> None:
        if self.record_error is not None:
            raise self.record_error
        self.recorded_streams.append(stream)

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        if self.sync_error is not None:
            raise self.sync_error


def _install_fake_events(monkeypatch) -> list[_FakeEvent]:
    created: list[_FakeEvent] = []

    def factory():
        event = _FakeEvent()
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


def test_pinned_transfer_slot_propagates_errors_and_rejects_foreign_stream(
    monkeypatch,
):
    created = _install_fake_events(monkeypatch)
    _install_fake_pinned_alloc(monkeypatch)

    slot = PinnedTransferSlot("cpu", torch.float32)
    with pytest.raises(RuntimeError, match="not recorded"):
        slot.synchronize()

    slot.record(object())
    record_error = RuntimeError("record failed")
    sync_error = RuntimeError("sync failed")
    created[0].record_error = record_error
    with pytest.raises(RuntimeError) as record_info:
        slot.record(object())
    assert record_info.value is record_error
    created[0].record_error = None
    created[0].sync_error = sync_error
    with pytest.raises(RuntimeError) as sync_info:
        slot.synchronize()
    assert sync_info.value is sync_error

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
    cuda_slot.record(SimpleNamespace(device=torch.device("cuda:0")))
    cuda_slot.synchronize()
    assert len(created) == 2
    assert guards == [torch.device("cuda", 0)] * 2
