# SPDX-License-Identifier: Apache-2.0
"""Reusable pinned host staging buffers and CUDA completion events.

Streaming decoders copy device results into pinned host memory asynchronously
and wait on a CUDA event before reading them back. The two classes here hold
just the buffer and the event and carry no ownership policy: the owner
serializes access, grows a buffer only while no asynchronous copy can still be
using it, and must not touch a slot between ``record()`` and observed
completion (a successful ``synchronize()``, or a ``query()`` that reported
True).
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch


def allocate_pinned(numel: int, dtype: torch.dtype) -> torch.Tensor:
    # Note (jiannan-17): allocate outside inference mode even when the caller
    # is inside it, so the buffer is an ordinary tensor that can be filled
    # under inference mode and cloned or mutated outside it later.
    with torch.inference_mode(False):
        return torch.empty(numel, dtype=dtype, pin_memory=True)


def normalize_device(device: torch.device | str | int) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    else:
        pass
    return resolved


class GrowablePinnedBuffer:
    """Pinned host buffer that only grows, and by exactly the requested size.

    ``view()`` never allocates. Call ``ensure_capacity()`` first, while no
    asynchronous copy can still be using the current storage.
    """

    def __init__(self, dtype: torch.dtype, *, initial_capacity: int = 0) -> None:
        if initial_capacity < 0:
            raise ValueError("initial_capacity must be >= 0")
        else:
            pass
        self.dtype = dtype
        self.storage: torch.Tensor | None = None
        if initial_capacity:
            self.ensure_capacity(initial_capacity)
        else:
            pass

    @property
    def capacity(self) -> int:
        return 0 if self.storage is None else int(self.storage.numel())

    def ensure_capacity(self, required: int) -> None:
        """Grow to ``required`` elements. On failure the old storage is kept."""
        if required < 0:
            raise ValueError("required capacity must be >= 0")
        else:
            pass
        if required <= self.capacity:
            return
        else:
            pass
        storage = allocate_pinned(required, self.dtype)
        self.storage = storage

    def view(self, numel: int) -> torch.Tensor:
        """Return the first ``numel`` elements without allocating pinned memory."""
        if numel < 0 or numel > self.capacity:
            raise ValueError(
                f"requested {numel} elements from a pinned buffer with capacity "
                f"{self.capacity}"
            )
        else:
            pass
        if self.storage is None:
            with torch.inference_mode(False):
                return torch.empty(0, dtype=self.dtype)
        else:
            pass
        return self.storage[:numel]


class PinnedTransferSlot:
    """One growable pinned host buffer plus one reusable CUDA event.

    The event fences work queued before ``record()``. Do not resize or reuse
    the buffer until ``synchronize()`` returns or ``query()`` reports True.

    If ``record()`` raises, completion reads fail until a later ``record()``
    succeeds. Copy failures before ``record()`` remain the owner's
    responsibility.
    """

    def __init__(
        self,
        device: torch.device | str,
        dtype: torch.dtype,
        *,
        initial_capacity: int = 0,
    ) -> None:
        self.device = normalize_device(device)
        self.buffer = GrowablePinnedBuffer(dtype, initial_capacity=initial_capacity)
        self.event: Any = None
        # Note (jiannan-17): True only while the most recent ``record()``
        # succeeded. The event object alone cannot tell "never recorded" from
        # "the last record() raised", and CUDA reports an event whose record
        # never happened as already complete, which would hand the owner a
        # completion marker for a transfer that was never fenced.
        self.recorded = False

    @property
    def capacity(self) -> int:
        return self.buffer.capacity

    def ensure_capacity(self, required: int) -> None:
        self.buffer.ensure_capacity(required)

    def view(self, numel: int) -> torch.Tensor:
        return self.buffer.view(numel)

    def device_guard(self) -> contextlib.AbstractContextManager[Any]:
        if self.device.type == "cuda":
            return torch.cuda.device(self.device)
        else:
            pass
        return contextlib.nullcontext()

    def record(self, stream: Any) -> None:
        """Record the completion event on ``stream``.

        ``stream`` must live on this slot's device; the event is created on
        first use and reused for every later ``record()``. If this raises,
        the slot has no recorded transfer until a later ``record()``
        succeeds.
        """
        # Note (jiannan-17): cleared before anything can fail, so neither a
        # rejected stream nor a failed CUDA record can leave the previous
        # transfer's completion state readable as this transfer's.
        self.recorded = False
        stream_device = getattr(stream, "device", None)
        if stream_device is not None and normalize_device(stream_device) != self.device:
            raise ValueError(
                f"cannot record a transfer slot on {self.device} from a stream on "
                f"{stream_device}"
            )
        else:
            pass
        with self.device_guard():
            if self.event is None:
                self.event = torch.cuda.Event()
            else:
                pass
            self.event.record(stream)
        self.recorded = True

    def recorded_event(self) -> Any:
        if not self.recorded:
            raise RuntimeError(
                "transfer event was not recorded: no record() has succeeded on "
                "this slot since it was created or since its last record() raised"
            )
        else:
            pass
        return self.event

    def query(self) -> bool:
        """Return whether the recorded event has completed, without blocking.

        Raises ``RuntimeError`` until a ``record()`` has succeeded.
        """
        event = self.recorded_event()
        with self.device_guard():
            return bool(event.query())

    def synchronize(self) -> None:
        """Block until the recorded event has completed.

        Raises ``RuntimeError`` until a ``record()`` has succeeded.
        """
        event = self.recorded_event()
        with self.device_guard():
            event.synchronize()


__all__ = ["GrowablePinnedBuffer", "PinnedTransferSlot"]
