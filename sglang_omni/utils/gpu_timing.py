# SPDX-License-Identifier: Apache-2.0
"""Hot-path-safe GPU span timing for shared-module contention analysis.

CUDA work is asynchronous: a Python call queues kernels and returns long before
the GPU runs them. A lock held only across kernel *launches* can therefore look
uncontended while the launched work still finishes late behind someone else's
kernels on the same stream. Lock counters alone cannot see that; this module
supplies the GPU-side half.

Two numbers are recorded per span:

``gpu_ms``
    Execution time between the start and end events, i.e. how long the GPU
    itself spent on the span.
``queue_ms``
    Delay between the CPU enqueuing the start event and the GPU reaching it.
    ``elapsed_time(start, end)`` deliberately excludes this, because the start
    event is itself queued behind whatever was already in the stream. When
    reference encoding delays streaming decode through the shared stream rather
    than through the lock, ``queue_ms`` is where it shows up.

No synchronization happens on the hot path. ``Event.record()`` is asynchronous;
``elapsed_time()`` is not, so it is deferred to :meth:`CudaSpanTimer.drain`,
which callers run off the critical path. Events are pre-allocated in a ring
buffer so steady-state recording performs no allocation.

Profiling is opt-in via ``SGLANG_OMNI_PROFILE_GPU_SPANS``; when disabled,
:meth:`CudaSpanTimer.span` returns a shared no-op context manager.
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

_ENV_FLAG = "SGLANG_OMNI_PROFILE_GPU_SPANS"

# Stateless and thread-safe, so one instance can serve every disabled call.
_DISABLED_SPAN: AbstractContextManager[None] = nullcontext()


def gpu_span_profiling_enabled() -> bool:
    """Read the opt-in flag. Accepts ``1``/``true``/``yes``/``on``."""
    raw = os.environ.get(_ENV_FLAG, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class GpuSpanStats:
    """Aggregate GPU execution and queueing delay for one call site."""

    spans: int = 0
    gpu_ms: float = 0.0
    max_gpu_ms: float = 0.0
    queue_ms: float = 0.0
    max_queue_ms: float = 0.0

    def observe(self, gpu_ms: float, queue_ms: float) -> None:
        self.spans += 1
        self.gpu_ms += gpu_ms
        self.max_gpu_ms = max(self.max_gpu_ms, gpu_ms)
        self.queue_ms += queue_ms
        self.max_queue_ms = max(self.max_queue_ms, queue_ms)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "spans": self.spans,
            "gpu_ms": round(self.gpu_ms, 4),
            "max_gpu_ms": round(self.max_gpu_ms, 4),
            "queue_ms": round(self.queue_ms, 4),
            "max_queue_ms": round(self.max_queue_ms, 4),
        }


@dataclass(frozen=True)
class _PendingSpan:
    label: str
    slot: int
    cpu_enqueue_s: float


def _default_event_factory() -> Any:
    import torch

    return torch.cuda.Event(enable_timing=True)


def _default_synchronize() -> None:
    import torch

    torch.cuda.synchronize()


def _cuda_is_available() -> bool:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency in serving
        return False
    return bool(torch.cuda.is_available())


class CudaSpanTimer:
    """Records GPU spans without synchronizing on the hot path."""

    def __init__(
        self,
        *,
        capacity: int = 2048,
        enabled: bool | None = None,
        event_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        synchronize: Callable[[], None] | None = None,
        available: Callable[[], bool] = _cuda_is_available,
    ) -> None:
        if capacity < 1:
            raise ValueError("CudaSpanTimer capacity must be positive")
        requested = gpu_span_profiling_enabled() if enabled is None else bool(enabled)
        explicit_backend = event_factory is not None
        self._enabled = requested and (explicit_backend or available())
        if requested and not self._enabled:
            logger.info("GPU span profiling requested but CUDA is unavailable")

        self._capacity = int(capacity)
        self._clock = clock
        self._event_factory = event_factory or _default_event_factory
        self._synchronize = synchronize or _default_synchronize
        self._counter = itertools.count()
        self._stats: dict[str, GpuSpanStats] = {}
        self._pending: deque[_PendingSpan] = deque()
        self._dropped = 0
        self._drain_lock = threading.Lock()

        self._starts: list[Any] = []
        self._ends: list[Any] = []
        self._epoch_event: Any = None
        self._epoch_cpu_s = 0.0
        if self._enabled:
            self._starts = [self._event_factory() for _ in range(self._capacity)]
            self._ends = [self._event_factory() for _ in range(self._capacity)]
            self._reset_epoch()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dropped(self) -> int:
        """Spans discarded because the ring wrapped before ``drain()``."""
        return self._dropped

    def _reset_epoch(self) -> None:
        epoch = self._event_factory()
        epoch.record()
        self._synchronize()
        self._epoch_event = epoch
        self._epoch_cpu_s = self._clock()

    def span(self, label: str) -> AbstractContextManager[None]:
        """Time one GPU span, attributing it to ``label``."""
        if not self._enabled:
            return _DISABLED_SPAN
        return self._recorded_span(label)

    @contextmanager
    def _recorded_span(self, label: str) -> Iterator[None]:
        slot = next(self._counter) % self._capacity
        cpu_enqueue_s = self._clock()
        self._starts[slot].record()
        try:
            yield
        finally:
            self._ends[slot].record()
            if len(self._pending) >= self._capacity:
                self._pending.popleft()
                self._dropped += 1
            self._pending.append(_PendingSpan(label, slot, cpu_enqueue_s))

    def _event_ready(self, event: Any) -> bool:
        query = getattr(event, "query", None)
        return True if query is None else bool(query())

    def drain(self, *, block: bool = False) -> None:
        """Fold completed spans into the stats.

        Non-blocking by default so periodic reporting preserves queueing;
        ``block=True`` is for final aggregation.
        """
        if not self._enabled:
            return
        with self._drain_lock:
            if not self._pending:
                return
            if block:
                self._synchronize()
            while self._pending:
                record = self._pending[0]
                end = self._ends[record.slot]
                if not block and not self._event_ready(end):
                    break  # still in flight; leave the rest for a later drain
                self._pending.popleft()
                start = self._starts[record.slot]
                gpu_ms = float(start.elapsed_time(end))
                gpu_start_cpu_s = (
                    self._epoch_cpu_s
                    + float(self._epoch_event.elapsed_time(start)) / 1000.0
                )
                queue_ms = (gpu_start_cpu_s - record.cpu_enqueue_s) * 1000.0
                site = self._stats.setdefault(record.label, GpuSpanStats())
                # Clock-domain skew can make a near-zero delay read slightly
                # negative; report the floor rather than a nonsensical value.
                site.observe(gpu_ms, max(queue_ms, 0.0))

    def stats(self, *, block: bool = False) -> dict[str, dict[str, float | int]]:
        self.drain(block=block)
        return {label: site.as_dict() for label, site in self._stats.items()}

    def reset(self) -> None:
        with self._drain_lock:
            self._pending.clear()
            self._stats.clear()
            self._dropped = 0

    def summary(self) -> str:
        snapshot = self.stats(block=True)
        if not snapshot:
            return "no GPU spans recorded"
        rows = sorted(snapshot.items(), key=lambda kv: -float(kv[1]["queue_ms"]))
        lines = [
            f"{'site':<28}{'spans':>8}{'gpu_ms':>12}{'max_gpu':>10}"
            f"{'queue_ms':>12}{'max_queue':>11}"
        ]
        for label, site in rows:
            lines.append(
                f"{label:<28}{site['spans']:>8}{site['gpu_ms']:>12.3f}"
                f"{site['max_gpu_ms']:>10.3f}{site['queue_ms']:>12.3f}"
                f"{site['max_queue_ms']:>11.3f}"
            )
        if self._dropped:
            lines.append(f"(dropped {self._dropped} spans: ring wrapped before drain)")
        return "\n".join(lines)


def span(timer: object, label: str) -> AbstractContextManager[None]:
    """Time a GPU span on ``timer``, or do nothing if it cannot.

    Mirrors ``lock_profile.labeled``: call sites stay agnostic about whether a
    timer was installed, so a codec built without one (or by a test) still runs.
    """
    record = getattr(timer, "span", None)
    if record is None:
        return _DISABLED_SPAN
    return record(label)


__all__ = [
    "CudaSpanTimer",
    "GpuSpanStats",
    "gpu_span_profiling_enabled",
    "span",
]
