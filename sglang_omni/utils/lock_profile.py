# SPDX-License-Identifier: Apache-2.0
"""Contention profiling for locks shared by pipeline stages.

Some models share one module bundle, and therefore one lock, across stages that
run on different threads. dots.tts is the current example: reference encoding
and vocoder decoding both serialize on ``DotsAudioCodec.lock``. Whether that
lock actually costs anything is an empirical question, so this profiler reports
per-call-site wait and hold time instead of guessing.

Profiling is opt-in. When disabled, ``labeled()`` returns the underlying lock
itself, so the hot streaming path pays one attribute read and no timing calls.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field

_ENV_FLAG = "SGLANG_OMNI_PROFILE_LOCKS"

UNLABELED = "unlabeled"


def lock_profiling_enabled() -> bool:
    """Read the opt-in flag. Accepts ``1``/``true``/``yes``, case-insensitive."""
    raw = os.environ.get(_ENV_FLAG, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class LockSiteStats:
    """Aggregate wait/hold timings for one labeled call site."""

    acquisitions: int = 0
    contended: int = 0
    wait_s: float = 0.0
    max_wait_s: float = 0.0
    hold_s: float = 0.0
    max_hold_s: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "acquisitions": self.acquisitions,
            "contended": self.contended,
            "wait_s": round(self.wait_s, 6),
            "max_wait_s": round(self.max_wait_s, 6),
            "hold_s": round(self.hold_s, 6),
            "max_hold_s": round(self.max_hold_s, 6),
        }


@dataclass
class _ThreadState:
    depth: int = 0
    frames: list[tuple[str, float, float] | None] = field(default_factory=list)


class ProfiledRLock:
    """Reentrant lock with per-label wait/hold timing.

    Behaves like ``threading.RLock``; only outermost acquires are measured.
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        contended_threshold_s: float = 1e-4,
    ) -> None:
        self._lock = threading.RLock()
        self._enabled = lock_profiling_enabled() if enabled is None else bool(enabled)
        self._contended_threshold_s = float(contended_threshold_s)
        self._stats: dict[str, LockSiteStats] = {}
        self._stats_lock = threading.Lock()
        self._thread = threading.local()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _thread_state(self) -> _ThreadState:
        state = getattr(self._thread, "state", None)
        if state is None:
            state = _ThreadState()
            self._thread.state = state
        return state

    def _record(self, label: str, wait_s: float, hold_s: float) -> None:
        with self._stats_lock:
            site = self._stats.get(label)
            if site is None:
                site = LockSiteStats()
                self._stats[label] = site
            site.acquisitions += 1
            site.wait_s += wait_s
            site.hold_s += hold_s
            if wait_s > site.max_wait_s:
                site.max_wait_s = wait_s
            if hold_s > site.max_hold_s:
                site.max_hold_s = hold_s
            if wait_s >= self._contended_threshold_s:
                site.contended += 1

    def _acquire(self, label: str) -> None:
        if not self._enabled:
            self._lock.acquire()
            return
        state = self._thread_state()
        if state.depth:
            self._lock.acquire()
            state.depth += 1
            state.frames.append(None)
            return
        wait_start = time.perf_counter()
        self._lock.acquire()
        held_at = time.perf_counter()
        state.depth = 1
        state.frames.append((label, wait_start, held_at))

    def _release(self) -> None:
        if not self._enabled:
            self._lock.release()
            return
        state = self._thread_state()
        frame = state.frames.pop() if state.frames else None
        if frame is None:
            state.depth = max(0, state.depth - 1)
            self._lock.release()
            return
        label, wait_start, held_at = frame
        released_at = time.perf_counter()
        state.depth = 0
        self._lock.release()
        self._record(label, held_at - wait_start, released_at - held_at)

    @contextmanager
    def _labeled_profiled(self, label: str) -> Iterator[None]:
        self._acquire(label)
        try:
            yield
        finally:
            self._release()

    def labeled(self, label: str) -> AbstractContextManager[None]:
        """Acquire the lock, attributing wait/hold time to ``label``."""
        if not self._enabled:
            return self._lock
        return self._labeled_profiled(label)

    def __enter__(self) -> "ProfiledRLock":
        self._acquire(UNLABELED)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._release()
        return False

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def stats(self) -> dict[str, dict[str, float | int]]:
        """Snapshot per-label timings. Empty when profiling is disabled."""
        with self._stats_lock:
            return {
                label: site.as_dict() for label, site in sorted(self._stats.items())
            }

    def reset(self) -> None:
        with self._stats_lock:
            self._stats.clear()

    def summary(self) -> str:
        """One-line-per-site report, ordered by total wait time."""
        snapshot = self.stats()
        if not snapshot:
            return "lock profiling disabled or no acquisitions recorded"
        rows = sorted(snapshot.items(), key=lambda kv: -float(kv[1]["wait_s"]))
        lines = [
            f"{'site':<28}{'acq':>8}{'contended':>11}"
            f"{'wait_s':>10}{'max_wait':>10}{'hold_s':>10}{'max_hold':>10}"
        ]
        for label, site in rows:
            lines.append(
                f"{label:<28}{site['acquisitions']:>8}{site['contended']:>11}"
                f"{site['wait_s']:>10.4f}{site['max_wait_s']:>10.4f}"
                f"{site['hold_s']:>10.4f}{site['max_hold_s']:>10.4f}"
            )
        return "\n".join(lines)


def labeled(lock: object, label: str) -> AbstractContextManager[None]:
    """Acquire ``lock``, attributing the wait to ``label`` when it can profile."""
    site = getattr(lock, "labeled", None)
    if site is None:
        return lock  # type: ignore[return-value]
    return site(label)


__all__ = [
    "labeled",
    "LockSiteStats",
    "ProfiledRLock",
    "lock_profiling_enabled",
    "UNLABELED",
]
