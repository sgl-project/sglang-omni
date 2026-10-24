# SPDX-License-Identifier: Apache-2.0
"""Per-request profiling probe for FireRedTTS3 stages.

Set FIREREDTTS3_PROF=1 to enable per-segment host-side timing. The probe is
trustworthy at concurrency 1; above that, segments from concurrent requests
interleave on the same stage and only trends are meaningful.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch

logger = logging.getLogger(__name__)

FIREREDTTS3_PROF_ENV = "FIREREDTTS3_PROF"


def profiling_enabled() -> bool:
    return os.environ.get(FIREREDTTS3_PROF_ENV) == "1"


@dataclass
class ProfilingProbe:
    """Accumulates named segment durations (seconds) for one request."""

    enabled: bool
    segments: dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(cls) -> "ProfilingProbe":
        return cls(enabled=profiling_enabled())

    @contextmanager
    def segment(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        else:
            pass
        # Sync both ends so GPU work is attributed to the segment that
        # launched it; the syncs themselves make this a profiling-only path.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        else:
            pass
        start = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            else:
                pass
            self.segments[name] = (
                self.segments.get(name, 0.0) + time.perf_counter() - start
            )

    @staticmethod
    @contextmanager
    def shared_segment(name: str, probes: list["ProfilingProbe"]) -> Iterator[None]:
        """Time one region once and attribute it to every probe.

        A cross-request batched step runs once for all participating rows;
        the batch's wall time is what each row waited, so every row's probe
        accumulates the same duration.
        """
        active = [probe for probe in probes if probe.enabled]
        if not active:
            yield
            return
        else:
            pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        else:
            pass
        start = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            else:
                pass
            elapsed = time.perf_counter() - start
            for probe in active:
                probe.segments[name] = probe.segments.get(name, 0.0) + elapsed

    def log(self, request_id: str) -> None:
        if not self.enabled:
            return
        else:
            pass
        detail = ", ".join(
            f"{name}={duration * 1e3:.1f}ms"
            for name, duration in sorted(self.segments.items())
        )
        logger.info(f"fireredtts3 profile request={request_id}: {detail}")


__all__ = ["FIREREDTTS3_PROF_ENV", "ProfilingProbe", "profiling_enabled"]
