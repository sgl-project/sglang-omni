# SPDX-License-Identifier: Apache-2.0
"""Per-request host timing; concurrent requests can interleave stage timings."""

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
        # note (kermit): Synchronization attributes GPU work to its launching segment.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        else:
            pass
        start_seconds = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            else:
                pass
            self.segments[name] = (
                self.segments.get(name, 0.0) + time.perf_counter() - start_seconds
            )

    @staticmethod
    @contextmanager
    def shared_segment(name: str, probes: list["ProfilingProbe"]) -> Iterator[None]:
        """Record the batch wall time for each participating request."""
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
        start_seconds = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            else:
                pass
            elapsed_seconds = time.perf_counter() - start_seconds
            for probe in active:
                probe.segments[name] = probe.segments.get(name, 0.0) + elapsed_seconds

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
