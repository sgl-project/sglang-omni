# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class _BatchStats:
    count: int = 0
    requests: int = 0
    graph_replays: int = 0
    graph_forwards: int = 0


class RuntimeStats:
    def __init__(self, stage: str, *, interval_s: float | None = None) -> None:
        self.stage = stage
        self.interval_s = (
            float(os.environ.get("SGLANG_OMNI_STATS_INTERVAL", "10"))
            if interval_s is None
            else interval_s
        )
        if not math.isfinite(self.interval_s) or self.interval_s < 0:
            raise ValueError(
                "SGLANG_OMNI_STATS_INTERVAL must be finite and nonnegative"
            )
        self._batches: dict[str, _BatchStats] = defaultdict(_BatchStats)
        self._queue_count = 0
        self._queue_sum_s = 0.0
        self._queue_max_s = 0.0
        self._started_at: float | None = None
        self._cpu_started_at = 0.0
        self._thread_id: int | None = None

    def record_batch(self, mode: str, size: int, *, graph: bool | None = None) -> None:
        if self.interval_s == 0:
            return
        if self._started_at is None:
            self._started_at = time.perf_counter()
            self._cpu_started_at = time.thread_time()
            self._thread_id = threading.get_ident()
        counts = self._batches[mode]
        counts.count += 1
        counts.requests += size
        counts.graph_replays += int(graph is True)
        counts.graph_forwards += int(graph is not None)

    def record_queue_wait(self, seconds: float) -> None:
        if self.interval_s == 0:
            return
        seconds = max(seconds, 0.0)
        self._queue_count += 1
        self._queue_sum_s += seconds
        self._queue_max_s = max(self._queue_max_s, seconds)

    def maybe_log(self, *, force: bool = False) -> None:
        if self.interval_s == 0 or self._started_at is None or not self._batches:
            return
        now = time.perf_counter()
        elapsed = now - self._started_at
        if not force and elapsed < self.interval_s:
            return
        cpu_now = time.thread_time()
        report = {
            "stage": self.stage,
            "window_s": elapsed,
            "scheduler_thread_cpu_s": (
                cpu_now - self._cpu_started_at
                if threading.get_ident() == self._thread_id
                else None
            ),
            "batches": {
                mode: {
                    "count": counts.count,
                    "mean_size": counts.requests / counts.count,
                    "graph_replays": counts.graph_replays,
                    "graph_forwards": counts.graph_forwards,
                    "graph_replay_rate": (
                        counts.graph_replays / counts.graph_forwards
                        if counts.graph_forwards
                        else None
                    ),
                }
                for mode, counts in self._batches.items()
            },
            "queue_wait_count": self._queue_count,
            "queue_wait_mean_ms": (
                1000 * self._queue_sum_s / self._queue_count
                if self._queue_count
                else None
            ),
            "queue_wait_max_ms": (
                1000 * self._queue_max_s if self._queue_count else None
            ),
        }
        logger.info("stage_stats %s", json.dumps(report, separators=(",", ":")))
        self._batches.clear()
        self._queue_count = 0
        self._queue_sum_s = self._queue_max_s = 0.0
        self._started_at = now
        self._cpu_started_at = cpu_now
