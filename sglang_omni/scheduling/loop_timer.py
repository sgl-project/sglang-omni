# SPDX-License-Identifier: Apache-2.0
"""Env-gated scheduler event-loop segment timer (scan tooling).

Enable with ``SGLANG_OMNI_LOOP_TIMER=1``; report interval via
``SGLANG_OMNI_LOOP_TIMER_INTERVAL_S`` (default 10). Emits one
``LOOP_TIMER {json}`` log line per interval with per-segment wall time,
call count, and share of window wall time.

An instance is owned by a single scheduler loop thread; it is not
thread-safe. The injected ``clock`` must be monotonic. With the timer
disabled the loops pay only a few local truthiness checks per iteration.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Callable

logger = logging.getLogger(__name__)


class LoopSegmentTimer:
    def __init__(
        self,
        interval_s: float = 10.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._acc: dict[str, float] = {}
        self._cnt: dict[str, int] = {}
        self._clock = clock
        now = clock()
        self._t_report = now
        self._t0_window = now
        # nan/inf would never report; <=0 would report every add(). Fail safe.
        if not math.isfinite(interval_s) or interval_s <= 0:
            interval_s = 10.0
        self._interval = interval_s

    def add(self, segment: str, dt: float) -> None:
        self._acc[segment] = self._acc.get(segment, 0.0) + dt
        self._cnt[segment] = self._cnt.get(segment, 0) + 1
        now = self._clock()
        if now - self._t_report >= self._interval:
            wall = max(now - self._t0_window, 1e-9)
            payload = {
                "wall_s": round(wall, 3),
                "segments": {
                    k: {
                        "s": round(v, 4),
                        "n": self._cnt[k],
                        "share": round(v / wall, 4),
                    }
                    for k, v in sorted(self._acc.items())
                },
            }
            logger.info("LOOP_TIMER %s", json.dumps(payload))
            self._acc = {}
            self._cnt = {}
            self._t_report = now
            self._t0_window = now


def maybe_loop_timer() -> LoopSegmentTimer | None:
    if os.environ.get("SGLANG_OMNI_LOOP_TIMER", "0") in ("", "0", "false", "False"):
        return None
    try:
        interval = float(os.environ.get("SGLANG_OMNI_LOOP_TIMER_INTERVAL_S", "10"))
    except ValueError:
        interval = 10.0
    return LoopSegmentTimer(interval)
