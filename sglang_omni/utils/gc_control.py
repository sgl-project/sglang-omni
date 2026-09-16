# SPDX-License-Identifier: Apache-2.0
"""Python GC control helpers shared by the launcher, stage runtime and admin path.

``gc.freeze()`` moves every object currently tracked by the cyclic collector
into a permanent generation that later collections skip.  After model load and
CUDA graph capture a serving process holds millions of long-lived objects
(weights, graph runners, tokenizer tables); freezing them keeps gen2 collections
from re-scanning that static set on every request, which shows up as tail
latency on the scheduler thread.  Freezing is idempotent and never affects
correctness: objects created afterwards stay in the regular generations.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

FREEZE_GC_AFTER_STARTUP_ENV = "SGLANG_OMNI_FREEZE_GC_AFTER_STARTUP"


def freeze_gc_after_startup_enabled() -> bool:
    """Return whether the launcher should freeze GC once the pipeline is ready.

    Enabled by default; set ``SGLANG_OMNI_FREEZE_GC_AFTER_STARTUP=0`` to keep the
    pre-existing behaviour (no freeze).  ``POST /freeze_gc`` stays available
    either way.
    """
    raw = os.environ.get(FREEZE_GC_AFTER_STARTUP_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


FREEZE_GC_AFTER_REQUESTS_ENV = "SGLANG_OMNI_FREEZE_GC_AFTER_REQUESTS"
DEFAULT_FREEZE_GC_AFTER_REQUESTS = 64


def freeze_gc_after_requests() -> int:
    """How many completed requests to wait for before the second freeze.

    The first requests lazily build kernels, caches and per-model state that
    the startup freeze cannot see; freezing once more after they complete
    moves that set into the permanent generation as well.  ``0`` disables the
    second freeze.
    """
    raw = os.environ.get(FREEZE_GC_AFTER_REQUESTS_ENV, "").strip()
    if not raw:
        return DEFAULT_FREEZE_GC_AFTER_REQUESTS
    value = int(raw)
    if value < 0:
        raise ValueError(f"{FREEZE_GC_AFTER_REQUESTS_ENV} must be >= 0, got {value}")
    return value


def gc_object_counts() -> tuple[int, int, int]:
    return tuple(len(gc.get_objects(generation=i)) for i in range(3))  # type: ignore[return-value]


def freeze_gc(context: str) -> dict[str, Any]:
    """Freeze the cyclic GC in this process and return before/after generation sizes."""
    before = gc_object_counts()
    gc.freeze()
    after = gc_object_counts()
    frozen = gc.get_freeze_count()
    logger.info(
        "Freezing GC in %s process (pid=%d): gen0 %d->%d, gen1 %d->%d, gen2 %d->%d, frozen=%d",
        context,
        os.getpid(),
        before[0],
        after[0],
        before[1],
        after[1],
        before[2],
        after[2],
        frozen,
    )
    return {
        "context": context,
        "pid": os.getpid(),
        "before": {"gen0": before[0], "gen1": before[1], "gen2": before[2]},
        "after": {"gen0": after[0], "gen1": after[1], "gen2": after[2]},
        "frozen": frozen,
    }


GC_STATS_ENV = "SGLANG_OMNI_GC_STATS"
_GC_STATS_SUMMARY_INTERVAL_S = 30.0
_GC_STATS_SLOW_GEN2_S = 0.05


def gc_stats_enabled() -> bool:
    raw = os.environ.get(GC_STATS_ENV, "0").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def install_gc_stats_if_enabled(context: str) -> bool:
    """Opt-in diagnostic: count cyclic GC passes per generation in this process.

    With ``SGLANG_OMNI_GC_STATS=1`` a ``gc.callbacks`` hook accumulates the
    number, total and maximum duration of collections per generation, logs one
    summary line every 30 s of GC activity, and logs every gen2 pass slower
    than 50 ms on its own.  This is the evidence path for the post-startup
    freeze: compare gen2 count / stall time with and without it.  Off by
    default; the hook costs one ``time.monotonic()`` per collection.
    """
    if not gc_stats_enabled():
        return False
    import time

    state = {
        "start": {},
        "count": [0, 0, 0],
        "total": [0.0, 0.0, 0.0],
        "max": [0.0, 0.0, 0.0],
        "last_log": time.monotonic(),
    }

    def _cb(phase: str, info: dict[str, Any]) -> None:
        gen = int(info.get("generation", 0))
        now = time.monotonic()
        if phase == "start":
            state["start"][gen] = now
            return
        dur = now - state["start"].pop(gen, now)
        state["count"][gen] += 1
        state["total"][gen] += dur
        state["max"][gen] = max(state["max"][gen], dur)
        if gen == 2 and dur >= _GC_STATS_SLOW_GEN2_S:
            logger.info(
                "GC gen2 pass in %s (pid=%d): %.1f ms, collected=%s, frozen=%d",
                context,
                os.getpid(),
                dur * 1000.0,
                info.get("collected"),
                gc.get_freeze_count(),
            )
        if now - state["last_log"] >= _GC_STATS_SUMMARY_INTERVAL_S:
            state["last_log"] = now
            logger.info(
                "GC stats %s (pid=%d): gen0 n=%d tot=%.0fms max=%.1fms | "
                "gen1 n=%d tot=%.0fms max=%.1fms | gen2 n=%d tot=%.0fms max=%.1fms | frozen=%d",
                context,
                os.getpid(),
                state["count"][0],
                state["total"][0] * 1000.0,
                state["max"][0] * 1000.0,
                state["count"][1],
                state["total"][1] * 1000.0,
                state["max"][1] * 1000.0,
                state["count"][2],
                state["total"][2] * 1000.0,
                state["max"][2] * 1000.0,
                gc.get_freeze_count(),
            )

    gc.callbacks.append(_cb)
    logger.info("GC stats enabled in %s process (pid=%d)", context, os.getpid())
    return True
