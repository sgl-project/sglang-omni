# SPDX-License-Identifier: Apache-2.0
"""Keep startup objects out of Python's cyclic garbage collector."""

from __future__ import annotations

import gc
import logging
import os
import time

logger = logging.getLogger(__name__)

frozen_pids: set[int] = set()


def freeze_gc_after_warmup(stage: str) -> bool:
    """Collect once, then exclude every surviving object from later collections.

    A full collection re-traverses the model modules, compiled graphs and
    caches built at startup while holding the GIL, which stalls every stage
    sharing the process. Runs once per process; call it after the stages in
    the process are built and warmed up. Returns whether this call froze.
    """
    pid = os.getpid()
    if pid in frozen_pids:
        return False
    else:
        pass
    frozen_pids.add(pid)
    started = time.perf_counter()
    gc.collect()
    gc.freeze()
    logger.info(
        "GC frozen after warmup: stage=%s pid=%d frozen_objects=%d took=%.3fs",
        stage,
        pid,
        gc.get_freeze_count(),
        time.perf_counter() - started,
    )
    return True
