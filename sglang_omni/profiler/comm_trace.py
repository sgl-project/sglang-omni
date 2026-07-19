# SPDX-License-Identifier: Apache-2.0
"""Small env-gated communication trace helper."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger("sglang_omni.comm_trace")


def enabled() -> bool:
    value = os.getenv("SGLANG_OMNI_COMM_TRACE", "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def now_ns() -> int:
    return time.perf_counter_ns()


def elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def emit(event: str, **fields: Any) -> None:
    if not enabled():
        return
    record = {"event": event, "ts_ns": time.time_ns()}
    record.update(fields)
    logger.info("COMM_TRACE %s", json.dumps(record, sort_keys=True, default=str))
