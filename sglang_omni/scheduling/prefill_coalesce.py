# SPDX-License-Identifier: Apache-2.0
"""Shared validation for prefill admission coalescing."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def validate_prefill_coalesce_requests(value: object) -> int:
    """Validate a concrete admission threshold from CLI, YAML, or Python."""

    if value is None:
        raise ValueError(
            "prefill_coalesce_requests cannot be null; omit it to use the "
            "default or use 0 to disable coalescing"
        )
    # bool is an int subclass, so `true` would otherwise mean 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"prefill_coalesce_requests must be a number, got {value!r}")
    # Reject non-integral and non-finite counts before int() changes their
    # meaning: YAML `-0.5` would land as 0 (silently off) and `2.9` as 2, while
    # int(inf) raises OverflowError, which callers do not catch.
    if isinstance(value, float) and not (math.isfinite(value) and value.is_integer()):
        raise ValueError(
            f"prefill_coalesce_requests must be a finite integer, got {value!r}"
        )
    requests = int(value)
    if requests < 0:
        raise ValueError("prefill_coalesce_requests must be >= 0")
    if requests == 1:
        logger.warning(
            "prefill_coalesce_requests=1 disables coalescing: the admission "
            "gate only engages at >= 2 (a batch of one has nothing to "
            "coalesce with). Use 0 to disable explicitly, or >= 2 to enable."
        )
    return requests


def validate_prefill_coalesce_wait_ms(value: object) -> float:
    """Validate a concrete admission deadline from CLI, YAML, or Python."""

    if value is None:
        raise ValueError(
            "prefill_coalesce_wait_ms cannot be null; omit it to use the default"
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"prefill_coalesce_wait_ms must be a number, got {value!r}")
    try:
        wait_ms = float(value)
    except OverflowError as exc:
        raise ValueError("prefill_coalesce_wait_ms must be a finite value > 0") from exc
    if not (math.isfinite(wait_ms) and wait_ms > 0):
        raise ValueError("prefill_coalesce_wait_ms must be a finite value > 0")
    return wait_ms
