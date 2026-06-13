# SPDX-License-Identifier: Apache-2.0
"""Compatibility helpers for SGLang chunked-prefill request counters."""

from __future__ import annotations

from typing import Any


def get_inflight_middle_chunks(req: Any) -> int:
    return int(
        getattr(req, "inflight_middle_chunks", getattr(req, "is_chunked", 0)) or 0
    )


def _try_setattr(obj: Any, name: str, value: int) -> bool:
    try:
        setattr(obj, name, value)
    except AttributeError:
        return False
    return True


def set_inflight_middle_chunks(req: Any, value: int) -> None:
    value = max(int(value), 0)
    wrote = False
    if hasattr(req, "inflight_middle_chunks"):
        wrote = _try_setattr(req, "inflight_middle_chunks", value) or wrote
    if hasattr(req, "is_chunked"):
        wrote = _try_setattr(req, "is_chunked", value) or wrote
    if not wrote:
        _try_setattr(req, "inflight_middle_chunks", value)


def increment_inflight_middle_chunks(req: Any) -> None:
    set_inflight_middle_chunks(req, get_inflight_middle_chunks(req) + 1)


def decrement_inflight_middle_chunks(req: Any) -> None:
    set_inflight_middle_chunks(req, get_inflight_middle_chunks(req) - 1)
