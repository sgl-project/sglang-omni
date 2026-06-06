# SPDX-License-Identifier: Apache-2.0
"""Shared runtime helpers for TTS model components."""

from __future__ import annotations

from typing import Any


def require_batch_result_count(
    *,
    owner: str,
    result_label: str,
    actual: int,
    expected: int,
) -> None:
    if actual != expected:
        raise RuntimeError(
            f"{owner} returned {actual} {result_label} for {expected} requests"
        )


def build_tts_usage(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    engine_time_s: float,
) -> dict[str, Any] | None:
    if not (prompt_tokens or completion_tokens or engine_time_s):
        return None
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if engine_time_s:
        usage["engine_time_s"] = round(engine_time_s, 6)
    return usage
