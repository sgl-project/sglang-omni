# SPDX-License-Identifier: Apache-2.0
"""Usage accounting helpers for Ming-Omni pipeline outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _mapping_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _count_ids(ids: Any) -> int:
    if ids is None:
        return 0
    if hasattr(ids, "numel"):
        return int(ids.numel())
    try:
        return len(ids)
    except TypeError:
        return 0


def build_text_usage(
    state: Any,
    thinker_out: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Build OpenAI-style token usage for Ming thinker text generation."""

    prompt = _mapping_get(state, "prompt", None)

    resolved_thinker_out = thinker_out
    if resolved_thinker_out is None:
        candidate = _mapping_get(state, "thinker_out", None)
        resolved_thinker_out = candidate if isinstance(candidate, Mapping) else {}

    prompt_tokens = _count_ids(_mapping_get(prompt, "input_ids"))
    completion_tokens = _count_ids(resolved_thinker_out.get("output_ids"))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
