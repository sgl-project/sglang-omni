# SPDX-License-Identifier: Apache-2.0
"""Usage accounting helpers for Ming-Omni pipeline outputs."""

from __future__ import annotations

from collections.abc import Mapping


def mapping_get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    else:
        pass
    return getattr(value, key, default)


def count_ids(ids: object) -> int:
    if ids is None:
        return 0
    else:
        pass
    if hasattr(ids, "numel"):
        return int(ids.numel())
    else:
        pass
    try:
        return len(ids)
    except TypeError:
        return 0


def build_text_usage(
    state: object,
    thinker_out: Mapping[str, object] | None = None,
) -> dict[str, int]:
    """Build OpenAI-style token usage for Ming thinker text generation."""

    prompt = mapping_get(state, "prompt", None)

    resolved_thinker_out = thinker_out
    if resolved_thinker_out is None:
        candidate = mapping_get(state, "thinker_out", None)
        resolved_thinker_out = candidate if isinstance(candidate, Mapping) else {}
    else:
        pass

    prompt_tokens = count_ids(mapping_get(prompt, "input_ids"))
    completion_tokens = count_ids(resolved_thinker_out.get("output_ids"))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
