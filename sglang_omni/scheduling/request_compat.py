# SPDX-License-Identifier: Apache-2.0
"""Request compatibility helpers for SGLang-backed schedulers."""

from __future__ import annotations

from typing import Any

_UNSET = object()


def attach_sglang_req_compat(
    req: Any,
    *,
    tokenizer: Any = _UNSET,
    codec_suppress_tokens: Any = _UNSET,
    input_embeds_are_projected: Any = _UNSET,
) -> None:
    """Attach Omni compatibility attrs consumed by SGLang backend hooks."""

    if tokenizer is not _UNSET:
        req.tokenizer = tokenizer
    if codec_suppress_tokens is not _UNSET:
        req._codec_suppress_tokens = codec_suppress_tokens
    if input_embeds_are_projected is not _UNSET:
        req._input_embeds_are_projected = bool(input_embeds_are_projected)
