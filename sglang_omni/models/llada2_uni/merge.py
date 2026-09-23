# SPDX-License-Identifier: Apache-2.0
"""Merge helpers for LLaDA2-Uni pipelines."""

from __future__ import annotations

import math
from typing import Any

from sglang_omni.models.llada2_uni.components.preprocessor import IMAGE_TOKEN_OFFSET
from sglang_omni.models.llada2_uni.config import THINKER_STAGE
from sglang_omni.models.llada2_uni.payload_types import (
    LLaDA2UniEvent,
    LLaDA2UniPipelineState,
)


def extract_image_vq_tokens(
    state: LLaDA2UniPipelineState,
) -> tuple[list[int], int, int, dict[str, Any]] | None:
    """Return decoder codebook IDs, semantic grid size, and generation options."""
    if state.task_kind not in ("t2i", "edit"):
        return None
    thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
    if not isinstance(thinker_out, dict):
        return None
    tokens = [
        int(tid) - IMAGE_TOKEN_OFFSET
        for tid in thinker_out.get("output_ids", [])
        if isinstance(tid, int) and tid >= IMAGE_TOKEN_OFFSET
    ]
    if not tokens:
        return None
    image_info = state.stream_state.get("image_info", [])
    if image_info:
        h, w = image_info[0].get("grid_h"), image_info[0].get("grid_w")
        if not (
            isinstance(h, int)
            and isinstance(w, int)
            and h > 0
            and w > 0
            and h * w == len(tokens)
        ):
            raise ValueError(
                f"Image output grid {h}x{w} does not match {len(tokens)} VQ tokens"
            )
    else:
        h = w = math.isqrt(len(tokens))
        if h * w != len(tokens):
            raise ValueError(f"Cannot infer an image grid from {len(tokens)} VQ tokens")
    params = state.request_metadata.get("image_generation", {})
    return tokens, h, w, params if isinstance(params, dict) else {}


def decode_events(
    *,
    thinker_out: dict[str, Any],
    tokenizer: Any,
) -> list[LLaDA2UniEvent]:
    """Convert thinker output tokens to a text_final event."""
    # TODO: add streaming support
    output_ids = thinker_out.get("output_ids", [])
    if not output_ids:
        return []

    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return [
        LLaDA2UniEvent(
            type="text_final",
            modality="text",
            payload={"text": text},
            is_final=True,
        )
    ]
