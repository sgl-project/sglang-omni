# SPDX-License-Identifier: Apache-2.0
"""Merge helpers for LLaDA2-Uni pipelines."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.llada2_uni.payload_types import (
    LLaDA2UniEvent,
    LLaDA2UniPipelineState,
)


def extract_image_vq_tokens(
    state: LLaDA2UniPipelineState,
) -> tuple[list[int], int, int, dict[str, Any]] | None:
    """Extract a complete native-image grid from thinker output.

    The vocabulary offset and grid are checkpoint/request state, not protocol
    constants. An incomplete grid is rejected rather than silently reshaped.
    """
    if state.task_kind not in {"t2i", "edit"}:
        return None
    if state.image_token_offset is None:
        raise ValueError("native image result is missing image_token_offset")

    thinker_out = state.thinker_out or state.engine_outputs.get("thinker")
    if not isinstance(thinker_out, dict):
        return None
    output_ids = thinker_out.get("output_ids")
    if not isinstance(output_ids, (list, tuple)):
        return None

    offset = state.image_token_offset
    vq_tokens = [
        int(token_id) - offset
        for token_id in output_ids
        if isinstance(token_id, int)
        and not isinstance(token_id, bool)
        and token_id >= offset
    ]
    grid = state.generation_state.get("image_grid")
    if not isinstance(grid, dict):
        raise ValueError("native image result is missing image_grid")
    height = grid.get("height")
    width = grid.get("width")
    if (
        not isinstance(height, int)
        or isinstance(height, bool)
        or height < 1
        or not isinstance(width, int)
        or isinstance(width, bool)
        or width < 1
    ):
        raise ValueError("native image grid dimensions must be positive integers")
    expected = height * width
    if len(vq_tokens) != expected:
        raise ValueError(
            f"native image expected {expected} VQ tokens for grid "
            f"{height}x{width}, got {len(vq_tokens)}"
        )

    metadata = state.request_metadata.get("image_generation")
    params = dict(metadata) if isinstance(metadata, dict) else {}
    return vq_tokens, height, width, params


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
