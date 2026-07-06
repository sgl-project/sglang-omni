# SPDX-License-Identifier: Apache-2.0
"""Merge helpers for LLaDA2-Uni pipelines."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniEvent


def decode_events(
    *,
    thinker_out: dict[str, Any],
    tokenizer: Any,
    generation: dict[str, Any] | None = None,
) -> list[LLaDA2UniEvent]:
    """Convert thinker output tokens to terminal modality events."""
    # TODO: add streaming support
    output_ids = thinker_out.get("output_ids", [])
    if not output_ids:
        return []

    generation = generation if isinstance(generation, dict) else {}
    if generation.get("type") == "image":
        offset = int(generation.get("image_token_offset", 0))
        num_tokens = int(generation.get("num_image_tokens") or len(output_ids))
        image_token_ids = [int(t) - offset for t in output_ids[:num_tokens]]
        return [
            LLaDA2UniEvent(
                type="image_tokens_final",
                modality="image",
                payload={
                    "image_token_ids": image_token_ids,
                    "token_grid_h": generation.get("token_grid_h"),
                    "token_grid_w": generation.get("token_grid_w"),
                    "width": generation.get("width"),
                    "height": generation.get("height"),
                    "decoder_steps": generation.get("decoder_steps"),
                    "resolution_multiplier": generation.get("resolution_multiplier"),
                    "decode_mode": generation.get("decode_mode"),
                },
                is_final=True,
            )
        ]

    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return [
        LLaDA2UniEvent(
            type="text_final",
            modality="text",
            payload={"text": text},
            is_final=True,
        )
    ]
