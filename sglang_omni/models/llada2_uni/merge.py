# SPDX-License-Identifier: Apache-2.0
"""Merge helpers for LLaDA2-Uni pipelines."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.llada2_uni.payload_types import OmniEvent


def decode_events(
    *,
    thinker_out: dict[str, Any],
    tokenizer: Any,
) -> list[OmniEvent]:
    """Convert thinker output tokens to a text_final event."""
    # TODO: add streaming support
    output_ids = thinker_out.get("output_ids", [])
    if not output_ids:
        return []

    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return [
        OmniEvent(
            type="text_final",
            modality="text",
            payload={"text": text},
            is_final=True,
        )
    ]
