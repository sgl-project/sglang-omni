# SPDX-License-Identifier: Apache-2.0
"""Merge helpers for LLaDA2-Uni pipelines."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniEvent


def decode_text_output(*, thinker_out: dict[str, Any], tokenizer: Any) -> str:
    """Decode terminal output and apply SGLang's default stop trimming."""
    output_ids = list(thinker_out.get("output_ids") or [])
    finish_reason_data = thinker_out.get("finish_reason_data")
    matched_stop: Any = None
    if (
        isinstance(finish_reason_data, dict)
        and finish_reason_data.get("type") == "stop"
    ):
        matched_stop = finish_reason_data.get("matched")

    if isinstance(matched_stop, int):
        if output_ids and output_ids[-1] == matched_stop:
            output_ids.pop()
    elif isinstance(matched_stop, (list, tuple)):
        matched_token_ids = [int(token_id) for token_id in matched_stop]
        if (
            matched_token_ids
            and output_ids[-len(matched_token_ids) :] == matched_token_ids
        ):
            del output_ids[-len(matched_token_ids) :]

    if not output_ids:
        return ""

    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    if isinstance(matched_stop, str) and matched_stop:
        stop_pos = text.find(matched_stop)
        if stop_pos >= 0:
            text = text[:stop_pos]
    return text


def decode_events(
    *,
    thinker_out: dict[str, Any],
    tokenizer: Any,
) -> list[LLaDA2UniEvent]:
    """Convert the terminal thinker output to a text event.

    Incremental blocks are decoded by the stream-aware decode scheduler; this
    function remains the canonical terminal reconstruction used by both modes.
    """
    output_ids = thinker_out.get("output_ids", [])
    if not output_ids:
        return []

    text = decode_text_output(thinker_out=thinker_out, tokenizer=tokenizer)

    return [
        LLaDA2UniEvent(
            type="text_final",
            modality="text",
            payload={"text": text},
            is_final=True,
        )
    ]
