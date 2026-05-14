# SPDX-License-Identifier: Apache-2.0
"""Decode helpers for LLaDA2-Uni pipelines."""

from __future__ import annotations

from typing import Any, Iterable

from sglang_omni.models.llada2_uni.payload_types import (
    OmniEvent,
    PipelineState,
    ThinkerOutput,
)


def decode_events(
    *,
    thinker_out: ThinkerOutput,
    state: PipelineState,
    tokenizer: Any,
    eos_token_id: int | None,
) -> Iterable[OmniEvent]:
    """Convert thinker output tokens to text events."""
    output_ids = thinker_out.get("output_ids", [])
    if not output_ids:
        return []

    stream_state = state.stream_state
    token_ids = stream_state.setdefault("token_ids", [])
    stream_state.setdefault("text", "")
    stream_state.setdefault("emitted_text", "")

    is_final = bool(thinker_out.get("is_final"))

    if is_final:
        tokens = [
            int(t)
            for t in output_ids
            if eos_token_id is None or int(t) != int(eos_token_id)
        ]
        text = tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
        stream_state["token_ids"] = tokens
        stream_state["text"] = text
        return [
            OmniEvent(
                type="text_final",
                modality="text",
                payload={"text": text},
                is_final=True,
            )
        ]

    token_id = int(output_ids[-1])
    if eos_token_id is not None and token_id == int(eos_token_id):
        text = str(stream_state.get("text", ""))
        return [
            OmniEvent(
                type="text_final",
                modality="text",
                payload={"text": text},
                is_final=True,
            )
        ]

    token_ids.append(token_id)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    stream_state["text"] = decoded

    if "\ufffd" in decoded:
        return []

    emitted_text = str(stream_state.get("emitted_text", ""))
    delta = decoded[len(emitted_text) :]
    if not delta:
        return []
    stream_state["emitted_text"] = decoded
    return [
        OmniEvent(
            type="text_delta", modality="text", payload={"text": delta}, is_final=False
        )
    ]
