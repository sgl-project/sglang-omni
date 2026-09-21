# SPDX-License-Identifier: Apache-2.0
"""Join encoder branches for the thinker, and build the decode-stage result."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.routing import THINKER_STAGE
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.streaming_detokenizer import Tokenizer


def merge_for_thinker(payloads: dict[str, StagePayload]) -> StagePayload:
    """Merge encoder embeddings into thinker inputs without duplicating payloads."""
    base = payloads.get("preprocessing") or next(iter(payloads.values()))
    state = MiniCPMOPipelineState.from_dict(base.data)

    model_inputs: dict[str, Any] = {}
    for payload in payloads.values():
        branch = MiniCPMOPipelineState.from_dict(payload.data)
        for encoder_out in branch.encoder_outs.values():
            if isinstance(encoder_out, dict):
                model_inputs.update(encoder_out)

    state.thinker_inputs = {"model_inputs": model_inputs}
    state.encoder_inputs = {}
    state.encoder_outs = {}
    return StagePayload(
        request_id=base.request_id,
        request=base.request,
        data=state.to_dict(),
    )


def thinker_output(state: MiniCPMOPipelineState) -> dict:
    thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
    if isinstance(thinker_out, dict):
        return thinker_out
    return {
        "output_ids": [],
        "step": 0,
        "is_final": True,
        "extra_model_outputs": {},
    }


def prompt_token_count(state: MiniCPMOPipelineState) -> int:
    if not isinstance(state.prompt, dict):
        return 0
    input_ids = state.prompt.get("input_ids")
    if input_ids is None:
        return 0
    if isinstance(input_ids, list):
        return len(input_ids)
    return int(input_ids.numel())


def build_decode_result(
    payload: StagePayload,
    *,
    tokenizer: Tokenizer,
    eos_token_id: int | None,
    is_streaming: bool,
) -> dict:
    state = MiniCPMOPipelineState.from_dict(payload.data)
    thinker_out = thinker_output(state)
    output_ids = thinker_out.get("output_ids") or []
    tokens = [
        int(token)
        for token in output_ids
        if eos_token_id is None or int(token) != int(eos_token_id)
    ]
    text = tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
    events = (
        [
            {
                "type": "text_final",
                "modality": "text",
                "payload": {"text": text},
                "is_final": True,
            }
        ]
        if tokens
        else []
    )
    result: dict = {"events": events}
    if events:
        result.update(events[-1]["payload"])
        result.setdefault("modality", "text")
    if is_streaming:
        result.pop("text", None)
    finish_reason = thinker_out.get("finish_reason")
    if finish_reason is not None:
        result.setdefault("finish_reason", finish_reason)
    prompt_tokens = prompt_token_count(state)
    completion_tokens = len(output_ids)
    result.setdefault(
        "usage",
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )
    return result
