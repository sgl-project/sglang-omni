# SPDX-License-Identifier: Apache-2.0
"""Engine request/response helpers for Qwen3-ASR stages."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.engines.omni.runtime import ARRequestData, EncoderRequestData
from sglang_omni.models.qwen3_asr.io import PipelineState, ThinkerOutput


def build_encoder_request(
    state: PipelineState, *, stage_name: str
) -> EncoderRequestData:

    inputs = state.mm_inputs.get(stage_name)

    if not isinstance(inputs, dict) or not inputs:
        return EncoderRequestData(input_dict={"_skip": True, "_result": {}})

    if inputs.get("_skip"):
        skip_result = inputs.get("_result")
        return EncoderRequestData(
            input_dict=inputs,
            output_dict=skip_result if isinstance(skip_result, dict) else {},
        )
    return EncoderRequestData(input_dict=inputs)


def apply_encoder_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> None:
    if isinstance(result, EncoderRequestData):
        if result.output_dict is not None:
            encoder_out = result.output_dict
        elif result.embeddings is not None:
            encoder_out = result.embeddings
        else:
            encoder_out = {}
    else:
        encoder_out = result if isinstance(result, dict) else {"result": result}

    state.encoder_outs[stage_name] = encoder_out
    state.engine_outputs[stage_name] = encoder_out


def build_thinker_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
) -> ARRequestData:
    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")

    attention_mask = prompt.get("attention_mask")
    thinker_inputs = state.thinker_inputs or {}

    # Gather encoder outputs to pass to thinker
    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if "audio" in state.encoder_outs:
        model_inputs["audio_embeds"] = state.encoder_outs["audio"]["audio_embeds"]

    return ARRequestData(
        input_ids=input_ids.to(dtype=torch.long),
        attention_mask=(
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        ),
        model_inputs=model_inputs,
        max_new_tokens=params.get("max_new_tokens"),
        temperature=params.get("temperature", 0.0),
    )


def apply_thinker_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> ThinkerOutput:
    if isinstance(result, ARRequestData):
        output_ids = list(result.output_ids)
        thinker_out: ThinkerOutput = {
            "output_ids": output_ids,
            "step": len(output_ids),
            "is_final": True,
            "extra_model_outputs": dict(result.extra_model_outputs),
        }
    else:
        thinker_out = result if isinstance(result, dict) else {"result": result}
        if "is_final" not in thinker_out:
            thinker_out["is_final"] = True

    state.thinker_out = thinker_out
    state.engine_outputs[stage_name] = thinker_out
    return thinker_out
