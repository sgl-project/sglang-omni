# SPDX-License-Identifier: Apache-2.0
"""Engine request/response helpers shared by omni pipelines."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.engines.omni.runtime import ARRequestData, EncoderRequestData
from sglang_omni.models.omni_base.types import ThinkerOutput
from sglang_omni.proto import StagePayload


def build_encoder_request(
    payload: StagePayload, *, stage_name: str
) -> EncoderRequestData:
    """Build encoder request from payload."""
    encoder_inputs = payload.encoder_inputs or {}
    inputs = encoder_inputs.get(stage_name)

    if not isinstance(inputs, dict) or not inputs:
        return EncoderRequestData(input_dict={"_skip": True, "_result": {}})

    if inputs.get("_skip"):
        skip_result = inputs.get("_result")
        return EncoderRequestData(
            input_dict=inputs,
            output_dict=skip_result if isinstance(skip_result, dict) else {},
        )

    cache_key = inputs.get("cache_key")
    return EncoderRequestData(
        input_dict=inputs,
        cache_key=str(cache_key) if cache_key is not None else None,
    )


def apply_encoder_result(
    payload: StagePayload,
    *,
    stage_name: str,
    result: Any,
) -> None:
    """Apply encoder result to payload."""
    if isinstance(result, EncoderRequestData):
        if result.output_dict is not None:
            encoder_out = result.output_dict
        elif result.embeddings is not None:
            encoder_out = result.embeddings
        else:
            encoder_out = {}
    else:
        encoder_out = result if isinstance(result, dict) else {"result": result}

    # Lazy initialization
    if payload.encoder_outs is None:
        payload.encoder_outs = {}
    if payload.engine_outputs is None:
        payload.engine_outputs = {}

    payload.encoder_outs[stage_name] = encoder_out
    payload.engine_outputs[stage_name] = encoder_out


def build_thinker_request(
    payload: StagePayload,
    *,
    params: dict[str, Any],
) -> ARRequestData:
    """Build thinker request from payload."""
    prompt = payload.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")

    attention_mask = prompt.get("attention_mask")
    thinker_inputs = payload.thinker_inputs or {}

    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v for k, v in thinker_inputs.items() if k != "capture_model_output_keys"
        }

    capture_keys = thinker_inputs.get("capture_model_output_keys", ())
    if "attention_mask" in model_inputs:
        model_inputs.pop("attention_mask", None)

    return ARRequestData(
        input_ids=input_ids.to(dtype=torch.long),
        attention_mask=(
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        ),
        model_inputs=model_inputs,
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=params.get("max_new_tokens"),
        temperature=params.get("temperature", 0.0),
    )


def apply_thinker_result(
    payload: StagePayload,
    *,
    stage_name: str,
    result: Any,
) -> ThinkerOutput:
    """Apply thinker result to payload."""
    if isinstance(result, ARRequestData):
        output_ids = list(result.output_ids)
        thinker_out: ThinkerOutput = {
            "output_ids": output_ids,
            "step": len(output_ids),
            "is_final": True,
            "extra_model_outputs": dict(result.extra_model_outputs),
        }
    else:
        thinker_out = {
            "output_ids": [],
            "step": 0,
            "is_final": True,
            "extra_model_outputs": {"result": result},
        }

    payload.thinker_out = thinker_out

    # Lazy initialization for engine_outputs
    if payload.engine_outputs is None:
        payload.engine_outputs = {}
    payload.engine_outputs[stage_name] = thinker_out

    return thinker_out
