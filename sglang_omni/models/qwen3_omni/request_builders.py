# SPDX-License-Identifier: Apache-2.0
"""Request builders for Qwen3-Omni split stages."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.engines.omni.runtime import ARRequestData
from sglang_omni.proto import StagePayload


def _data_dict(payload: StagePayload) -> dict[str, Any]:
    if not isinstance(payload.data, dict):
        raise TypeError(f"Expected payload.data to be dict, got {type(payload.data)}")
    return payload.data


def _engine_inputs(payload: StagePayload, key: str) -> dict[str, Any]:
    data = _data_dict(payload)
    engine_inputs = data.get("engine_inputs", {})
    if not isinstance(engine_inputs, dict):
        return {}
    value = engine_inputs.get(key, {})
    return value if isinstance(value, dict) else {}


def build_image_encoder_request(payload: StagePayload) -> dict[str, Any]:
    inputs = _engine_inputs(payload, "image_encoder")
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    if not isinstance(pixel_values, torch.Tensor) or not isinstance(image_grid_thw, torch.Tensor):
        return {
            "_skip": True,
            "_result": {
                "image_embeds": torch.empty(0),
                "image_grid_thw": torch.empty((0, 3), dtype=torch.long),
                "image_token_counts": torch.empty(0, dtype=torch.long),
            },
        }
    return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}


def build_audio_encoder_request(payload: StagePayload) -> dict[str, Any]:
    inputs = _engine_inputs(payload, "audio_encoder")
    input_features = inputs.get("input_features")
    feature_attention_mask = inputs.get("feature_attention_mask")
    audio_feature_lengths = inputs.get("audio_feature_lengths")
    if not isinstance(input_features, torch.Tensor):
        return {
            "_skip": True,
            "_result": {
                "audio_embeds": torch.empty(0),
                "audio_feature_lengths": torch.empty(0, dtype=torch.long),
                "audio_output_lengths": torch.empty(0, dtype=torch.long),
            },
        }
    return {
        "input_features": input_features,
        "feature_attention_mask": feature_attention_mask,
        "audio_feature_lengths": audio_feature_lengths,
    }


def build_thinker_ar_request(payload: StagePayload) -> ARRequestData:
    data = _data_dict(payload)
    prompt = data.get("prompt", {})
    if not isinstance(prompt, dict):
        raise TypeError("prompt must be a dict")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")
    attention_mask = prompt.get("attention_mask")

    thinker_inputs = _engine_inputs(payload, "thinker")
    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    capture_keys = thinker_inputs.get("capture_model_output_keys", ())

    if "attention_mask" in model_inputs:
        model_inputs.pop("attention_mask", None)

    req = ARRequestData(
        input_ids=input_ids.to(dtype=torch.long),
        attention_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
        model_inputs=model_inputs,
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=payload.request.params.get("max_new_tokens"),
        temperature=payload.request.params.get("temperature", 0.0),
    )
    return req
