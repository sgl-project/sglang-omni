# SPDX-License-Identifier: Apache-2.0
"""Result builders for Qwen3-Omni split stages."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.engines.omni.runtime import ARRequestData
from sglang_omni.proto import StagePayload


def _ensure_data(payload: StagePayload) -> dict[str, Any]:
    if not isinstance(payload.data, dict):
        payload.data = {}
    return payload.data


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def build_image_result(payload: StagePayload, result: Any) -> StagePayload:
    data = _ensure_data(payload)
    engine_outputs = data.setdefault("engine_outputs", {})
    engine_outputs["image_encoder"] = {
        "image_embeds": _to_cpu(result.get("image_embeds")) if isinstance(result, dict) else result,
        "image_grid_thw": _to_cpu(result.get("image_grid_thw")) if isinstance(result, dict) else None,
        "image_token_counts": _to_cpu(result.get("image_token_counts")) if isinstance(result, dict) else None,
    }
    return payload


def build_audio_result(payload: StagePayload, result: Any) -> StagePayload:
    data = _ensure_data(payload)
    engine_outputs = data.setdefault("engine_outputs", {})
    engine_outputs["audio_encoder"] = {
        "audio_embeds": _to_cpu(result.get("audio_embeds")) if isinstance(result, dict) else result,
        "audio_feature_lengths": _to_cpu(result.get("audio_feature_lengths")) if isinstance(result, dict) else None,
        "audio_output_lengths": _to_cpu(result.get("audio_output_lengths")) if isinstance(result, dict) else None,
    }
    return payload


def build_thinker_result(payload: StagePayload, result: Any) -> StagePayload:
    data = _ensure_data(payload)
    engine_outputs = data.setdefault("engine_outputs", {})
    if isinstance(result, ARRequestData):
        engine_outputs["thinker"] = {
            "output_ids": list(result.output_ids),
            "extra_model_outputs": dict(result.extra_model_outputs),
        }
    else:
        engine_outputs["thinker"] = {"result": result}
    return payload

