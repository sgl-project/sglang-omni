# SPDX-License-Identifier: Apache-2.0
"""Fan-in merge logic for Qwen3-Omni."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.proto import StagePayload


def _as_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value
    try:
        return torch.as_tensor(value, dtype=dtype)
    except Exception:
        return None


def _merge_engine_outputs(payloads: dict[str, StagePayload]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads.values():
        if not isinstance(payload.data, dict):
            continue
        outputs = payload.data.get("engine_outputs")
        if isinstance(outputs, dict):
            merged.update(outputs)
    return merged


def merge_to_thinker(payloads: dict[str, StagePayload]) -> StagePayload:
    """Merge frontend + encoder branches into thinker inputs."""
    base = payloads.get("frontend") or next(iter(payloads.values()))
    if not isinstance(base.data, dict):
        raise TypeError("Expected dict payload data for merge_to_thinker")

    data = base.data
    data["engine_outputs"] = _merge_engine_outputs(payloads)

    mm_inputs = data.get("mm_inputs", {})
    engine_outputs = data.get("engine_outputs", {})

    image_out = engine_outputs.get("image_encoder", {}) if isinstance(engine_outputs, dict) else {}
    audio_out = engine_outputs.get("audio_encoder", {}) if isinstance(engine_outputs, dict) else {}

    image_embeds = _as_tensor(image_out.get("image_embeds"))
    audio_embeds = _as_tensor(audio_out.get("audio_embeds"))

    image_grid_thw = _as_tensor(
        image_out.get("image_grid_thw")
        if isinstance(image_out, dict) and image_out.get("image_grid_thw") is not None
        else (mm_inputs.get("image", {}) if isinstance(mm_inputs, dict) else {}).get("image_grid_thw"),
        dtype=torch.long,
    )
    audio_feature_lengths = _as_tensor(
        audio_out.get("audio_feature_lengths")
        if isinstance(audio_out, dict) and audio_out.get("audio_feature_lengths") is not None
        else (mm_inputs.get("audio", {}) if isinstance(mm_inputs, dict) else {}).get("audio_feature_lengths"),
        dtype=torch.long,
    )

    engine_inputs = data.setdefault("engine_inputs", {})
    thinker_inputs = engine_inputs.setdefault("thinker", {})
    thinker_model_inputs = thinker_inputs.setdefault("model_inputs", {})

    if image_embeds is not None and image_embeds.numel() > 0:
        thinker_model_inputs["image_embeds"] = image_embeds
    if audio_embeds is not None and audio_embeds.numel() > 0:
        thinker_model_inputs["audio_embeds"] = audio_embeds
    if image_grid_thw is not None and image_grid_thw.numel() > 0:
        thinker_model_inputs["image_grid_thw"] = image_grid_thw
    if audio_feature_lengths is not None and audio_feature_lengths.numel() > 0:
        thinker_model_inputs["audio_feature_lengths"] = audio_feature_lengths

    # Drop large raw modality tensors before sending to the thinker stage.
    data["mm_inputs"] = {
        "image": {"image_grid_thw": image_grid_thw},
        "audio": {"audio_feature_lengths": audio_feature_lengths},
    }

    return base
