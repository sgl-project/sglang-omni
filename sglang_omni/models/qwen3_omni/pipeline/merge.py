# SPDX-License-Identifier: Apache-2.0
"""Merge and decode helpers for Qwen3-Omni pipelines."""

from __future__ import annotations

from typing import Any, Iterable

import torch

from sglang_omni.models.qwen3_omni.io import OmniEvent, ThinkerOutput
from sglang_omni.models.qwen3_omni.pipeline.next_stage import AUDIO_STAGE, IMAGE_STAGE
from sglang_omni.proto import StagePayload


def _ensure_data(payload: StagePayload) -> dict[str, Any]:
    if not isinstance(payload.data, dict):
        payload.data = {}
    return payload.data


def _as_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value
    try:
        return torch.as_tensor(value, dtype=dtype)
    except Exception:
        return None


def _as_tensor_list(value: Any) -> list[torch.Tensor] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors = [v for v in value if isinstance(v, torch.Tensor)]
        return tensors or None
    return None


def _non_empty(tensor: torch.Tensor | None) -> bool:
    return isinstance(tensor, torch.Tensor) and tensor.numel() > 0


def merge_for_thinker(payloads: dict[str, StagePayload]) -> StagePayload:
    """Aggregate frontend + encoder outputs into thinker inputs."""
    base = payloads.get("frontend") or next(iter(payloads.values()))
    data = _ensure_data(base)

    encoder_outs: dict[str, Any] = {}
    existing = data.get("encoder_outs")
    if isinstance(existing, dict):
        encoder_outs.update(existing)

    for stage_name, payload in payloads.items():
        stage_data = payload.data if isinstance(payload.data, dict) else {}
        stage_encoder_outs = stage_data.get("encoder_outs")
        if isinstance(stage_encoder_outs, dict) and stage_name in stage_encoder_outs:
            encoder_outs[stage_name] = stage_encoder_outs[stage_name]
            continue
        stage_engine_outputs = stage_data.get("engine_outputs")
        if (
            isinstance(stage_engine_outputs, dict)
            and stage_name in stage_engine_outputs
        ):
            encoder_outs[stage_name] = stage_engine_outputs[stage_name]

    thinker_inputs = build_thinker_inputs(data, encoder_outs)

    data["encoder_outs"] = encoder_outs
    data["thinker_inputs"] = thinker_inputs
    engine_inputs = data.setdefault("engine_inputs", {})
    engine_inputs["thinker"] = thinker_inputs
    data["encoder_inputs"] = {}

    _prune_frontend_for_thinker(data, encoder_outs)
    return base


def build_thinker_inputs(
    frontend_data: dict[str, Any],
    encoder_outs: dict[str, Any],
) -> dict[str, Any]:
    mm_inputs = frontend_data.get("mm_inputs", {})
    mm_image = mm_inputs.get("image", {}) if isinstance(mm_inputs, dict) else {}
    mm_audio = mm_inputs.get("audio", {}) if isinstance(mm_inputs, dict) else {}

    image_out = (
        encoder_outs.get(IMAGE_STAGE, {}) if isinstance(encoder_outs, dict) else {}
    )
    audio_out = (
        encoder_outs.get(AUDIO_STAGE, {}) if isinstance(encoder_outs, dict) else {}
    )

    image_embeds = (
        _as_tensor(image_out.get("image_embeds"))
        if isinstance(image_out, dict)
        else None
    )
    deepstack_visual_embeds = (
        _as_tensor_list(image_out.get("deepstack_visual_embeds"))
        if isinstance(image_out, dict)
        else None
    )
    audio_embeds = (
        _as_tensor(audio_out.get("audio_embeds"))
        if isinstance(audio_out, dict)
        else None
    )

    image_grid_thw = _as_tensor(
        (
            image_out.get("image_grid_thw")
            if isinstance(image_out, dict)
            and image_out.get("image_grid_thw") is not None
            else mm_image.get("image_grid_thw")
        ),
        dtype=torch.long,
    )
    feature_attention_mask = _as_tensor(
        mm_audio.get("feature_attention_mask"),
        dtype=torch.long,
    )
    audio_feature_lengths = _as_tensor(
        (
            audio_out.get("audio_feature_lengths")
            if isinstance(audio_out, dict)
            and audio_out.get("audio_feature_lengths") is not None
            else mm_audio.get("audio_feature_lengths")
        ),
        dtype=torch.long,
    )

    thinker_model_inputs: dict[str, Any] = {}
    if _non_empty(image_embeds):
        thinker_model_inputs["image_embeds"] = image_embeds
    if deepstack_visual_embeds:
        thinker_model_inputs["deepstack_visual_embeds"] = deepstack_visual_embeds
    if _non_empty(audio_embeds):
        thinker_model_inputs["audio_embeds"] = audio_embeds
    if _non_empty(image_grid_thw):
        thinker_model_inputs["image_grid_thw"] = image_grid_thw
    if _non_empty(feature_attention_mask):
        thinker_model_inputs["feature_attention_mask"] = feature_attention_mask
    if _non_empty(audio_feature_lengths):
        thinker_model_inputs["audio_feature_lengths"] = audio_feature_lengths

    return {"model_inputs": thinker_model_inputs}


def _prune_frontend_for_thinker(
    frontend_data: dict[str, Any],
    encoder_outs: dict[str, Any],
) -> None:
    mm_inputs = frontend_data.get("mm_inputs", {})
    mm_image = mm_inputs.get("image", {}) if isinstance(mm_inputs, dict) else {}
    mm_audio = mm_inputs.get("audio", {}) if isinstance(mm_inputs, dict) else {}

    image_out = (
        encoder_outs.get(IMAGE_STAGE, {}) if isinstance(encoder_outs, dict) else {}
    )
    audio_out = (
        encoder_outs.get(AUDIO_STAGE, {}) if isinstance(encoder_outs, dict) else {}
    )

    image_grid_thw = _as_tensor(
        (
            image_out.get("image_grid_thw")
            if isinstance(image_out, dict)
            and image_out.get("image_grid_thw") is not None
            else mm_image.get("image_grid_thw")
        ),
        dtype=torch.long,
    )
    audio_feature_lengths = _as_tensor(
        (
            audio_out.get("audio_feature_lengths")
            if isinstance(audio_out, dict)
            and audio_out.get("audio_feature_lengths") is not None
            else mm_audio.get("audio_feature_lengths")
        ),
        dtype=torch.long,
    )

    frontend_data["mm_inputs"] = {
        "image": {"image_grid_thw": image_grid_thw},
        "audio": {"audio_feature_lengths": audio_feature_lengths},
    }


def decode_events(
    *,
    thinker_out: ThinkerOutput,
    data: dict[str, Any],
    tokenizer: Any,
    eos_token_id: int | None,
    step: int,
) -> Iterable[OmniEvent]:
    output_ids = thinker_out.get("output_ids", [])
    if not isinstance(output_ids, list) or not output_ids:
        return []

    stream_state = data.setdefault("stream_state", {"token_ids": [], "text": ""})
    token_ids = stream_state.setdefault("token_ids", [])
    prev_text = str(stream_state.setdefault("text", ""))

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
    if decoded.startswith(prev_text):
        delta = decoded[len(prev_text) :]
    else:
        delta = decoded
    stream_state["text"] = decoded
    return [
        OmniEvent(
            type="text_delta", modality="text", payload={"text": delta}, is_final=False
        )
    ]
