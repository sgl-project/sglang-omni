# SPDX-License-Identifier: Apache-2.0
"""Merge and decode helpers for Qwen3-Omni pipelines."""

from __future__ import annotations

from typing import Any, Iterable

import torch

from sglang_omni_v1.models.qwen3_omni.payload_types import (
    OmniEvent,
    PipelineState,
    ThinkerOutput,
)
from sglang_omni_v1.proto import StagePayload

IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"


def _cast_tensor(
    value: torch.Tensor | None, dtype: torch.dtype | None = None
) -> torch.Tensor | None:
    if value is None:
        return None
    return value.to(dtype=dtype) if dtype is not None else value


def _non_empty(tensor: torch.Tensor | None) -> bool:
    return tensor is not None and tensor.numel() > 0


def merge_for_thinker(payloads: dict[str, StagePayload]) -> StagePayload:
    """Aggregate preprocessing + encoder outputs into thinker inputs."""
    base = payloads.get("preprocessing") or next(iter(payloads.values()))
    state = PipelineState.from_dict(base.data)
    encoder_outs: dict[str, Any] = {}
    if state.encoder_outs:
        encoder_outs.update(state.encoder_outs)

    for stage_name, payload in payloads.items():
        stage_state = PipelineState.from_dict(payload.data)
        if stage_name in stage_state.encoder_outs:
            encoder_outs[stage_name] = stage_state.encoder_outs[stage_name]
            continue
        if stage_name in stage_state.engine_outputs:
            encoder_outs[stage_name] = stage_state.engine_outputs[stage_name]

    thinker_inputs = build_thinker_inputs(state, encoder_outs)

    state.thinker_inputs = thinker_inputs
    state.encoder_inputs = {}
    _prune_preprocessing_for_thinker(state, encoder_outs)
    # Encoder outputs have been consumed into thinker_inputs; keeping both
    # doubles multimodal tensor payloads sent to the thinker.
    state.encoder_outs = {}
    base.data = state.to_dict()
    return base


def build_thinker_inputs(
    state: PipelineState,
    encoder_outs: dict[str, Any],
) -> dict[str, Any]:
    mm_inputs = state.mm_inputs
    mm_image = mm_inputs.get("image", {})
    mm_audio = mm_inputs.get("audio", {})
    mm_video = mm_inputs.get("video", {})

    image_out = encoder_outs.get(IMAGE_STAGE, {})
    audio_out = encoder_outs.get(AUDIO_STAGE, {})
    video_out = image_out

    image_embeds = image_out.get("image_embeds")
    image_deepstack_visual_embeds = image_out.get("deepstack_visual_embeds_image")
    video_deepstack_visual_embeds = video_out.get("deepstack_visual_embeds_video")
    audio_embeds = audio_out.get("audio_embeds")
    video_embeds = video_out.get("video_embeds")

    image_grid_thw = _cast_tensor(
        (
            image_out.get("image_grid_thw")
            if image_out.get("image_grid_thw") is not None
            else mm_image.get("image_grid_thw")
        ),
        dtype=torch.long,
    )
    video_grid_thw = _cast_tensor(
        (
            video_out.get("video_grid_thw")
            if video_out.get("video_grid_thw") is not None
            else mm_video.get("video_grid_thw")
        ),
        dtype=torch.long,
    )
    feature_attention_mask = _cast_tensor(
        mm_audio.get("feature_attention_mask"),
        dtype=torch.long,
    )
    audio_feature_lengths = _cast_tensor(
        (
            audio_out.get("audio_feature_lengths")
            if audio_out.get("audio_feature_lengths") is not None
            else mm_audio.get("audio_feature_lengths")
        ),
        dtype=torch.long,
    )
    video_second_per_grid = _cast_tensor(
        mm_video.get("video_second_per_grid"),
        dtype=torch.float,
    )

    thinker_model_inputs: dict[str, Any] = {}
    has_image = _non_empty(image_embeds)
    has_video = _non_empty(video_embeds)
    if has_image:
        thinker_model_inputs["image_embeds"] = image_embeds
    if has_video:
        thinker_model_inputs["video_embeds"] = video_embeds
    if (
        has_image
        and image_deepstack_visual_embeds
        and has_video
        and video_deepstack_visual_embeds
    ):
        thinker_model_inputs["image_deepstack_visual_embeds"] = (
            image_deepstack_visual_embeds
        )
        thinker_model_inputs["video_deepstack_visual_embeds"] = (
            video_deepstack_visual_embeds
        )
    elif has_image and image_deepstack_visual_embeds:
        thinker_model_inputs["deepstack_visual_embeds"] = image_deepstack_visual_embeds
    elif has_video and video_deepstack_visual_embeds:
        thinker_model_inputs["deepstack_visual_embeds"] = video_deepstack_visual_embeds
    if _non_empty(audio_embeds):
        thinker_model_inputs["audio_embeds"] = audio_embeds
    if _non_empty(image_grid_thw):
        thinker_model_inputs["image_grid_thw"] = image_grid_thw
    if _non_empty(video_grid_thw):
        thinker_model_inputs["video_grid_thw"] = video_grid_thw
    if _non_empty(feature_attention_mask):
        thinker_model_inputs["feature_attention_mask"] = feature_attention_mask
    if _non_empty(audio_feature_lengths):
        thinker_model_inputs["audio_feature_lengths"] = audio_feature_lengths
    if _non_empty(video_second_per_grid):
        thinker_model_inputs["video_second_per_grid"] = video_second_per_grid
    if mm_video.get("use_audio_in_video") is True:
        thinker_model_inputs["use_audio_in_video"] = True

    media_cache_keys: dict[str, str] = {}
    encoder_inputs = state.encoder_inputs or {}
    image_ck = (encoder_inputs.get("image_encoder") or {}).get("cache_key")
    audio_ck = (encoder_inputs.get("audio_encoder") or {}).get("cache_key")
    if image_ck:
        media_cache_keys["image"] = f"image:{image_ck}"
        # Note (Xuesong): Image and video share the same encoder cache key, so prefix them
        # differently to avoid hashed pad-value collisions across modalities.
        media_cache_keys["video"] = f"video:{image_ck}"
    if audio_ck:
        media_cache_keys["audio"] = f"audio:{audio_ck}"

    result: dict[str, Any] = {"model_inputs": thinker_model_inputs}
    if media_cache_keys:
        result["media_cache_keys"] = media_cache_keys
    return result


def _prune_preprocessing_for_thinker(
    state: PipelineState,
    encoder_outs: dict[str, Any],
) -> None:
    mm_inputs = state.mm_inputs
    mm_image = mm_inputs.get("image", {})
    mm_audio = mm_inputs.get("audio", {})
    mm_video = mm_inputs.get("video", {})

    image_out = encoder_outs.get(IMAGE_STAGE, {})
    audio_out = encoder_outs.get(AUDIO_STAGE, {})
    video_out = image_out

    image_grid_thw = _cast_tensor(
        (
            image_out.get("image_grid_thw")
            if image_out.get("image_grid_thw") is not None
            else mm_image.get("image_grid_thw")
        ),
        dtype=torch.long,
    )
    audio_feature_lengths = _cast_tensor(
        (
            audio_out.get("audio_feature_lengths")
            if audio_out.get("audio_feature_lengths") is not None
            else mm_audio.get("audio_feature_lengths")
        ),
        dtype=torch.long,
    )
    video_grid_thw = _cast_tensor(
        (
            video_out.get("video_grid_thw")
            if video_out.get("video_grid_thw") is not None
            else mm_video.get("video_grid_thw")
        ),
        dtype=torch.long,
    )
    video_second_per_grid = _cast_tensor(
        mm_video.get("video_second_per_grid"),
        dtype=torch.float,
    )
    use_audio_in_video = mm_video.get("use_audio_in_video")

    state.mm_inputs = {
        "image": {"image_grid_thw": image_grid_thw},
        "audio": {"audio_feature_lengths": audio_feature_lengths},
        "video": {
            "video_grid_thw": video_grid_thw,
            "video_second_per_grid": video_second_per_grid,
            "use_audio_in_video": use_audio_in_video,
        },
    }


def decode_events(
    *,
    thinker_out: ThinkerOutput,
    state: PipelineState,
    tokenizer: Any,
    eos_token_id: int | None,
    step: int,
) -> Iterable[OmniEvent]:
    output_ids = thinker_out.get("output_ids", [])
    if not output_ids:
        return []

    stream_state = state.stream_state
    if not stream_state:
        stream_state.update({"token_ids": [], "text": "", "emitted_text": ""})
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

    # Skip incomplete multi-byte characters (replacement char).
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
