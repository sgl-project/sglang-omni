# SPDX-License-Identifier: Apache-2.0
"""Merge and decode helpers for Qwen3-Omni pipelines."""

from __future__ import annotations

from typing import Any, Iterable

import torch

from sglang_omni.models.omni_base.merge import (
    as_tensor,
    as_tensor_list,
    create_standard_merge_fn,
    is_non_empty,
)
from sglang_omni.models.omni_base.types import OmniEvent, ThinkerOutput
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    AUDIO_STAGE,
    IMAGE_STAGE,
    PREPROCESSING_STAGE,
)
from sglang_omni.proto import StagePayload


def build_thinker_inputs(
    payload: StagePayload,
    encoder_outs: dict[str, Any],
) -> dict[str, Any]:
    mm_inputs = payload.mm_inputs or {}
    mm_image = mm_inputs.get("image", {}) if isinstance(mm_inputs, dict) else {}
    mm_audio = mm_inputs.get("audio", {}) if isinstance(mm_inputs, dict) else {}
    mm_video = mm_inputs.get("video", {}) if isinstance(mm_inputs, dict) else {}

    image_out = (
        encoder_outs.get(IMAGE_STAGE, {}) if isinstance(encoder_outs, dict) else {}
    )
    audio_out = (
        encoder_outs.get(AUDIO_STAGE, {}) if isinstance(encoder_outs, dict) else {}
    )
    video_out = image_out

    image_embeds = (
        as_tensor(image_out.get("image_embeds"))
        if isinstance(image_out, dict)
        else None
    )
    image_deepstack_visual_embeds = (
        as_tensor_list(image_out.get("deepstack_visual_embeds_image"))
        if isinstance(image_out, dict)
        else None
    )
    video_deepstack_visual_embeds = (
        as_tensor_list(video_out.get("deepstack_visual_embeds_video"))
        if isinstance(video_out, dict)
        else None
    )
    audio_embeds = (
        as_tensor(audio_out.get("audio_embeds"))
        if isinstance(audio_out, dict)
        else None
    )
    video_embeds = (
        as_tensor(video_out.get("video_embeds"))
        if isinstance(video_out, dict)
        else None
    )

    image_grid_thw = as_tensor(
        (
            image_out.get("image_grid_thw")
            if isinstance(image_out, dict)
            and image_out.get("image_grid_thw") is not None
            else mm_image.get("image_grid_thw")
        ),
        dtype=torch.long,
    )
    video_grid_thw = as_tensor(
        (
            video_out.get("video_grid_thw")
            if isinstance(video_out, dict)
            and video_out.get("video_grid_thw") is not None
            else mm_video.get("video_grid_thw")
        ),
        dtype=torch.long,
    )
    feature_attention_mask = as_tensor(
        mm_audio.get("feature_attention_mask"),
        dtype=torch.long,
    )
    audio_feature_lengths = as_tensor(
        (
            audio_out.get("audio_feature_lengths")
            if isinstance(audio_out, dict)
            and audio_out.get("audio_feature_lengths") is not None
            else mm_audio.get("audio_feature_lengths")
        ),
        dtype=torch.long,
    )
    video_second_per_grid = as_tensor(
        mm_video.get("video_second_per_grid"),
        dtype=torch.float,
    )

    thinker_model_inputs: dict[str, Any] = {}
    has_image = is_non_empty(image_embeds)
    has_video = is_non_empty(video_embeds)
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
    if is_non_empty(audio_embeds):
        thinker_model_inputs["audio_embeds"] = audio_embeds
    if is_non_empty(image_grid_thw):
        thinker_model_inputs["image_grid_thw"] = image_grid_thw
    if is_non_empty(video_grid_thw):
        thinker_model_inputs["video_grid_thw"] = video_grid_thw
    if is_non_empty(feature_attention_mask):
        thinker_model_inputs["feature_attention_mask"] = feature_attention_mask
    if is_non_empty(audio_feature_lengths):
        thinker_model_inputs["audio_feature_lengths"] = audio_feature_lengths
    if is_non_empty(video_second_per_grid):
        thinker_model_inputs["video_second_per_grid"] = video_second_per_grid
    if mm_video.get("use_audio_in_video") is True:
        thinker_model_inputs["use_audio_in_video"] = True

    return {"model_inputs": thinker_model_inputs}


def _prune_preprocessing_for_thinker(
    payload: StagePayload,
    encoder_outs: dict[str, Any],
) -> None:
    mm_inputs = payload.mm_inputs or {}
    mm_image = mm_inputs.get("image", {}) if isinstance(mm_inputs, dict) else {}
    mm_audio = mm_inputs.get("audio", {}) if isinstance(mm_inputs, dict) else {}
    mm_video = mm_inputs.get("video", {}) if isinstance(mm_inputs, dict) else {}

    image_out = (
        encoder_outs.get(IMAGE_STAGE, {}) if isinstance(encoder_outs, dict) else {}
    )
    audio_out = (
        encoder_outs.get(AUDIO_STAGE, {}) if isinstance(encoder_outs, dict) else {}
    )
    video_out = image_out

    image_grid_thw = as_tensor(
        (
            image_out.get("image_grid_thw")
            if isinstance(image_out, dict)
            and image_out.get("image_grid_thw") is not None
            else mm_image.get("image_grid_thw")
        ),
        dtype=torch.long,
    )
    audio_feature_lengths = as_tensor(
        (
            audio_out.get("audio_feature_lengths")
            if isinstance(audio_out, dict)
            and audio_out.get("audio_feature_lengths") is not None
            else mm_audio.get("audio_feature_lengths")
        ),
        dtype=torch.long,
    )
    video_grid_thw = as_tensor(
        (
            video_out.get("video_grid_thw")
            if isinstance(video_out, dict)
            and video_out.get("video_grid_thw") is not None
            else mm_video.get("video_grid_thw")
        ),
        dtype=torch.long,
    )
    video_second_per_grid = as_tensor(
        mm_video.get("video_second_per_grid"),
        dtype=torch.float,
    )
    use_audio_in_video = mm_video.get("use_audio_in_video")

    payload.mm_inputs = {
        "image": {"image_grid_thw": image_grid_thw},
        "audio": {"audio_feature_lengths": audio_feature_lengths},
        "video": {
            "video_grid_thw": video_grid_thw,
            "video_second_per_grid": video_second_per_grid,
            "use_audio_in_video": use_audio_in_video,
        },
    }


# Create the standard merge function for Qwen3-Omni
merge_for_thinker = create_standard_merge_fn(
    source_stages=[PREPROCESSING_STAGE, IMAGE_STAGE, AUDIO_STAGE],
    build_thinker_inputs_fn=build_thinker_inputs,
    prune_fn=_prune_preprocessing_for_thinker,
)


def decode_events(
    *,
    thinker_out: ThinkerOutput,
    payload: StagePayload,
    tokenizer: Any,
    eos_token_id: int | None,
    step: int,
) -> Iterable[OmniEvent]:
    output_ids = thinker_out.get("output_ids", [])
    if not isinstance(output_ids, list) or not output_ids:
        return []

    # Lazy initialization for stream_state
    if payload.stream_state is None:
        payload.stream_state = {"token_ids": [], "text": ""}
    stream_state = payload.stream_state

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
