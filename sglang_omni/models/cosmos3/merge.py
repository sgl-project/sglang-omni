# SPDX-License-Identifier: Apache-2.0
"""Merge standalone Cosmos3 encoder outputs at the thinker boundary."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.cosmos3.payload_types import Cosmos3PipelineState
from sglang_omni.models.cosmos3.request_builders import VISION_STAGE
from sglang_omni.proto import StagePayload


def _non_empty(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.numel() > 0


def build_thinker_inputs(
    state: Cosmos3PipelineState,
    vision_out: dict[str, Any],
) -> dict[str, Any]:
    model_inputs: dict[str, Any] = {}
    for key in ("image_embeds", "video_embeds"):
        value = vision_out.get(key)
        if _non_empty(value):
            model_inputs[key] = value

    image_deepstack = vision_out.get("deepstack_visual_embeds_image")
    video_deepstack = vision_out.get("deepstack_visual_embeds_video")
    if image_deepstack and video_deepstack:
        model_inputs["image_deepstack_visual_embeds"] = image_deepstack
        model_inputs["video_deepstack_visual_embeds"] = video_deepstack
    elif image_deepstack:
        model_inputs["deepstack_visual_embeds"] = image_deepstack
    elif video_deepstack:
        model_inputs["deepstack_visual_embeds"] = video_deepstack

    for key in ("image_grid_thw", "video_grid_thw"):
        value = vision_out.get(key)
        if not _non_empty(value):
            value = state.mm_inputs.get(key)
        if _non_empty(value):
            model_inputs[key] = value.to(dtype=torch.long)

    result: dict[str, Any] = {"model_inputs": model_inputs}
    cache_key = (state.encoder_inputs.get(VISION_STAGE) or {}).get("cache_key")
    if cache_key is not None:
        result["media_cache_key"] = str(cache_key)
    return result


def merge_for_thinker(payloads: dict[str, StagePayload]) -> StagePayload:
    base = payloads.get("preprocessing") or next(iter(payloads.values()))
    state = Cosmos3PipelineState.from_dict(base.data)
    vision_out = dict(state.encoder_outs.get(VISION_STAGE, {}))

    vision_payload = payloads.get(VISION_STAGE)
    if vision_payload is not None:
        vision_state = Cosmos3PipelineState.from_dict(vision_payload.data)
        value = vision_state.encoder_outs.get(VISION_STAGE)
        if isinstance(value, dict):
            vision_out = value

    state.thinker_inputs = build_thinker_inputs(state, vision_out)
    state.encoder_inputs = {}
    state.encoder_outs = {}
    base.data = state.to_dict()
    return base


__all__ = ["build_thinker_inputs", "merge_for_thinker"]
