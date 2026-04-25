# SPDX-License-Identifier: Apache-2.0
"""Route-specific payload trimming for Qwen3-Omni stages."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.qwen3_omni.io import PipelineState
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    AGGREGATE_STAGE,
    AUDIO_STAGE,
    IMAGE_STAGE,
)
from sglang_omni.proto import StagePayload


def preprocessing_payload_filter(
    request_id: str, next_stage: str, payload: StagePayload
) -> StagePayload:
    """Send only the tensors required by each preprocessing fan-out target."""
    if payload.request_id != request_id:
        raise ValueError(
            "Payload request_id mismatch "
            f"(expected={request_id} got={payload.request_id})"
        )

    state = PipelineState.from_dict(payload.data)
    if next_stage in (IMAGE_STAGE, AUDIO_STAGE):
        return _with_state(payload, _encoder_state(state, next_stage))
    if next_stage == AGGREGATE_STAGE:
        return _with_state(payload, _aggregate_state(state))
    return payload


def encoder_payload_filter(
    request_id: str, next_stage: str, payload: StagePayload
) -> StagePayload:
    """Drop consumed encoder inputs before routing encoder outputs onward."""
    if payload.request_id != request_id:
        raise ValueError(
            "Payload request_id mismatch "
            f"(expected={request_id} got={payload.request_id})"
        )
    if next_stage != AGGREGATE_STAGE:
        return payload

    state = PipelineState.from_dict(payload.data)
    return _with_state(
        payload,
        PipelineState(encoder_outs=_to_cpu_detached(state.encoder_outs)),
    )


def _with_state(payload: StagePayload, state: PipelineState) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def _encoder_state(state: PipelineState, stage_name: str) -> PipelineState:
    inputs = state.encoder_inputs.get(stage_name)
    return PipelineState(
        encoder_inputs={stage_name: inputs if isinstance(inputs, dict) else {}},
        stream_state=dict(state.stream_state),
    )


def _aggregate_state(state: PipelineState) -> PipelineState:
    return PipelineState(
        prompt=state.prompt,
        mm_inputs=_lightweight_mm_inputs(state.mm_inputs),
        encoder_inputs=_encoder_cache_keys(state.encoder_inputs),
        stream_state=dict(state.stream_state),
    )


def _lightweight_mm_inputs(mm_inputs: dict[str, Any]) -> dict[str, Any]:
    image = mm_inputs.get("image", {}) if isinstance(mm_inputs, dict) else {}
    audio = mm_inputs.get("audio", {}) if isinstance(mm_inputs, dict) else {}
    video = mm_inputs.get("video", {}) if isinstance(mm_inputs, dict) else {}

    return {
        "image": _copy_keys(image, ("image_grid_thw",)),
        "audio": _copy_keys(audio, ("feature_attention_mask", "audio_feature_lengths")),
        "video": _copy_keys(
            video,
            ("video_grid_thw", "video_second_per_grid", "use_audio_in_video"),
        ),
    }


def _encoder_cache_keys(
    encoder_inputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for stage_name in (IMAGE_STAGE, AUDIO_STAGE):
        inputs = encoder_inputs.get(stage_name)
        if not isinstance(inputs, dict):
            continue
        if inputs.get("_skip"):
            result[stage_name] = {"_skip": True, "_result": inputs.get("_result", {})}
            continue
        cache_key = inputs.get("cache_key")
        if cache_key is not None:
            result[stage_name] = {"cache_key": cache_key}
    return result


def _copy_keys(source: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    return {key: source[key] for key in keys if key in source}


def _to_cpu_detached(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {key: _to_cpu_detached(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_detached(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_detached(item) for item in value)
    return value
