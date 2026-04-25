# SPDX-License-Identifier: Apache-2.0
"""Route-specific payload trimming for Qwen3-Omni stages."""

from __future__ import annotations

from typing import Any

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
    del request_id
    state = PipelineState.from_dict(payload.data)
    if next_stage in (IMAGE_STAGE, AUDIO_STAGE):
        return _with_state(payload, _encoder_input_state(state, next_stage))
    if next_stage == AGGREGATE_STAGE:
        return _with_state(payload, _aggregate_state(state))
    raise ValueError(f"Unexpected Qwen3-Omni preprocessing target: {next_stage}")


def encoder_payload_filter(
    request_id: str, next_stage: str, payload: StagePayload
) -> StagePayload:
    """Drop consumed encoder inputs before routing encoder outputs onward."""
    del request_id
    if next_stage != AGGREGATE_STAGE:
        raise ValueError(f"Unexpected Qwen3-Omni encoder target: {next_stage}")

    state = PipelineState.from_dict(payload.data)
    return _with_state(payload, PipelineState(encoder_outs=state.encoder_outs))


def _with_state(payload: StagePayload, state: PipelineState) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def _encoder_input_state(state: PipelineState, stage_name: str) -> PipelineState:
    inputs = state.encoder_inputs.get(stage_name)
    assert isinstance(inputs, dict), f"missing encoder inputs for {stage_name}"
    return PipelineState(
        encoder_inputs={stage_name: inputs},
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
    result: dict[str, Any] = {}
    image = _copy_keys(mm_inputs.get("image", {}), ("image_grid_thw",))
    audio = _copy_keys(
        mm_inputs.get("audio", {}), ("feature_attention_mask", "audio_feature_lengths")
    )
    video = _copy_keys(
        mm_inputs.get("video", {}),
        ("video_grid_thw", "video_second_per_grid", "use_audio_in_video"),
    )
    if image:
        result["image"] = image
    if audio:
        result["audio"] = audio
    if video:
        result["video"] = video
    return result


def _encoder_cache_keys(
    encoder_inputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for stage_name in (IMAGE_STAGE, AUDIO_STAGE):
        inputs = encoder_inputs.get(stage_name)
        if inputs is None:
            continue
        assert isinstance(inputs, dict), f"invalid encoder inputs for {stage_name}"
        if inputs.get("_skip"):
            result[stage_name] = {"_skip": True, "_result": inputs.get("_result", {})}
            continue
        cache_key = inputs.get("cache_key")
        if cache_key is not None:
            result[stage_name] = {"cache_key": cache_key}
    return result


def _copy_keys(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    assert isinstance(source, dict)
    return {key: source[key] for key in keys if key in source}
