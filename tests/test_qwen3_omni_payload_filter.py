# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen3-Omni route-specific payload trimming."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.qwen3_omni.io import PipelineState
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    AGGREGATE_STAGE,
    IMAGE_STAGE,
)
from sglang_omni.models.qwen3_omni.pipeline.payload_filter import (
    encoder_payload_filter,
    preprocessing_payload_filter,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _payload(state: PipelineState) -> StagePayload:
    return StagePayload(
        request_id="req",
        request=OmniRequest(inputs={}, params={}),
        data=state.to_dict(),
    )


def test_preprocessing_filter_strips_raw_video_from_aggregate_payload() -> None:
    raw_video = torch.zeros((8, 3, 16, 16), dtype=torch.float32)
    state = PipelineState(
        prompt={"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1])},
        mm_inputs={
            "video": {
                "pixel_values_videos": raw_video,
                "video_grid_thw": torch.tensor([[1, 2, 4]]),
                "video_second_per_grid": torch.tensor([0.5]),
            }
        },
        encoder_inputs={
            IMAGE_STAGE: {
                "pixel_values_videos": raw_video,
                "cache_key": "video-key",
            }
        },
    )

    filtered = preprocessing_payload_filter("req", AGGREGATE_STAGE, _payload(state))
    filtered_state = PipelineState.from_dict(filtered.data)

    assert "pixel_values_videos" not in filtered_state.mm_inputs["video"]
    assert filtered_state.mm_inputs["video"]["video_grid_thw"].shape == (1, 3)
    assert filtered_state.encoder_inputs[IMAGE_STAGE] == {"cache_key": "video-key"}


def test_encoder_filter_sends_only_encoder_outputs() -> None:
    encoder_out = torch.ones((2, 4), dtype=torch.float32)
    state = PipelineState(
        encoder_inputs={IMAGE_STAGE: {"pixel_values_videos": torch.zeros((2, 4))}},
        encoder_outs={IMAGE_STAGE: {"video_embeds": encoder_out}},
        engine_outputs={IMAGE_STAGE: {"video_embeds": encoder_out}},
    )

    filtered = encoder_payload_filter("req", AGGREGATE_STAGE, _payload(state))
    filtered_state = PipelineState.from_dict(filtered.data)

    assert filtered_state.encoder_inputs == {}
    assert filtered_state.engine_outputs == {}
    filtered_video_embeds = filtered_state.encoder_outs[IMAGE_STAGE]["video_embeds"]
    assert torch.equal(filtered_video_embeds, encoder_out)


def test_preprocessing_filter_rejects_unknown_route() -> None:
    with pytest.raises(ValueError, match="Unexpected Qwen3-Omni preprocessing target"):
        preprocessing_payload_filter("req", "unknown", _payload(PipelineState()))
