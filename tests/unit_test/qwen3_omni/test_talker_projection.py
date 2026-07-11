# SPDX-License-Identifier: Apache-2.0
"""project_mm_aggregate_to_talker_ar drops the deepstack visual embeds the talker
never reads, whether they arrive as tensors or (on the #932 fast path) as refs. #934
"""
from __future__ import annotations

import torch

from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    project_mm_aggregate_to_talker_ar,
)
from sglang_omni.pipeline.tensor_ref import TensorRef
from tests.unit_test.fixtures.pipeline_fakes import make_stage_payload


def _ref(path: str) -> dict:
    return TensorRef(
        ref_id="req-1:ref",
        request_id="req-1",
        producer_stage="image_encoder",
        consumer_stage="thinker",
        path=path,
        shape=(4, 2),
        dtype="torch.float32",
        nbytes=32,
        blob_key="req-1:blob",
        blob_metadata={},
    ).to_dict()


def _talker_model_inputs(model_inputs: dict) -> dict:
    state = Qwen3OmniPipelineState(
        prompt={"input_ids": torch.zeros(3, dtype=torch.long)},
        thinker_inputs={"model_inputs": model_inputs},
    )
    projected = project_mm_aggregate_to_talker_ar(
        make_stage_payload(data=state.to_dict(), request_id="req-1")
    )
    return Qwen3OmniPipelineState.from_dict(projected.data).thinker_inputs[
        "model_inputs"
    ]


def test_talker_projection_drops_deepstack_keeps_used_embeds() -> None:
    out = _talker_model_inputs(
        {
            "video_embeds": torch.zeros(4, 2),
            "image_embeds": torch.zeros(4, 2),
            "audio_embeds": torch.zeros(4, 2),
            "image_deepstack_visual_embeds": torch.zeros(4, 2),
            "video_deepstack_visual_embeds": torch.zeros(4, 2),
            "deepstack_visual_embeds": torch.zeros(4, 2),
            "video_grid_thw": torch.ones(1, 3, dtype=torch.long),
        }
    )
    for dropped in (
        "image_deepstack_visual_embeds",
        "video_deepstack_visual_embeds",
        "deepstack_visual_embeds",
    ):
        assert dropped not in out
    for kept in ("video_embeds", "image_embeds", "audio_embeds", "video_grid_thw"):
        assert kept in out


def test_talker_projection_drops_deepstack_ref() -> None:
    out = _talker_model_inputs(
        {
            "video_embeds": torch.zeros(4, 2),
            "video_deepstack_visual_embeds": _ref("deepstack_visual_embeds_video"),
        }
    )
    assert "video_deepstack_visual_embeds" not in out
    assert "video_embeds" in out
