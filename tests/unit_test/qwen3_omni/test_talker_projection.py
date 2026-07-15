# SPDX-License-Identifier: Apache-2.0
"""The talker projection drops deepstack visual embeds it never reads."""
from __future__ import annotations

import torch

from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    project_mm_aggregate_to_talker_ar,
)
from tests.unit_test.fixtures.pipeline_fakes import make_stage_payload


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
