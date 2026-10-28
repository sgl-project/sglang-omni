# SPDX-License-Identifier: Apache-2.0
"""The talker edge ships only the encoder metadata its MRoPE positions read."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import project_encoder_to_talker_ar
from tests.unit_test.fixtures.pipeline_fakes import make_stage_payload


@pytest.mark.parametrize(
    ("stage_name", "encoder_out", "kept"),
    [
        (
            "image_encoder",
            {
                "image_embeds": torch.zeros(4, 2),
                "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
                "image_token_counts": torch.tensor([4]),
                "deepstack_visual_embeds_image": [torch.zeros(4, 2)],
                "video_embeds": torch.zeros(4, 2),
                "video_grid_thw": torch.ones(1, 3, dtype=torch.long),
                "video_token_counts": torch.tensor([4]),
                "deepstack_visual_embeds_video": [torch.zeros(4, 2)],
            },
            {"image_grid_thw", "video_grid_thw"},
        ),
        (
            "audio_encoder",
            {
                "audio_embeds": torch.zeros(4, 2),
                "audio_feature_lengths": torch.tensor([400]),
                "audio_output_lengths": torch.tensor([4]),
            },
            {"audio_feature_lengths"},
        ),
    ],
)
def test_talker_edge_drops_encoder_features(
    stage_name: str,
    encoder_out: dict[str, torch.Tensor | list[torch.Tensor]],
    kept: set[str],
) -> None:
    state = Qwen3OmniPipelineState(encoder_outs={stage_name: encoder_out})
    projected = project_encoder_to_talker_ar(
        make_stage_payload(data=state.to_dict(), request_id="req-1")
    )
    out = Qwen3OmniPipelineState.from_dict(projected.data).encoder_outs[stage_name]
    assert set(out) == kept
    for key in kept:
        assert torch.equal(out[key], encoder_out[key])
