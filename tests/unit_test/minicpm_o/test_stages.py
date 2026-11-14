# SPDX-License-Identifier: Apache-2.0
"""Verify the image encoder stage accepts a preprocessing-supplied cache hit."""

from __future__ import annotations

import torch

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import build_encoder_request
from sglang_omni.models.minicpm_o.stages import create_encoder_executor
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest

STAGE_NAME = "cached_image_encoder"


class CountingEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.forward_calls: int = 0

    def forward(self, **model_inputs: object) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        return {"image_embeds": torch.zeros(2, 3)}


def build_payload(stage_inputs: dict[str, object]) -> StagePayload:
    return StagePayload(
        request_id="cached-image",
        request=OmniRequest(inputs=[]),
        data=MiniCPMOPipelineState(encoder_inputs={STAGE_NAME: stage_inputs}).to_dict(),
    )


def test_cached_result_skips_the_encoder_forward() -> None:
    encoder = CountingEncoder()
    executor = create_encoder_executor(encoder, stage_name=STAGE_NAME)
    cached_result = {"image_embeds": torch.ones(2, 3)}

    output = executor.fn(
        build_payload(
            {"_active": True, "cache_key": "image-key", "cached_result": cached_result}
        )
    )

    assert encoder.forward_calls == 0
    state = MiniCPMOPipelineState.from_dict(output.data)
    torch.testing.assert_close(
        state.encoder_outs[STAGE_NAME]["image_embeds"],
        torch.ones(2, 3),
        rtol=0,
        atol=0,
    )


def test_build_encoder_request_keeps_cached_result_out_of_model_inputs() -> None:
    state = MiniCPMOPipelineState(
        encoder_inputs={
            STAGE_NAME: {
                "_active": True,
                "cache_key": "image-key",
                "cached_result": {"image_embeds": torch.zeros(2, 3)},
            }
        }
    )

    request = build_encoder_request(state, stage_name=STAGE_NAME)

    assert request.model_inputs == {}
    assert request.skip_result is not None
