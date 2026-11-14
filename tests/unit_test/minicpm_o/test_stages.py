# SPDX-License-Identifier: Apache-2.0
"""Verify the image encoder stage accepts a preprocessing-supplied cache hit."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.minicpm_o import stages as stages_mod
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import build_encoder_request
from sglang_omni.models.minicpm_o.routing import (
    IMAGE_STAGE,
    project_preprocessing_to_encoder,
    resolve_preprocessing_next_stages,
    resolve_thinker_wait_sources,
)
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


def build_payload(
    stage_inputs: dict[str, object], stage_name: str = STAGE_NAME
) -> StagePayload:
    return StagePayload(
        request_id="cached-image",
        request=OmniRequest(inputs=[]),
        data=MiniCPMOPipelineState(encoder_inputs={stage_name: stage_inputs}).to_dict(),
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


def test_preprocessing_executor_uses_the_encoder_cache(monkeypatch) -> None:
    """Preprocessing can only hit the cache the encoder stage writes."""
    created: dict[str, object] = {}

    class RecordingPreprocessor:
        def __init__(self, model_path: str, **kwargs: object) -> None:
            created.update(kwargs)

    monkeypatch.setattr(stages_mod, "MiniCPMOPreprocessor", RecordingPreprocessor)
    monkeypatch.setattr(stages_mod, "register_minicpm_o_hf_config", lambda: None)
    monkeypatch.setattr(
        stages_mod,
        "resolve_concrete_device",
        lambda *args, **kwargs: torch.device("cpu"),
    )

    stages_mod.create_preprocessing_executor("checkpoint", max_concurrency=1)

    assert created["image_cache"] is stages_mod.encoder_output_cache(IMAGE_STAGE)


def test_projection_copies_the_cached_result_container() -> None:
    """A shared container would send the payload through the relay, not a handoff."""
    cached_result = {"image_embeds": torch.ones(2, 3)}
    payload = build_payload(
        {"_active": True, "cache_key": "image-key", "cached_result": cached_result},
        stage_name=IMAGE_STAGE,
    )

    projected = project_preprocessing_to_encoder(payload, stage_name=IMAGE_STAGE)
    stage_inputs = MiniCPMOPipelineState.from_dict(projected.data).encoder_inputs[
        IMAGE_STAGE
    ]

    assert stage_inputs["cached_result"] == cached_result
    assert stage_inputs["cached_result"] is not cached_result


@pytest.mark.parametrize(
    "stage_inputs,expected_dispatch,expected_wait",
    [
        (
            {"_active": True, "cached_result": {"image_embeds": torch.zeros(2, 3)}},
            ["image_encoder", "thinker"],
            ["preprocessing", "image_encoder"],
        ),
        (
            {"pixel_values": [torch.zeros(3, 4, 4)], "tgt_sizes": torch.zeros(1, 2)},
            ["image_encoder", "thinker"],
            ["preprocessing", "image_encoder"],
        ),
        (
            {"cache_key": "image-key"},
            ["thinker"],
            ["preprocessing"],
        ),
    ],
    ids=["cache-hit", "cache-miss", "no-image"],
)
def test_image_branch_is_dispatched_without_pixel_values(
    stage_inputs: dict[str, object],
    expected_dispatch: list[str],
    expected_wait: list[str],
) -> None:
    """A hit ships no pixel values, so _active has to keep the branch scheduled."""
    payload = build_payload(stage_inputs, stage_name=IMAGE_STAGE)

    assert (
        resolve_preprocessing_next_stages("cached-image", payload) == expected_dispatch
    )
    assert (
        resolve_thinker_wait_sources("cached-image", "preprocessing", payload)
        == expected_wait
    )
