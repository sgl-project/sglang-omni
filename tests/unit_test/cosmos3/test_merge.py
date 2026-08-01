# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang_omni.models.cosmos3.merge import merge_for_thinker
from sglang_omni.models.cosmos3.payload_types import Cosmos3PipelineState
from sglang_omni.proto import OmniRequest, StagePayload


def _payload(stage: str, state: Cosmos3PipelineState) -> StagePayload:
    return StagePayload(
        request_id="merge",
        request=OmniRequest(inputs=None),
        data=state.to_dict(),
    )


def test_merge_builds_thinker_inputs_without_duplicate_encoder_payload() -> None:
    prompt = {
        "input_ids": torch.tensor([10, 151655, 11]),
        "attention_mask": torch.ones(3, dtype=torch.long),
        "prompt_text": "prompt",
        "mm_token_type_ids": torch.tensor([0, 1, 0]),
    }
    preprocessing = Cosmos3PipelineState(
        prompt=prompt,
        mm_inputs={"image_grid_thw": torch.tensor([[1, 2, 2]])},
        encoder_inputs={"vision_encoder": {"cache_key": "key", "_active": True}},
    )
    embeddings = torch.ones((1, 4))
    deepstack = [torch.full((1, 4), index) for index in range(3)]
    vision = Cosmos3PipelineState(
        encoder_outs={
            "vision_encoder": {
                "image_embeds": embeddings,
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "deepstack_visual_embeds_image": deepstack,
            }
        }
    )

    result = merge_for_thinker(
        {
            "preprocessing": _payload("preprocessing", preprocessing),
            "vision_encoder": _payload("vision_encoder", vision),
        }
    )
    state = Cosmos3PipelineState.from_dict(result.data)
    model_inputs = state.thinker_inputs["model_inputs"]

    assert model_inputs["image_embeds"] is embeddings
    assert model_inputs["deepstack_visual_embeds"] is deepstack
    assert state.thinker_inputs["media_cache_key"] == "key"
    assert state.encoder_inputs == {}
    assert state.encoder_outs == {}
