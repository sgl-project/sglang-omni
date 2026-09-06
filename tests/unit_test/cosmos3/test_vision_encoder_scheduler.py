# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang_omni.models.cosmos3 import vision_encoder_scheduler
from sglang_omni.models.cosmos3.payload_types import Cosmos3PipelineState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.stage_cache import StageOutputCache


class _FakeVisionModel:
    spatial_merge_size = 2
    out_hidden_size = 4
    deepstack_layers = 1
    visual_dtype_bytes = 4

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **inputs):
        self.calls += 1
        grid = inputs["image_grid_thw"]
        token_count = grid.prod(-1) // 4
        total = int(token_count.sum().item())
        embeds = torch.arange(total * 4, dtype=torch.float32).reshape(total, 4)
        return {
            "image_embeds": embeds,
            "image_grid_thw": grid,
            "image_token_counts": token_count,
            "deepstack_visual_embeds_image": [embeds + 1],
        }


def _vision_payload(request_id: str, cache_key: str) -> StagePayload:
    state = Cosmos3PipelineState(
        encoder_inputs={
            "vision_encoder": {
                "pixel_values": torch.ones((4, 6)),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "image_cache_key": cache_key,
            }
        }
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=None),
        data=state.to_dict(),
    )


def test_vision_encoder_batches_and_deduplicates_same_cache_key() -> None:
    model = _FakeVisionModel()
    cache = StageOutputCache(max_size=4, cache_device="cpu")
    outputs = vision_encoder_scheduler._batch_vision_requests(
        [
            _vision_payload("first", "same-image"),
            _vision_payload("second", "same-image"),
        ],
        model=model,
        cache=cache,
    )

    assert model.calls == 1
    assert len(cache) == 1
    first = Cosmos3PipelineState.from_dict(outputs[0].data)
    second = Cosmos3PipelineState.from_dict(outputs[1].data)
    assert torch.equal(
        first.encoder_outs["vision_encoder"]["image_embeds"],
        second.encoder_outs["vision_encoder"]["image_embeds"],
    )
