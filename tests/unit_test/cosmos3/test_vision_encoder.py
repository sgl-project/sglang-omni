# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from transformers import Qwen3VLVisionConfig, Qwen3VLVisionModel

from sglang_omni.models.cosmos3.components import vision_encoder
from sglang_omni.models.cosmos3.components.vision_encoder import Cosmos3VisionEncoder


def _tiny_vision_config() -> Qwen3VLVisionConfig:
    return Qwen3VLVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=8,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
    )


def test_vision_encoder_loads_standalone_hf_tower_and_returns_deepstack(
    monkeypatch,
    tmp_path,
) -> None:
    config = _tiny_vision_config()
    checkpoint = tmp_path / "vision_encoder"
    checkpoint.mkdir()
    reference = Qwen3VLVisionModel(config)
    save_file(reference.state_dict(), checkpoint / "model.safetensors")

    monkeypatch.setattr(
        vision_encoder,
        "load_hf_config",
        lambda *args, **kwargs: SimpleNamespace(vision_config=config),
    )
    monkeypatch.setattr(
        vision_encoder,
        "resolve_vision_encoder_path",
        lambda model_path: str(checkpoint),
    )
    encoder = Cosmos3VisionEncoder(
        "unused-model",
        device="cpu",
        dtype="float32",
    )
    output = encoder(
        pixel_values=torch.randn(4, 24),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
    )

    assert output["image_embeds"].shape == (1, 8)
    assert output["image_token_counts"].tolist() == [1]
    assert len(output["deepstack_visual_embeds_image"]) == 1
    assert output["deepstack_visual_embeds_image"][0].shape == (1, 8)
    assert hasattr(encoder.visual.patch_embed, "linear")
