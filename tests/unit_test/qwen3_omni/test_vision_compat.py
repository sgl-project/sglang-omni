# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components import vision_compat


def _make_encoder() -> vision_compat.Qwen3OmniMoeVisionEncoderCompat:
    encoder = vision_compat.Qwen3OmniMoeVisionEncoderCompat.__new__(
        vision_compat.Qwen3OmniMoeVisionEncoderCompat
    )
    torch.nn.Module.__init__(encoder)
    encoder.config = SimpleNamespace(spatial_merge_size=2)
    encoder.num_grid_per_side = 48
    encoder.pos_embed = torch.nn.Embedding(
        48 * 48,
        16,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        values = torch.arange(
            encoder.pos_embed.weight.numel(),
            dtype=torch.float32,
        ).reshape_as(encoder.pos_embed.weight)
        encoder.pos_embed.weight.copy_(torch.sin(values / 113))
    return encoder


def _transformers_5_6_pos_embed_interpolate(
    encoder: vision_compat.Qwen3OmniMoeVisionEncoderCompat,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Frozen reference for the Transformers 5.6 interpolation implementation."""
    grid_thw_list = grid_thw.tolist()
    grid_ts = [row[0] for row in grid_thw_list]
    grid_hs = [row[1] for row in grid_thw_list]
    grid_ws = [row[2] for row in grid_thw_list]
    device = encoder.pos_embed.weight.device

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for _, h, w in grid_thw_list:
        h_idxs = torch.linspace(0, encoder.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, encoder.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=encoder.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=encoder.num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * encoder.num_grid_per_side
        base_h_ceil = h_idxs_ceil * encoder.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]
        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for corner in range(4):
            idx_list[corner].extend(indices[corner].tolist())
            weight_list[corner].extend(weights[corner].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(
        weight_list,
        dtype=encoder.pos_embed.weight.dtype,
        device=device,
    )
    corners = encoder.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    patch_pos_embeds = corners[0] + corners[1] + corners[2] + corners[3]
    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

    permuted = []
    merge_size = encoder.config.spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(
                t,
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
                -1,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        permuted.append(pos_embed)
    return torch.cat(permuted)


@pytest.mark.parametrize(
    "grid_thw",
    [
        [[1, 4, 6]],
        [[3, 6, 4]],
        [[1, 4, 6], [2, 6, 8]],
        [[1, 50, 52]],
    ],
    ids=["image", "video", "mixed-grid", "upsampling"],
)
def test_interpolation_is_bit_exact_to_transformers_5_6(grid_thw) -> None:
    encoder = _make_encoder()
    grid = torch.tensor(grid_thw, dtype=torch.long)

    expected = _transformers_5_6_pos_embed_interpolate(encoder, grid)
    actual = encoder._legacy_pos_embed_interpolate(grid)

    assert torch.equal(actual, expected)


def test_compat_encoder_preserves_transformers_5_12_output_contract() -> None:
    config = hf_modeling.Qwen3OmniMoeVisionEncoderConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=16,
        num_position_embeddings=64,
        deepstack_visual_indexes=(0,),
    )
    encoder = vision_compat.Qwen3OmniMoeVisionEncoderCompat(config)
    encoder.eval().to(torch.bfloat16)
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    hidden_states = torch.randn(16, 24, dtype=torch.bfloat16)

    with torch.inference_mode():
        output = encoder(hidden_states, grid_thw=grid)

    assert isinstance(output, hf_modeling.BaseModelOutputWithDeepstackFeatures)
    assert output.last_hidden_state.shape == (16, 16)
    assert output.pooler_output.shape == (4, 16)
    assert len(output.deepstack_features) == 1
    assert output.deepstack_features[0].shape == (4, 16)
