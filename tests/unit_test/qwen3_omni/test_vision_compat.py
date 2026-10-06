# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for Qwen3-Omni vision encoding."""

from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

import sglang_omni.models.qwen3_omni.components.vision_encoder as vision_runtime
from sglang_omni.models.qwen3_omni.components import image_encoder, vision_compat
from sglang_omni.models.qwen3_omni.components.vision_encoder import (
    Qwen3OmniVisionEncoder,
)


def make_encoder() -> vision_compat.Qwen3OmniMoeVisionEncoderCompat:
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


def transformers_5_6_pos_embed_interpolate(
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
    encoder = make_encoder()
    grid = torch.tensor(grid_thw, dtype=torch.long)

    expected = transformers_5_6_pos_embed_interpolate(encoder, grid)
    actual = encoder.legacy_pos_embed_interpolate(grid)

    assert torch.equal(actual, expected)


def apply_cpu_joint_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
) -> None:
    assert is_neox
    assert cos_sin_cache.dtype == torch.float32
    cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
    rotated_query, rotated_key = hf_modeling.apply_rotary_pos_emb_vision(
        query, key, torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)
    )
    query.copy_(rotated_query)
    key.copy_(rotated_key)


@pytest.fixture
def vision_encoder() -> vision_compat.Qwen3OmniMoeVisionEncoderCompat:
    # note (yzxiao): Keep the serving head dimension (72) in the small CPU model.
    config = hf_modeling.Qwen3OmniMoeVisionEncoderConfig(
        depth=2,
        hidden_size=144,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=16,
        num_position_embeddings=64,
        deepstack_visual_indexes=(0, 1),
    )
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.manual_seed(734)
        encoder = vision_compat.Qwen3OmniMoeVisionEncoderCompat(config)
    return encoder.eval().to(torch.bfloat16)


@pytest.mark.parametrize(
    ("platform_device", "has_provider", "should_use_joint_kernel"),
    [("cpu", True, True), ("cpu", False, False), ("cuda", True, False)],
    ids=["joint-provider", "native-rope", "model-platform-mismatch"],
)
def test_image_encoder_preserves_outputs_with_platform_rotary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    vision_encoder: vision_compat.Qwen3OmniMoeVisionEncoderCompat,
    platform_device: Literal["cpu", "cuda"],
    has_provider: bool,
    should_use_joint_kernel: bool,
) -> None:
    save_file(
        {
            f"thinker.visual.{name}": tensor
            for name, tensor in vision_encoder.state_dict().items()
        },
        tmp_path / "model.safetensors",
    )
    kernel = Mock(side_effect=apply_cpu_joint_rope)
    provider = Mock(return_value=kernel if has_provider else None)
    monkeypatch.setattr(
        vision_runtime,
        "current_platform",
        SimpleNamespace(
            device_type=platform_device, get_joint_rope_inplace_kernel=provider
        ),
    )
    monkeypatch.setattr(
        image_encoder,
        "load_thinker_config",
        Mock(return_value=SimpleNamespace(vision_config=vision_encoder.config)),
    )
    model = image_encoder.Qwen3OmniImageEncoder(
        str(tmp_path), device="cpu", dtype="bf16"
    )
    image_encoder.optimize_patch_embed(vision_encoder)
    image_grid = torch.tensor([[1, 4, 4], [1, 4, 6]], dtype=torch.long, device="cpu")
    video_grid = torch.tensor([[2, 6, 4]], dtype=torch.long, device="cpu")
    image_pixels = torch.randn(40, 24, dtype=torch.bfloat16, device="cpu")
    video_pixels = torch.randn(48, 24, dtype=torch.bfloat16, device="cpu")

    with torch.no_grad():
        actual = model(
            pixel_values=image_pixels,
            image_grid_thw=image_grid,
            pixel_values_videos=video_pixels,
            video_grid_thw=video_grid,
        )
        expected_image = vision_encoder(
            image_pixels, grid_thw=image_grid, return_dict=True
        )
        expected_video = vision_encoder(
            video_pixels, grid_thw=video_grid, return_dict=True
        )

    torch.testing.assert_close(
        actual,
        {
            "image_embeds": expected_image.pooler_output,
            "image_grid_thw": image_grid,
            "image_token_counts": torch.tensor([4, 6], device="cpu"),
            "deepstack_visual_embeds_image": expected_image.deepstack_features,
            "video_embeds": expected_video.pooler_output,
            "video_grid_thw": video_grid,
            "video_token_counts": torch.tensor([12], device="cpu"),
            "deepstack_visual_embeds_video": expected_video.deepstack_features,
        },
        rtol=0,
        atol=0,
    )
    expected_kernel_calls = (
        2 * vision_encoder.config.depth if should_use_joint_kernel else 0
    )
    assert kernel.call_count == expected_kernel_calls
    assert provider.call_count == (1 if platform_device == "cpu" else 0)


@pytest.mark.parametrize(
    "has_joint_kernel", [False, True], ids=["native-rope", "joint-rope"]
)
def test_flash_backend_receives_host_max_sequence_lengths(
    monkeypatch: pytest.MonkeyPatch,
    vision_encoder: vision_compat.Qwen3OmniMoeVisionEncoderCompat,
    has_joint_kernel: bool,
) -> None:
    kernel = Mock(side_effect=apply_cpu_joint_rope)
    encoder = (
        Qwen3OmniVisionEncoder(
            vision_encoder.config,
            joint_rope_kernel=kernel if has_joint_kernel else None,
        )
        .eval()
        .to(torch.bfloat16)
    )
    encoder.load_state_dict(vision_encoder.state_dict(), strict=True)
    grid = torch.tensor([[1, 4, 4], [2, 4, 6]], dtype=torch.long, device="cpu")
    pixels = torch.randn(64, 24, dtype=torch.bfloat16, device="cpu")
    attention_output = pixels.new_zeros(
        1,
        pixels.shape[0],
        encoder.config.num_heads,
        encoder.config.hidden_size // encoder.config.num_heads,
    )
    flash_backend = Mock(return_value=(attention_output, None))
    monkeypatch.setattr(
        hf_modeling, "is_flash_attention_requested", Mock(return_value=True)
    )
    monkeypatch.setattr(
        hf_modeling.ALL_ATTENTION_FUNCTIONS,
        "get_interface",
        Mock(return_value=flash_backend),
    )

    with torch.no_grad():
        encoder(pixels, grid_thw=grid)

    assert flash_backend.call_count == vision_encoder.config.depth
    for backend_call in flash_backend.call_args_list:
        assert isinstance(backend_call.kwargs["max_length_q"], int)
        assert isinstance(backend_call.kwargs["max_length_k"], int)
        assert backend_call.kwargs["max_length_q"] == 24
        assert backend_call.kwargs["max_length_k"] == 24
