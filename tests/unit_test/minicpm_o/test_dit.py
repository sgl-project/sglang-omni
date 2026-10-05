# SPDX-License-Identifier: Apache-2.0
"""Tests for MiniCPM-o DiT timestep embedding and variable-length execution."""

from __future__ import annotations

import math

import pytest
import torch

from sglang_omni.models.minicpm_o.components.token2wav.dit import (
    CausalConvBlock,
    DiT,
    TimestepEmbedder,
)

TIMESTEP_MAX_PERIOD = 10000
TIMESTEP_SCALE = 1000
PACKED_MAX_RELATIVE_RMS_ERROR = 5e-2


def reference_timestep_embedding(
    timesteps: torch.Tensor, frequency_embedding_size: int
) -> torch.Tensor:
    half = frequency_embedding_size // 2
    frequencies = torch.exp(-math.log(TIMESTEP_MAX_PERIOD) * torch.arange(half) / half)
    angles = (timesteps * TIMESTEP_SCALE)[:, None] * frequencies.to(timesteps)[None]
    embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
    if frequency_embedding_size % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        pass
    return embedding


def relative_rms_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        ((actual - expected).square().mean() / expected.square().mean()).sqrt().item()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("frequency_embedding_size", [255, 256])
def test_timestep_embedding_matches_reference(
    dtype: torch.dtype, frequency_embedding_size: int
) -> None:
    embedder = TimestepEmbedder(16, frequency_embedding_size).to(dtype).eval()
    timesteps = torch.linspace(0, 1, 11, dtype=dtype)
    expected = embedder.mlp(
        reference_timestep_embedding(timesteps, frequency_embedding_size)
    )
    torch.testing.assert_close(embedder(timesteps), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.bfloat16])
def test_timestep_embedding_autocast_keeps_fp32_frequencies(
    weight_dtype: torch.dtype,
) -> None:
    embedder = TimestepEmbedder(16).to(device="cuda", dtype=weight_dtype).eval()
    timesteps = torch.linspace(0, 1, 11, device="cuda", dtype=torch.float32)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=weight_dtype):
        expected = embedder.mlp(reference_timestep_embedding(timesteps, 256))
        torch.testing.assert_close(embedder(timesteps), expected, rtol=0, atol=0)


def test_packed_causal_conv_preserves_sequence_boundaries() -> None:
    torch.manual_seed(0)
    block = CausalConvBlock(4, 4).eval()
    causal_padding_frames = block.kernel_size - 1
    rows = [torch.randn(length, 4) for length in (3, 5, 2)]
    expected = torch.cat([block(row.unsqueeze(0))[0].squeeze(0) for row in rows])
    sequence_lengths = torch.tensor([len(row) for row in rows])
    frame_count = int(sequence_lengths.sum())
    sequence_ids = torch.repeat_interleave(torch.arange(len(rows)), sequence_lengths)
    real_frame_positions = (
        torch.arange(frame_count) + (sequence_ids + 1) * causal_padding_frames
    )
    real_frame_mask = torch.zeros(
        frame_count + len(rows) * causal_padding_frames, dtype=torch.bool
    )
    real_frame_mask[real_frame_positions] = True
    actual = block.forward_packed(
        torch.cat(rows), real_frame_positions, real_frame_mask
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_dit_matches_padded_dit_on_valid_frames() -> None:
    torch.manual_seed(0)
    channels = 8
    lengths = [5, 9, 3]
    batch_size, padded_length = len(lengths), max(lengths)
    model = DiT(
        in_channels=4 * channels,
        out_channels=channels,
        depth=2,
        num_heads=2,
        head_dim=32,
        hidden_size=64,
    )
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    model = model.cuda().eval()
    frame_indices = torch.arange(padded_length).unsqueeze(0)
    mask = (frame_indices < torch.tensor(lengths).unsqueeze(1)).unsqueeze(1)
    mask = mask.float().cuda()
    noisy_mel, mu, cond = (
        torch.randn(batch_size, channels, padded_length, device="cuda")
        for _ in range(3)
    )
    speaker_embeddings = torch.randn(batch_size, channels, device="cuda")
    timesteps = torch.rand(batch_size, device="cuda")
    with torch.inference_mode():
        padded = model(noisy_mel, mask, mu, timesteps, speaker_embeddings, cond)
        model.enable_variable_length = True
        packed = model(noisy_mel, mask, mu, timesteps, speaker_embeddings, cond)
    for row, length in enumerate(lengths):
        error = relative_rms_error(
            packed[row, :, :length].float(), padded[row, :, :length]
        )
        assert error < PACKED_MAX_RELATIVE_RMS_ERROR, f"row {row}"


def test_variable_length_stays_padded_off_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    channels = 8
    lengths = [5, 9, 3]
    batch_size, padded_length = len(lengths), max(lengths)
    model = DiT(
        in_channels=4 * channels,
        out_channels=channels,
        depth=2,
        num_heads=2,
        head_dim=32,
        hidden_size=64,
    ).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    frame_indices = torch.arange(padded_length).unsqueeze(0)
    mask = (frame_indices < torch.tensor(lengths).unsqueeze(1)).unsqueeze(1).float()
    noisy_mel, mu, cond = (
        torch.randn(batch_size, channels, padded_length) for _ in range(3)
    )
    speaker_embeddings = torch.randn(batch_size, channels)
    timesteps = torch.rand(batch_size)

    def reject_packed_forward(
        hidden: torch.Tensor,
        conditioning: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        raise AssertionError(
            "packed attention is unavailable off CUDA, "
            f"hidden={tuple(hidden.shape)} conditioning={tuple(conditioning.shape)} "
            f"sequence_lengths={tuple(sequence_lengths.shape)}"
        )

    monkeypatch.setattr(model, "forward_packed", reject_packed_forward)
    model.enable_variable_length = True
    with torch.inference_mode():
        variable_length = model(
            noisy_mel, mask, mu, timesteps, speaker_embeddings, cond
        )
        model.enable_variable_length = False
        padded = model(noisy_mel, mask, mu, timesteps, speaker_embeddings, cond)
    torch.testing.assert_close(variable_length, padded)
