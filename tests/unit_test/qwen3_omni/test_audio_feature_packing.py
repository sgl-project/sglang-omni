# SPDX-License-Identifier: Apache-2.0
"""Equivalence tests for packing padded audio features."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.audio_encoder import (
    pack_padded_audio_features,
)

MEL = 128


def _reference(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The expression this replaces, verbatim."""
    return features.permute(0, 2, 1)[mask.bool()].permute(1, 0).contiguous()


def _prefix_mask(lengths: list[int], total: int) -> torch.Tensor:
    steps = torch.arange(total).unsqueeze(0)
    return (steps < torch.tensor(lengths).unsqueeze(1)).to(torch.long)


def test_single_row_full_length() -> None:
    feats = torch.randn(1, MEL, 300)
    mask = _prefix_mask([300], 300)
    out = pack_padded_audio_features(feats, mask, mask.sum(dim=1))
    torch.testing.assert_close(out, _reference(feats, mask))


def test_single_row_padded() -> None:
    feats = torch.randn(1, MEL, 512)
    mask = _prefix_mask([200], 512)
    out = pack_padded_audio_features(feats, mask, mask.sum(dim=1))
    assert out.shape == (MEL, 200)
    torch.testing.assert_close(out, _reference(feats, mask))


def test_ragged_batch() -> None:
    lengths = [128, 400, 37, 400]
    feats = torch.randn(len(lengths), MEL, 400)
    mask = _prefix_mask(lengths, 400)
    out = pack_padded_audio_features(feats, mask, mask.sum(dim=1))
    assert out.shape == (MEL, sum(lengths))
    torch.testing.assert_close(out, _reference(feats, mask))


def test_ragged_batch_preserves_sample_then_frame_order() -> None:
    feats = torch.zeros(2, MEL, 4)
    feats[0, :, :2] = torch.tensor([10.0, 11.0]).repeat(MEL, 1)
    feats[1, :, :3] = torch.tensor([20.0, 21.0, 22.0]).repeat(MEL, 1)
    mask = _prefix_mask([2, 3], 4)

    out = pack_padded_audio_features(feats, mask, mask.sum(dim=1))

    assert out[0].tolist() == [10.0, 11.0, 20.0, 21.0, 22.0]


def test_non_prefix_mask_falls_back_to_gather() -> None:
    feats = torch.randn(2, MEL, 10)
    mask = torch.ones(2, 10, dtype=torch.long)
    mask[0, 3] = 0
    mask[1, 7:9] = 0
    out = pack_padded_audio_features(feats, mask, mask.sum(dim=1))
    torch.testing.assert_close(out, _reference(feats, mask))


def test_dtype_and_contiguity_preserved() -> None:
    feats = torch.randn(2, MEL, 64, dtype=torch.float32)
    mask = _prefix_mask([64, 20], 64)
    out = pack_padded_audio_features(feats, mask, mask.sum(dim=1))
    assert out.dtype == feats.dtype
    assert out.is_contiguous()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_accelerator_pack_matches_cpu_reference() -> None:
    lengths = [1000, 3000, 512]
    feats = torch.randn(len(lengths), MEL, 3000)
    mask = _prefix_mask(lengths, 3000)
    out = pack_padded_audio_features(feats.cuda(), mask, mask.sum(dim=1))
    assert out.device.type == "cuda"
    torch.testing.assert_close(out.cpu(), _reference(feats, mask))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_accelerator_fallback_gather_matches_cpu_reference() -> None:
    feats = torch.randn(2, MEL, 16)
    mask = torch.ones(2, 16, dtype=torch.long)
    mask[0, 5] = 0
    mask[1, 11:13] = 0
    out = pack_padded_audio_features(feats.cuda(), mask, mask.sum(dim=1))
    torch.testing.assert_close(out.cpu(), _reference(feats, mask))
