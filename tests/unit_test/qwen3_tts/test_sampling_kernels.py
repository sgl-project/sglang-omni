# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS optional sampling kernels."""

from __future__ import annotations

import pytest
import torch
from sglang.srt.layers.sampler import multinomial_with_seed

from sglang_omni.models.qwen3_tts.sampling_kernels import (
    sample_from_sorted_probs_with_seed_small_k,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Qwen3-TTS sampling kernel needs CUDA"
)


@pytest.mark.parametrize("batch_size,num_cols", [(1, 1), (3, 2), (7, 30), (16, 64)])
def test_seeded_small_k_sampler_matches_sglang_multinomial(
    batch_size: int, num_cols: int
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(batch_size * 100 + num_cols)
    probs = torch.rand(
        batch_size,
        num_cols,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    probs = probs / probs.sum(dim=1, keepdim=True)
    sorted_idx = torch.randint(
        0,
        8192,
        (batch_size, num_cols),
        generator=generator,
        device="cuda",
        dtype=torch.long,
    )
    seeds = torch.arange(17, 17 + batch_size, device="cuda", dtype=torch.long)
    positions = torch.arange(3, 3 + batch_size, device="cuda", dtype=torch.long)

    sampled = sample_from_sorted_probs_with_seed_small_k(
        probs, sorted_idx, seeds, positions
    )
    assert sampled is not None

    sampled_rank = multinomial_with_seed(probs, seeds, positions).view(-1, 1)
    expected = sorted_idx.gather(1, sampled_rank).view(-1)
    assert torch.equal(sampled, expected)


def test_seeded_small_k_sampler_falls_back_for_cpu() -> None:
    probs = torch.ones((1, 2), dtype=torch.float32)
    sorted_idx = torch.arange(2, dtype=torch.long).view(1, 2)
    seeds = torch.ones((1,), dtype=torch.long)
    positions = torch.zeros((1,), dtype=torch.long)

    assert (
        sample_from_sorted_probs_with_seed_small_k(probs, sorted_idx, seeds, positions)
        is None
    )
