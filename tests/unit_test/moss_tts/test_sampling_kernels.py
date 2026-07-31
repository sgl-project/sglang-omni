# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for MOSS-TTS sampling kernels."""

from __future__ import annotations

import pytest
import torch
from sglang.srt.layers.sampler import multinomial_with_seed

from sglang_omni.models.moss_tts.sampling_kernels import seeded_gumbel_argmax


@pytest.mark.parametrize("equal_scores", [False, True], ids=["random", "tied"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_seeded_gumbel_argmax_matches_production_shape(
    equal_scores: bool,
) -> None:
    device = torch.device("cuda")
    rows, vocab = 32, 1025
    scores = (
        torch.zeros(rows, vocab, device=device, dtype=torch.float32)
        if equal_scores
        else torch.randn(rows, vocab, device=device, dtype=torch.float32)
    )
    seeds = torch.tensor([20260720], device=device, dtype=torch.long).expand(rows)
    positions = torch.arange(rows, device=device, dtype=torch.long)
    output = torch.empty(rows, device=device, dtype=torch.long)

    expected = multinomial_with_seed(scores, seeds, positions).view(-1)
    actual = seeded_gumbel_argmax(scores, seeds, positions, output)
    torch.cuda.synchronize()

    assert seeds.stride(0) == 0
    assert torch.equal(expected, actual)
