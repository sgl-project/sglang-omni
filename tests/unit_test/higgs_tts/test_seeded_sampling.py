# SPDX-License-Identifier: Apache-2.0
"""Seeded reproducibility of the Higgs batched sampler.

``multinomial_with_seed`` uses a Triton kernel, so these run on CUDA only.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.sampler import _sample_independent_batched

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="multinomial_with_seed needs CUDA"
)

B, N, V = 3, 8, 64


def _logits(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(B, N, V, generator=g, device="cuda")


def _draw(logits, seeds_B, step):
    return _sample_independent_batched(
        logits,
        temperature=torch.ones(B, device="cuda"),
        top_p=None,
        seeds_B=torch.tensor(seeds_B, device="cuda"),
        step_B=torch.full((B,), step, dtype=torch.long, device="cuda"),
    )


def test_same_seed_same_step_is_reproducible():
    logits = _logits(0)
    assert torch.equal(_draw(logits, [123] * B, 4), _draw(logits, [123] * B, 4))


def test_different_seed_differs():
    logits = _logits(1)
    assert not torch.equal(_draw(logits, [1] * B, 4), _draw(logits, [2] * B, 4))


def test_position_decorrelates_steps():
    logits = _logits(2)
    assert not torch.equal(_draw(logits, [9] * B, 0), _draw(logits, [9] * B, 1))


def test_per_row_seed_isolation():
    """A row's draw depends only on its own seed, not its batch neighbours."""
    logits = _logits(4)
    full = _draw(logits, [55, 77, 99], step=3)
    alone = _sample_independent_batched(
        logits[:1],
        temperature=torch.ones(1, device="cuda"),
        top_p=None,
        seeds_B=torch.tensor([55], device="cuda"),
        step_B=torch.full((1,), 3, dtype=torch.long, device="cuda"),
    )
    assert torch.equal(alone[0], full[0])


def test_seeded_sampler_preserves_probability_distribution():
    batch = 20_000
    probs = torch.tensor([0.9, 0.1], device="cuda")
    logits = probs.log().view(1, 1, 2).expand(batch, 1, 2).contiguous()
    sampled = _sample_independent_batched(
        logits,
        temperature=torch.ones(batch, device="cuda"),
        top_p=None,
        seeds_B=torch.arange(1, batch + 1, device="cuda"),
        step_B=torch.zeros(batch, dtype=torch.long, device="cuda"),
    )
    token0_rate = (sampled[:, 0] == 0).float().mean().item()
    assert 0.87 < token0_rate < 0.93
