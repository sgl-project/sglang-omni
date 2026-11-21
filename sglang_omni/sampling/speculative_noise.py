# SPDX-License-Identifier: Apache-2.0
"""Request-seeded random tensors for speculative drafting and verification."""

from __future__ import annotations

import torch
from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32

from sglang_omni.sampling.speculative_noise_kernel import draw_seeded_noise

# note (YifanLi3): a seeded step keys every draw by the committed length, which grows
# each step, so no draw is reused; the salts keep the three draws of a step apart.
DRAFT_NOISE_SEED_SALT = 0x68E31DA4
ACCEPTANCE_NOISE_SEED_SALT = 0x1D8E4E27
RESIDUAL_NOISE_SEED_SALT = 0x7A4C2D61
HASH_VALUE_COUNT = 2**32
# note (YifanLi3): the top hash buckets round to 1.0 in float32, and an acceptance
# uniform of 1.0 rejects a draft whose target and draft probabilities are equal.
LARGEST_FLOAT32_BELOW_ONE = 1.0 - 2**-24


def seeded_uniform_noise(
    seeds: torch.Tensor, output_lengths: torch.Tensor, *, salt: int, column_count: int
) -> torch.Tensor:
    """Float64 Uniform(0, 1) draws fixed by each row's seed and output length."""
    hashes = murmur_hash32(
        seeds ^ salt,
        output_lengths,
        torch.arange(column_count, device=seeds.device),
    )
    return (hashes.double() + 0.5) / HASH_VALUE_COUNT


def seeded_exponential_noise(
    seeds: torch.Tensor, output_lengths: torch.Tensor, *, salt: int, column_count: int
) -> torch.Tensor:
    """Float32 Exp(1) draws fixed by each row's seed and output length."""
    if seeds.device.type == "cuda":
        return draw_seeded_noise(
            seeds,
            output_lengths,
            exponential_columns=column_count,
            exponential_salt=salt,
        )[0]
    else:
        return (
            -seeded_uniform_noise(
                seeds, output_lengths, salt=salt, column_count=column_count
            ).log()
        ).float()
