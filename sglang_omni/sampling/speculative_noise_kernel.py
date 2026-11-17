# SPDX-License-Identifier: Apache-2.0
"""Fused request-seeded exponential and acceptance noise for DSpark."""

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.sampling.murmur_hash import fmix32, murmur3_mix
from triton.language.extra import libdevice


@triton.jit
def seeded_noise_kernel(
    seeds: tl.tensor,
    output_lengths: tl.tensor,
    exponential_noise: tl.tensor,
    acceptance_noise: tl.tensor,
    seed_stride: tl.constexpr,
    length_stride: tl.constexpr,
    exponential_columns: tl.constexpr,
    acceptance_columns: tl.constexpr,
    exponential_salt: tl.constexpr,
    acceptance_salt: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_acceptance = offsets >= exponential_columns
    columns = tl.where(is_acceptance, offsets - exponential_columns, offsets)
    salt = tl.where(is_acceptance, acceptance_salt, exponential_salt).to(tl.uint64)
    seed = tl.load(seeds + row * seed_stride).to(tl.uint64) ^ salt
    position = tl.load(output_lengths + row * length_stride).to(tl.uint32)
    hashed = murmur3_mix(0, (seed & 0xFFFFFFFF).to(tl.uint32))
    hashed = murmur3_mix(hashed, (seed >> 32).to(tl.uint32))
    hashed = murmur3_mix(hashed, position)
    hashed = murmur3_mix(hashed, columns.to(tl.uint32))
    hashed = fmix32(hashed ^ 16)
    uniform = (hashed.to(tl.float64) + 0.5) * (1.0 / (2.0**32))
    exponential = -libdevice.log(uniform)
    tl.store(
        exponential_noise + row * exponential_columns + offsets,
        exponential.to(tl.float32),
        offsets < exponential_columns,
    )
    if acceptance_columns:
        tl.store(
            acceptance_noise + row * acceptance_columns + columns,
            tl.minimum(uniform.to(tl.float32), 1.0 - 2.0**-24),
            is_acceptance & (columns < acceptance_columns),
        )
    else:
        pass


def draw_seeded_noise(
    seeds: torch.Tensor,
    output_lengths: torch.Tensor,
    *,
    exponential_columns: int,
    exponential_salt: int,
    acceptance_columns: int = 0,
    acceptance_salt: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the existing salted hashes and float64 transforms in one launch."""
    exponential_noise = torch.empty(
        (seeds.numel(), exponential_columns), dtype=torch.float32, device=seeds.device
    )
    acceptance_noise = torch.empty(
        (seeds.numel(), acceptance_columns), dtype=torch.float32, device=seeds.device
    )
    seeded_noise_kernel[
        (seeds.numel(), triton.cdiv(exponential_columns + acceptance_columns, 256))
    ](
        seeds,
        output_lengths,
        exponential_noise,
        acceptance_noise,
        seeds.stride(0),
        output_lengths.stride(0),
        exponential_columns,
        acceptance_columns,
        exponential_salt,
        acceptance_salt,
        BLOCK_SIZE=256,
    )
    return exponential_noise, acceptance_noise
