# SPDX-License-Identifier: Apache-2.0
"""CPU boundary for Higgs seeded sampling on Apple MPS.

MPS has neither Triton nor float64 tensors. Match the pinned SGLang 0.5.18
MurmurHash/Gumbel construction on CPU, then return token IDs to the input
device. This is a compatibility implementation, not a native MLX sampler.
Reference: sglang/kernels/ops/sampling/murmur_hash.py and
sglang/srt/layers/sampler.py in sgl-project/sglang at v0.5.18.
"""

from __future__ import annotations

import torch

_UINT32_MASK = 0xFFFFFFFF


def _rotate_left(value: torch.Tensor, bits: int) -> torch.Tensor:
    return ((value << bits) | (value >> (32 - bits))) & _UINT32_MASK


def _murmur_hash_cpu(seeds: torch.Tensor, positions: torch.Tensor, vocab: int):
    # Signed int64 provides portable bit operations; mask each multiply to
    # preserve uint32 wraparound. No uint64 or float64 operation runs on MPS.
    seeds = seeds.detach().to(device="cpu", dtype=torch.int64).reshape(-1, 1)
    positions = positions.detach().to(device="cpu", dtype=torch.int64).reshape(-1, 1)
    columns = torch.arange(vocab, dtype=torch.int64).reshape(1, -1)
    hashed = torch.zeros_like(seeds)
    for block in (
        seeds & _UINT32_MASK,
        (seeds >> 32) & _UINT32_MASK,
        positions & _UINT32_MASK,
        columns,
    ):
        mixed = (block * 0xCC9E2D51) & _UINT32_MASK
        mixed = (_rotate_left(mixed, 15) * 0x1B873593) & _UINT32_MASK
        hashed = (_rotate_left(hashed ^ mixed, 13) * 5 + 0xE6546B64) & _UINT32_MASK
    hashed = hashed ^ 16
    for shift, multiplier in ((16, 0x85EBCA6B), (13, 0xC2B2AE35)):
        hashed = ((hashed ^ (hashed >> shift)) * multiplier) & _UINT32_MASK
    return hashed ^ (hashed >> 16)


@torch.no_grad()
def multinomial_with_seed_cpu(
    logprobs: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Draw one token per row, independent of global RNG and batch neighbours.

    Inputs are [rows, vocabulary], [rows], [rows]; output is int64 [rows, 1]
    on the original device. Cross-device logits can still differ numerically,
    so this does not promise identical generated speech across backends.
    """
    if logprobs.ndim != 2 or logprobs.shape[1] == 0:
        raise ValueError("logprobs must have shape [rows, nonempty vocabulary]")
    if seeds.shape != (logprobs.shape[0],) or positions.shape != seeds.shape:
        raise ValueError("seeds and positions must have one entry per logprob row")
    hashes = _murmur_hash_cpu(seeds, positions, logprobs.shape[1])
    uniform = hashes.to(torch.float64) / _UINT32_MASK
    # Hash endpoints must not produce +inf noise that can select masked tokens.
    log_uniform = uniform.log().clamp(
        min=torch.finfo(torch.float64).min, max=-(2.0**-32)
    )
    scores = -(-log_uniform).log()
    # Transfer first: MPS does not support float64 conversion on the device.
    scores += logprobs.detach().cpu().to(dtype=torch.float64)
    return scores.argmax(dim=-1, keepdim=True).to(logprobs.device)
