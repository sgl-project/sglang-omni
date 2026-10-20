# SPDX-License-Identifier: Apache-2.0
"""MOSS seeded sampling using device tensor operations on Ascend."""

from __future__ import annotations

import torch

UINT32_MASK = (1 << 32) - 1
MOSS_HASH_SHIFT_CONSTANTS: dict[torch.device, tuple[torch.Tensor, ...]] = {}


def moss_hash_shift_constants(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Keep distinct device operands alive across sampling graph replays."""
    if device not in MOSS_HASH_SHIFT_CONSTANTS:
        MOSS_HASH_SHIFT_CONSTANTS[device] = torch.tensor(
            [32, 15, 17, 13, 19, 16], dtype=torch.int64, device=device
        ).unbind()
    else:
        pass
    return MOSS_HASH_SHIFT_CONSTANTS[device]


def moss_hash32(
    seeds: torch.Tensor, positions: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """Hash seed halves, position and original token ID with uint32 wrapping."""
    seeds = seeds.to(torch.int64)
    positions = positions.to(torch.int64)
    token_ids = token_ids.to(torch.int64)
    shift32, shift15, shift17, shift13, shift19, shift16 = moss_hash_shift_constants(
        seeds.device
    )
    values = torch.zeros(
        (seeds.numel(), token_ids.numel()), dtype=torch.int64, device=seeds.device
    )
    for block in (
        (seeds & UINT32_MASK)[:, None],
        ((seeds >> shift32) & UINT32_MASK)[:, None],
        (positions & UINT32_MASK)[:, None],
        (token_ids & UINT32_MASK)[None, :],
    ):
        keys = (block * 0xCC9E2D51) & UINT32_MASK
        keys = ((keys << shift15) | (keys >> shift17)) & UINT32_MASK
        keys = (keys * 0x1B873593) & UINT32_MASK
        values = values ^ keys
        values = ((values << shift13) | (values >> shift19)) & UINT32_MASK
        values = (values * 5 + 0xE6546B64) & UINT32_MASK
    values = values ^ 16
    values = values ^ (values >> shift16)
    values = (values * 0x85EBCA6B) & UINT32_MASK
    values = values ^ (values >> shift13)
    values = (values * 0xC2B2AE35) & UINT32_MASK
    return values ^ (values >> shift16)


def multinomial_with_seed_npu(
    scores: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return sampled column indices using the MOSS seeded Gumbel rule."""
    import torch_npu

    assert scores.device.type == "npu"
    columns = (
        torch.arange(scores.shape[1], device=scores.device)
        if token_ids is None
        else token_ids
    )
    hashes = moss_hash32(seeds, positions, columns)
    noise = torch_npu.npu_dtype_cast(hashes, torch.float64) / UINT32_MASK
    noise.log_().clamp_(min=torch.finfo(torch.float64).min, max=-(2.0**-32)).neg_()
    noise.log_().neg_()
    noise.add_(torch_npu.npu_dtype_cast(scores, torch.float64))
    return torch.argmax(noise, dim=1)
