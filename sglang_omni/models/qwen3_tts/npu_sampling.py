# SPDX-License-Identifier: Apache-2.0
"""Device-only seed hashing for Qwen3-TTS NPU graph capture."""

import torch
import triton
import triton.language as tl


@triton.jit
def mix32(value: tl.tensor, key: tl.tensor) -> tl.tensor:
    # note (cocoa): Masks prevent sign extension in Ascend right-shift lowering.
    key = (key * 0xCC9E2D51).to(tl.uint32)
    key = (key << 15) | ((key >> 17) & 0x7FFF)
    key = (key * 0x1B873593).to(tl.uint32)
    value = value ^ key
    value = (value << 13) | ((value >> 19) & 0x1FFF)
    return (value * 5 + 0xE6546B64).to(tl.uint32)


@triton.jit
def hash32_kernel(
    seeds: tl.tensor,
    positions: tl.tensor,
    output: tl.tensor,
    width: tl.constexpr,
    block: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    column = tl.program_id(1) * block + tl.arange(0, block)
    low = tl.load(seeds + row * 2).to(tl.uint32, bitcast=True)
    high = tl.load(seeds + row * 2 + 1).to(tl.uint32, bitcast=True)
    position = tl.load(positions + row * 2).to(tl.uint32, bitcast=True)
    value = tl.full((block,), 0, tl.uint32)
    value = mix32(value, low)
    value = mix32(value, high)
    value = mix32(value, position)
    value = mix32(value, column.to(tl.uint32)) ^ 16
    value = value ^ ((value >> 16) & 0xFFFF)
    value = (value * 0x85EBCA6B).to(tl.uint32)
    value = value ^ ((value >> 13) & 0x7FFFF)
    value = (value * 0xC2B2AE35).to(tl.uint32)
    value = value ^ ((value >> 16) & 0xFFFF)
    tl.store(
        output + row * width + column, value.to(tl.int64) & 0xFFFFFFFF, column < width
    )


def murmur_hash32_npu(
    seeds: torch.Tensor, positions: torch.Tensor, num_cols: int
) -> torch.Tensor:
    """Preserve 64-bit seed bits without NPU int64 rotate or host copies."""
    seeds = seeds.to(torch.int64).contiguous().view(torch.int32)
    positions = positions.to(torch.int64).contiguous().view(torch.int32)
    result = torch.empty(
        (seeds.numel() // 2, num_cols), dtype=torch.int64, device=seeds.device
    )
    hash32_kernel[(result.shape[0], triton.cdiv(num_cols, 256))](
        seeds, positions, result, num_cols, 256
    )
    return result
