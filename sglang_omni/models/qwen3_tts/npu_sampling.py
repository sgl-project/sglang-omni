# SPDX-License-Identifier: Apache-2.0
"""Fused Ascend Predictor seeded sampling."""

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
def gumbel32_values(
    seeds: tl.tensor, positions: tl.tensor, row: tl.tensor, column: tl.tensor
) -> tl.tensor:
    low = tl.load(seeds + row * 2).to(tl.uint32, bitcast=True)
    high = tl.load(seeds + row * 2 + 1).to(tl.uint32, bitcast=True)
    position = tl.load(positions + row * 2).to(tl.uint32, bitcast=True)
    value = tl.full(column.shape, 0, tl.uint32)
    value = mix32(value, low)
    value = mix32(value, high)
    value = mix32(value, position)
    value = mix32(value, column.to(tl.uint32)) ^ 16
    value = value ^ ((value >> 16) & 0xFFFF)
    value = (value * 0x85EBCA6B).to(tl.uint32)
    value = value ^ ((value >> 13) & 0x7FFFF)
    value = (value * 0xC2B2AE35).to(tl.uint32)
    value = value ^ ((value >> 16) & 0xFFFF)
    hashes = value
    # note (cocoa): Ascend lacks uint32-to-float; both 16-bit halves convert exactly.
    high = ((hashes >> 16) & 0xFFFF).to(tl.int32).to(tl.float32)
    low = (hashes & 0xFFFF).to(tl.int32).to(tl.float32)
    uniform = high * (1.0 / 65536.0) + low * (1.0 / 4294967296.0)
    uniform = tl.minimum(tl.maximum(uniform, 1.1754943508222875e-38), 1.0 - 2.0**-24)
    return -tl.log(-tl.log(uniform))


@triton.jit
def first_argmax(
    scores: tl.tensor, column: tl.tensor, width: tl.constexpr, block: tl.constexpr
) -> tl.tensor:
    maximum = tl.max(scores, axis=0)
    winner = tl.min(
        tl.where((column < width) & (scores == maximum), column, block), axis=0
    )
    first_nan = tl.min(
        tl.where((column < width) & (scores != scores), column, block), axis=0
    )
    return tl.where(first_nan < width, first_nan, winner)


@triton.jit
def gumbel_argmax_kernel(
    logprobs: tl.tensor,
    seeds: tl.tensor,
    positions: tl.tensor,
    output: tl.tensor,
    row_stride: tl.constexpr,
    column_stride: tl.constexpr,
    width: tl.constexpr,
    block: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    column = tl.arange(0, block)
    values = tl.load(
        logprobs + row * row_stride + column * column_stride,
        column < width,
        other=-float("inf"),
    ).to(tl.float32)
    scores = values + gumbel32_values(seeds, positions, row, column)
    winner = first_argmax(scores, column, width, block)
    tl.store(output + row, winner.to(tl.int64))


@triton.jit
def top_k_sample_kernel(
    scores: tl.tensor,
    indices: tl.tensor,
    top_ks: tl.tensor,
    seeds: tl.tensor,
    positions: tl.tensor,
    output: tl.tensor,
    score_stride: tl.constexpr,
    index_stride: tl.constexpr,
    width: tl.constexpr,
    block: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    column = tl.arange(0, block)
    top_k = tl.load(top_ks + row)
    values = tl.load(
        scores + row * score_stride + column, column < width, other=-float("inf")
    )
    values = tl.where(column < top_k, values, -float("inf"))
    probabilities = tl.exp(values - tl.max(values, axis=0))
    probabilities = probabilities / tl.sum(probabilities, axis=0)
    logprobs = tl.where(probabilities > 0, tl.log(probabilities), -float("inf"))
    sampled_scores = logprobs + gumbel32_values(seeds, positions, row, column)
    winner = first_argmax(sampled_scores, column, width, block)
    token = tl.load(indices + row * index_stride + winner)
    tl.store(output + row, token)


def sample_top_k_npu(
    sorted_scores: torch.Tensor,
    sorted_indices: torch.Tensor,
    top_ks: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Sample bounded sorted FP32 logits with per-row Top-K and no Top-P."""
    seeds = seeds.to(torch.int64).contiguous().view(torch.int32)
    positions = positions.to(torch.int64).contiguous().view(torch.int32)
    batch_size, width = sorted_scores.shape
    result = torch.empty(batch_size, dtype=torch.long, device=sorted_scores.device)
    if batch_size:
        top_k_sample_kernel[(batch_size,)](
            sorted_scores,
            sorted_indices,
            top_ks,
            seeds,
            positions,
            result,
            sorted_scores.stride(0),
            sorted_indices.stride(0),
            width,
            triton.next_power_of_2(width),
        )
    else:
        pass
    return result


def seeded_gumbel_argmax_npu(
    logprobs: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Fuse hashing, float32 Gumbel noise and first-index argmax on NPU."""
    seeds = seeds.to(torch.int64).contiguous().view(torch.int32)
    positions = positions.to(torch.int64).contiguous().view(torch.int32)
    batch_size, width = logprobs.shape
    result = torch.empty(batch_size, dtype=torch.long, device=logprobs.device)
    if batch_size:
        gumbel_argmax_kernel[(batch_size,)](
            logprobs,
            seeds,
            positions,
            result,
            logprobs.stride(0),
            logprobs.stride(1),
            width,
            triton.next_power_of_2(width),
        )
    else:
        pass
    return result
