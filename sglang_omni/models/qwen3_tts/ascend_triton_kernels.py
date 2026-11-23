# SPDX-License-Identifier: Apache-2.0
"""Ascend Triton kernels for Qwen3-TTS attention, hashing and sampling."""

import torch
import triton
import triton.language as tl


# Block sizes stay constexpr because tl.arange requires static tensor shapes.
@triton.jit(
    do_not_specialize=[
        "query_batch_stride",
        "query_head_stride",
        "key_batch_stride",
        "key_head_stride",
        "key_position_stride",
        "value_batch_stride",
        "value_head_stride",
        "value_position_stride",
        "length",
    ]
)
def short_cache_gqa_kernel(
    q: tl.tensor,
    key: tl.tensor,
    value: tl.tensor,
    output: tl.tensor,
    query_batch_stride: int,
    query_head_stride: int,
    key_batch_stride: int,
    key_head_stride: int,
    key_position_stride: int,
    value_batch_stride: int,
    value_head_stride: int,
    value_position_stride: int,
    length: int,
    block: tl.constexpr,
) -> None:
    batch = tl.program_id(0)
    head = tl.program_id(1)
    column = tl.arange(0, 128)
    position = tl.arange(0, block)
    query = tl.load(
        q + batch * query_batch_stride + head * query_head_stride + column
    ).to(tl.float32)
    keys = tl.load(
        key
        + batch * key_batch_stride
        + (head // 2) * key_head_stride
        + position[:, None] * key_position_stride
        + column[None, :],
        position[:, None] < length,
        other=0,
    ).to(tl.float32)
    scores = tl.sum(keys * query[None, :], axis=1) * (128**-0.5)
    scores = tl.where(position < length, scores, -float("inf"))
    scores = tl.exp(scores - tl.max(scores, axis=0))
    weights = scores / tl.sum(scores, axis=0)
    values = tl.load(
        value
        + batch * value_batch_stride
        + (head // 2) * value_head_stride
        + position[:, None] * value_position_stride
        + column[None, :],
        position[:, None] < length,
        other=0,
    ).to(tl.float32)
    result = tl.sum(values * weights[:, None], axis=0)
    tl.store(output + (batch * 16 + head) * 128 + column, result)


def short_cache_gqa_npu(
    q: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor | None:
    """Fuse BF16 1.7B Predictor decode: 16 query heads, 8 KV heads, head size 128.

    Only single-query caches of 1-16 positions and batches up to 32 are validated;
    other shapes use the general attention path.
    """
    if q.device.type != "npu":
        return None
    else:
        pass
    if (q.ndim, key.ndim) != (4, 4) or value.shape != key.shape:
        return None
    else:
        pass
    batch = q.shape[0]
    length = key.shape[2]
    if not (
        0 < batch <= 32
        and q.shape[1:] == (16, 1, 128)
        and key.shape == (batch, 8, length, 128)
        and 0 < length <= 16
        and q.dtype == key.dtype == value.dtype == torch.bfloat16
        and q.device == key.device == value.device
        and q.stride(-1) == key.stride(-1) == value.stride(-1) == 1
    ):
        return None
    else:
        pass
    result = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    short_cache_gqa_kernel[(batch, 16)](
        q,
        key,
        value,
        result,
        q.stride(0),
        q.stride(1),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        length,
        triton.next_power_of_2(length),
    )
    return result


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
def hash32_values(
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
    return value ^ ((value >> 16) & 0xFFFF)


@triton.jit(do_not_specialize=["width"])
def hash32_kernel(
    seeds: tl.tensor,
    positions: tl.tensor,
    output: tl.tensor,
    width: int,
    block: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    column = tl.program_id(1) * block + tl.arange(0, block)
    value = hash32_values(seeds, positions, row, column)
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


@triton.jit
def gumbel32_values(
    seeds: tl.tensor, positions: tl.tensor, row: tl.tensor, column: tl.tensor
) -> tl.tensor:
    hashes = hash32_values(seeds, positions, row, column)
    # Ascend lacks uint32-to-float; both 16-bit halves convert exactly.
    high = ((hashes >> 16) & 0xFFFF).to(tl.int32).to(tl.float32)
    low = (hashes & 0xFFFF).to(tl.int32).to(tl.float32)
    uniform = high * (1.0 / 65536.0) + low * (1.0 / 4294967296.0)
    uniform = tl.minimum(tl.maximum(uniform, 1.1754943508222875e-38), 1.0 - 2.0**-24)
    return -tl.log(-tl.log(uniform))


@triton.jit
def first_argmax(
    scores: tl.tensor, column: tl.tensor, width: int, block: tl.constexpr
) -> tl.tensor:
    maximum = tl.max(scores, axis=0)
    winner = tl.min(
        tl.where((column < width) & (scores == maximum), column, block), axis=0
    )
    first_nan = tl.min(
        tl.where((column < width) & (scores != scores), column, block), axis=0
    )
    return tl.where(first_nan < width, first_nan, winner)


# A static column stride avoids Ascend UB overflow for strided 2048-column inputs.
@triton.jit(do_not_specialize=["row_stride", "width"])
def gumbel_argmax_kernel(
    logprobs: tl.tensor,
    seeds: tl.tensor,
    positions: tl.tensor,
    output: tl.tensor,
    row_stride: int,
    column_stride: tl.constexpr,
    width: int,
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


@triton.jit(do_not_specialize=["score_stride", "index_stride", "width"])
def top_k_sample_kernel(
    scores: tl.tensor,
    indices: tl.tensor,
    top_ks: tl.tensor,
    seeds: tl.tensor,
    positions: tl.tensor,
    output: tl.tensor,
    score_stride: int,
    index_stride: int,
    width: int,
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
) -> torch.Tensor | None:
    """Sample bounded sorted FP32 logits with per-row Top-K and no Top-P."""
    if sorted_scores.device.type != "npu":
        return None
    else:
        pass
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
