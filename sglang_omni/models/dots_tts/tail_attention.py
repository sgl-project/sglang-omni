# SPDX-License-Identifier: Apache-2.0
"""Attention of a short query block over a slot's cached keys plus the block itself."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def cached_block_attention_kernel(
    query_pointer,
    key_pointer,
    value_pointer,
    key_pool_pointer,
    value_pool_pointer,
    slot_pointer,
    valid_pointer,
    output_pointer,
    query_batch_stride,
    query_head_stride,
    query_row_stride,
    key_batch_stride,
    key_head_stride,
    key_row_stride,
    value_batch_stride,
    value_head_stride,
    value_row_stride,
    pool_slot_stride,
    pool_head_stride,
    pool_token_stride,
    output_batch_stride,
    output_head_stride,
    output_row_stride,
    pool_slots,
    pool_tokens,
    scale,
    query_rows: tl.constexpr,
    previous_rows: tl.constexpr,
    block_rows: tl.constexpr,
    block_tokens: tl.constexpr,
    head_dim: tl.constexpr,
) -> None:
    batch = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    slot = tl.load(slot_pointer + batch)
    valid = tl.load(valid_pointer + batch)
    rows = tl.arange(0, block_rows)
    dims = tl.arange(0, head_dim)
    row_mask = rows < query_rows
    query = tl.load(
        query_pointer
        + batch * query_batch_stride
        + head * query_head_stride
        + rows[:, None] * query_row_stride
        + dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    key = tl.load(
        key_pointer
        + batch * key_batch_stride
        + head * key_head_stride
        + rows[:, None] * key_row_stride
        + dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    value = tl.load(
        value_pointer
        + batch * value_batch_stride
        + head * value_head_stride
        + rows[:, None] * value_row_stride
        + dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    maximum = tl.full((block_rows,), float("-inf"), tl.float32)
    total = tl.zeros((block_rows,), tl.float32)
    accumulator = tl.zeros((block_rows, head_dim), tl.float32)
    pool = slot * pool_slot_stride + head * pool_head_stride
    for start in range(0, valid, block_tokens):
        tokens = start + tl.arange(0, block_tokens)
        token_mask = tokens < valid
        cached_key = tl.load(
            key_pool_pointer
            + pool
            + tokens[:, None] * pool_token_stride
            + dims[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(cached_key)) * scale
        scores = tl.where(token_mask[None, :], scores, float("-inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        weights = tl.exp(scores - new_maximum[:, None])
        correction = tl.exp(maximum - new_maximum)
        total = total * correction + tl.sum(weights, 1)
        cached_value = tl.load(
            value_pool_pointer
            + pool
            + tokens[:, None] * pool_token_stride
            + dims[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        accumulator = accumulator * correction[:, None] + tl.dot(
            weights.to(cached_value.dtype), cached_value
        )
        maximum = new_maximum
    # note (0xtoward): rows before previous_rows see the block causally, later rows
    # see all of it; this is batched_causal_update_mask without the cached part.
    scores = tl.dot(query, tl.trans(key)) * scale
    allowed = (rows[None, :] < query_rows) & (
        (rows[:, None] >= previous_rows) | (rows[None, :] <= rows[:, None])
    )
    scores = tl.where(allowed, scores, float("-inf"))
    new_maximum = tl.maximum(maximum, tl.max(scores, 1))
    weights = tl.exp(scores - new_maximum[:, None])
    correction = tl.exp(maximum - new_maximum)
    total = total * correction + tl.sum(weights, 1)
    accumulator = accumulator * correction[:, None] + tl.dot(
        weights.to(value.dtype), value
    )
    output = accumulator / total[:, None]
    tl.store(
        output_pointer
        + batch * output_batch_stride
        + head * output_head_stride
        + rows[:, None] * output_row_stride
        + dims[None, :],
        output.to(output_pointer.dtype.element_ty),
        mask=row_mask[:, None],
    )
    # note (0xtoward): the first previous_rows of the block become cached history;
    # padded rows carry an out-of-range slot and never store.
    keep = (rows < previous_rows) & (slot >= 0) & (slot < pool_slots)
    keep &= valid + rows < pool_tokens
    target = pool + (valid + rows)[:, None] * pool_token_stride + dims[None, :]
    tl.store(key_pool_pointer + target, key, mask=keep[:, None])
    tl.store(value_pool_pointer + target, value, mask=keep[:, None])


@torch.library.custom_op(
    "dots_tts::cached_block_attention", mutates_args=("key_pool", "value_pool")
)
def cached_block_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    slots: torch.Tensor,
    valid: torch.Tensor,
    previous_rows: int,
) -> torch.Tensor:
    """Attend [B, H, M, D] queries over each slot's first valid cached tokens plus the block.

    Returns [B, M, H, D]. Rows before previous_rows are causal within the block and are
    also stored into the pools at the slot's valid position.
    """
    batch, heads, rows, head_dim = query.shape
    output = query.new_empty(batch, rows, heads, head_dim)
    cached_block_attention_kernel[(batch, heads)](
        query,
        key,
        value,
        key_pool,
        value_pool,
        slots,
        valid,
        output,
        *query.stride()[:3],
        *key.stride()[:3],
        *value.stride()[:3],
        *key_pool.stride()[:3],
        output.stride(0),
        output.stride(2),
        output.stride(1),
        key_pool.size(0),
        key_pool.size(2),
        head_dim**-0.5,
        query_rows=rows,
        previous_rows=previous_rows,
        block_rows=max(16, triton.next_power_of_2(rows)),
        block_tokens=64,
        head_dim=head_dim,
        num_warps=4,
    )
    return output


@cached_block_attention.register_fake
def cached_block_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    slots: torch.Tensor,
    valid: torch.Tensor,
    previous_rows: int,
) -> torch.Tensor:
    batch, heads, rows, head_dim = query.shape
    return query.new_empty(batch, rows, heads, head_dim)


__all__ = ["cached_block_attention"]
