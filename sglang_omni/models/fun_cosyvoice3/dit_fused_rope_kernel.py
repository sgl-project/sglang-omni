# SPDX-License-Identifier: Apache-2.0
"""One launch for CosyVoice3's two partial rotary embeddings."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

PROJECTED_CHANNELS = 1024
ROTARY_CHANNELS = 64
WARPS_PER_BLOCK = 4


@triton.jit(do_not_specialize=["sequence_length"])
def partial_qk_rope(
    query_pointer: tl.tensor,
    key_pointer: tl.tensor,
    cosine_pointer: tl.tensor,
    sine_pointer: tl.tensor,
    rotated_query_pointer: tl.tensor,
    rotated_key_pointer: tl.tensor,
    sequence_length: int,
    PROJECTION_WIDTH: tl.constexpr,
    ROTARY_DIMENSION: tl.constexpr,
    CHANNEL_BLOCK_SIZE: tl.constexpr,
) -> None:
    token_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    channel_indices = tl.arange(0, CHANNEL_BLOCK_SIZE)
    projection_offsets = (
        batch_index * sequence_length + token_index
    ) * PROJECTION_WIDTH + channel_indices
    is_rotary_channel = channel_indices < ROTARY_DIMENSION
    cosine = tl.load(
        cosine_pointer + token_index * ROTARY_DIMENSION + channel_indices,
        is_rotary_channel,
        other=0,
    )
    sine = tl.load(
        sine_pointer + token_index * ROTARY_DIMENSION + channel_indices,
        is_rotary_channel,
        other=0,
    )
    # note (wirybeaver): CosyVoice rotates interleaved pairs before splitting heads.
    partner_offsets = (
        batch_index * sequence_length + token_index
    ) * PROJECTION_WIDTH + (channel_indices ^ 1)
    rotation_sign = tl.where(channel_indices % 2 == 0, -1.0, 1.0)
    query = tl.load(
        query_pointer + projection_offsets, channel_indices < PROJECTION_WIDTH, other=0
    ).to(tl.float32)
    key = tl.load(
        key_pointer + projection_offsets, channel_indices < PROJECTION_WIDTH, other=0
    ).to(tl.float32)
    partner_query = tl.load(
        query_pointer + partner_offsets, is_rotary_channel, other=0
    ).to(tl.float32)
    partner_key = tl.load(key_pointer + partner_offsets, is_rotary_channel, other=0).to(
        tl.float32
    )
    rotated_query = query * cosine + (rotation_sign * partner_query) * sine
    rotated_key = key * cosine + (rotation_sign * partner_key) * sine
    tl.store(
        rotated_query_pointer + projection_offsets,
        tl.where(is_rotary_channel, rotated_query, query),
        channel_indices < PROJECTION_WIDTH,
    )
    tl.store(
        rotated_key_pointer + projection_offsets,
        tl.where(is_rotary_channel, rotated_key, key),
        channel_indices < PROJECTION_WIDTH,
    )


def fused_qk_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply partial interleaved RoPE to contiguous Q and K projections."""
    rotated_query = torch.empty_like(query)
    rotated_key = torch.empty_like(key)
    with torch.cuda.device(query.device):
        partial_qk_rope[(query.shape[1], query.shape[0])](
            query,
            key,
            cosine,
            sine,
            rotated_query,
            rotated_key,
            query.shape[1],
            PROJECTION_WIDTH=PROJECTED_CHANNELS,
            ROTARY_DIMENSION=ROTARY_CHANNELS,
            CHANNEL_BLOCK_SIZE=PROJECTED_CHANNELS,
            num_warps=WARPS_PER_BLOCK,
            # note (wirybeaver): Native rotary uses separate multiply and add rounding.
            enable_fp_fusion=False,
        )
    return rotated_query, rotated_key


__all__ = ["fused_qk_rope"]
