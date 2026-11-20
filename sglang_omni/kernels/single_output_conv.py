# SPDX-License-Identifier: Apache-2.0
"""A stride-one, undilated conv from channels-last activations to one output channel, its
bias added and the result clamped to [-1, 1], accumulated in fp32 and rounded once."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

VECTOR_BYTES = 16
THREADS_PER_PROGRAM = 256


@triton.jit(do_not_specialize=["length"])
def clamped_single_output_conv_kernel(
    padded_ptr,
    padded_batch_stride,
    weight_ptr,
    weight_channel_stride,
    weight_tap_stride,
    bias_ptr,
    out_ptr,
    length,
    CHANNELS: tl.constexpr,
    TAPS: tl.constexpr,
    BLOCK_POSITIONS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    # note (ratish): each tap reloads its shifted tile and the products accumulate per channel,
    # summed once after the last tap, so every position takes one order whatever its tile.
    batch = tl.program_id(1).to(tl.int64)
    positions = tl.program_id(0) * BLOCK_POSITIONS + tl.arange(0, BLOCK_POSITIONS)
    channels = tl.arange(0, BLOCK_CHANNELS)
    is_position = positions < length
    padded_row = padded_ptr + batch * padded_batch_stride
    products = tl.zeros((BLOCK_POSITIONS, BLOCK_CHANNELS), dtype=tl.float32)
    for tap in tl.static_range(TAPS):
        for channel_start in tl.static_range(0, CHANNELS, BLOCK_CHANNELS):
            values = tl.load(
                padded_row
                + (positions + tap)[:, None] * CHANNELS
                + (channel_start + channels)[None, :],
                mask=is_position[:, None],
                other=0,
            ).to(tl.float32)
            weights = tl.load(
                weight_ptr
                + (channel_start + channels) * weight_channel_stride
                + tap * weight_tap_stride
            ).to(tl.float32)
            products += values * weights[None, :]
    summed = tl.sum(products, axis=1) + tl.load(bias_ptr).to(tl.float32)
    clamped = tl.clamp(summed, -1.0, 1.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(
        out_ptr + batch * length + positions,
        clamped.to(out_ptr.dtype.element_ty),
        mask=is_position,
    )


def clamped_single_output_conv(
    padded: torch.Tensor, conv: torch.nn.Conv1d, length: int
) -> torch.Tensor:
    """The (B, 1, length) outputs of a stride-one, undilated, ungrouped conv to one output
    channel over a (B, length + kernel - 1, C) activation that carries its left context, its
    bias added and clamped to [-1, 1]."""
    batch_size, padded_length, channels = padded.shape
    taps = conv.kernel_size[0]
    assert (
        conv.out_channels == 1
        and conv.stride == (1,)
        and conv.dilation == (1,)
        and conv.groups == 1
    ), "a stride-one, undilated, ungrouped conv to one output channel"
    assert padded_length == length + taps - 1, "the input carries the left context"
    assert padded.stride(2) == 1 and padded.stride(1) == channels, "contiguous rows"
    output = padded.new_empty(batch_size, 1, length)
    # note (ratish): a channel tile is the largest power of two dividing the channels, so no lane
    # idles, and a program holds one 16-byte vector of every tap's tile per thread.
    block_channels = channels & -channels
    block_positions = (
        THREADS_PER_PROGRAM * (VECTOR_BYTES // padded.element_size()) // block_channels
    )
    clamped_single_output_conv_kernel[
        (triton.cdiv(length, block_positions), batch_size)
    ](
        padded,
        padded.stride(0),
        conv.weight,
        conv.weight.stride(1),
        conv.weight.stride(2),
        conv.bias,
        output,
        length,
        CHANNELS=channels,
        TAPS=taps,
        BLOCK_POSITIONS=block_positions,
        BLOCK_CHANNELS=block_channels,
        num_warps=THREADS_PER_PROGRAM // 32,
    )
    return output
