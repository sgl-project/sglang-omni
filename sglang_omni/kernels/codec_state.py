# SPDX-License-Identifier: Apache-2.0
"""In-place exchanges of per-slot codec state: a causal conv's history in front of its new samples,
and a transposed conv's overlap added into its output."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

CHANNEL_BLOCK = 64
NO_PAD_SLOT = -1


@triton.jit
def history_exchange_kernel(
    padded_ptr,
    history_pool_ptr,
    slot_index_ptr,
    history_samples,
    new_samples,
    tail_samples,
    channels,
    padded_row_stride,
    padded_sample_stride,
    padded_channel_stride,
    pool_slot_stride,
    pool_sample_stride,
    pool_channel_stride,
    pad_slot,
    IS_COLD: tl.constexpr,
    IS_SHIFT: tl.constexpr,
    HISTORY_BLOCK: tl.constexpr,
    TAIL_BLOCK: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr,
):
    # note (ratish): a program owns a row's channel block, so it reads the old history before
    # writing the new; padding rows share one slot across programs, so they skip the pool.
    row = tl.program_id(0).to(tl.int64)
    channel = tl.program_id(1) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
    channel_mask = channel < channels
    slot = tl.load(slot_index_ptr + row).to(tl.int64)
    sample = tl.arange(0, HISTORY_BLOCK)
    history_mask = (sample < history_samples)[:, None] & channel_mask[None, :]
    pool_mask = history_mask & (slot != pad_slot)
    padded_base = (
        padded_ptr + row * padded_row_stride + channel[None, :] * padded_channel_stride
    )
    pool_base = (
        history_pool_ptr
        + slot * pool_slot_stride
        + channel[None, :] * pool_channel_stride
    )
    source = sample + new_samples
    from_new = (source >= history_samples)[:, None] & history_mask
    newest = tl.load(
        padded_base + source[:, None] * padded_sample_stride, mask=from_new, other=0
    )
    if IS_COLD:
        old = tl.zeros(
            [HISTORY_BLOCK, CHANNEL_BLOCK], dtype=padded_ptr.dtype.element_ty
        )
        new_history = newest
    else:
        old = tl.load(
            pool_base + sample[:, None] * pool_sample_stride,
            mask=pool_mask,
            other=0,
        )
        if IS_SHIFT:
            kept = tl.load(
                pool_base + source[:, None] * pool_sample_stride,
                mask=pool_mask & (source < history_samples)[:, None],
                other=0,
            )
            new_history = tl.where(from_new, newest, kept)
            # note (ratish): with fewer new samples than the history, a kept sample is
            # read by one thread and overwritten by another.
            tl.debug_barrier()
        else:
            new_history = newest
    tl.store(
        padded_base + sample[:, None] * padded_sample_stride, old, mask=history_mask
    )
    tl.store(
        pool_base + sample[:, None] * pool_sample_stride,
        new_history,
        mask=pool_mask,
    )
    tail = tl.arange(0, TAIL_BLOCK)
    tl.store(
        padded_base
        + (history_samples + new_samples + tail)[:, None] * padded_sample_stride,
        tl.zeros([TAIL_BLOCK, CHANNEL_BLOCK], dtype=padded_ptr.dtype.element_ty),
        mask=(tail < tail_samples)[:, None] & channel_mask[None, :],
    )


@triton.jit
def overlap_exchange_kernel(
    output_ptr,
    overlap_pool_ptr,
    slot_index_ptr,
    overlap_samples,
    emitted_samples,
    channels,
    output_row_stride,
    output_sample_stride,
    output_channel_stride,
    pool_slot_stride,
    pool_sample_stride,
    pool_channel_stride,
    pad_slot,
    IS_COLD: tl.constexpr,
    OVERLAP_BLOCK: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    channel = tl.program_id(1) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
    channel_mask = channel < channels
    slot = tl.load(slot_index_ptr + row).to(tl.int64)
    sample = tl.arange(0, OVERLAP_BLOCK)
    mask = (sample < overlap_samples)[:, None] & channel_mask[None, :]
    pool_mask = mask & (slot != pad_slot)
    output_base = (
        output_ptr + row * output_row_stride + channel[None, :] * output_channel_stride
    )
    pool_base = (
        overlap_pool_ptr
        + slot * pool_slot_stride
        + channel[None, :] * pool_channel_stride
    )
    tail = tl.load(
        output_base + (emitted_samples + sample)[:, None] * output_sample_stride,
        mask=mask,
        other=0,
    )
    if IS_COLD:
        pass
    else:
        previous = tl.load(
            pool_base + sample[:, None] * pool_sample_stride, mask=pool_mask, other=0
        )
        head = tl.load(
            output_base + sample[:, None] * output_sample_stride, mask=mask, other=0
        )
        tl.store(
            output_base + sample[:, None] * output_sample_stride,
            (head.to(tl.float32) + previous.to(tl.float32)).to(head.dtype),
            mask=mask,
        )
    tl.store(pool_base + sample[:, None] * pool_sample_stride, tail, mask=pool_mask)


def history_exchange(
    padded: torch.Tensor,
    history_pool: torch.Tensor,
    slot_index: torch.Tensor,
    new_samples: int,
    is_cold: bool,
    pad_slot: int = NO_PAD_SLOT,
) -> None:
    """Fill a causal conv's (rows, history + new + tail, channels) input around the new samples
    a producer already wrote: the slot's history in front (zeros for a cold slot), the dilation's
    zeros after; then keep the last history samples of history and new samples as the slot's
    history. history_pool is (slots, history, channels); slot_index names each row's slot, and
    rows naming pad_slot take zeros in front and leave the pool unwritten.
    """
    rows, padded_samples, channels = padded.shape
    history_samples = int(history_pool.shape[1])
    tail_samples = padded_samples - history_samples - new_samples
    history_exchange_kernel[(rows, triton.cdiv(channels, CHANNEL_BLOCK))](
        padded,
        history_pool,
        slot_index,
        history_samples,
        new_samples,
        tail_samples,
        channels,
        *padded.stride(),
        *history_pool.stride(),
        pad_slot,
        IS_COLD=is_cold,
        IS_SHIFT=new_samples < history_samples,
        HISTORY_BLOCK=triton.next_power_of_2(max(history_samples, 1)),
        TAIL_BLOCK=triton.next_power_of_2(max(tail_samples, 1)),
        CHANNEL_BLOCK=CHANNEL_BLOCK,
        num_warps=4,
    )


def overlap_exchange(
    output: torch.Tensor,
    overlap_pool: torch.Tensor,
    slot_index: torch.Tensor,
    emitted_samples: int,
    is_cold: bool,
    pad_slot: int = NO_PAD_SLOT,
) -> None:
    """Add each slot's overlap into the front of a transposed conv's (rows, emitted + overlap,
    channels) output (not for a cold slot) and keep the samples past the emitted ones as the
    slot's overlap; rows naming pad_slot add nothing and leave the pool unwritten. Needs
    emitted_samples >= the overlap, so the two ranges never meet.
    """
    rows, _, channels = output.shape
    overlap_samples = int(overlap_pool.shape[1])
    overlap_exchange_kernel[(rows, triton.cdiv(channels, CHANNEL_BLOCK))](
        output,
        overlap_pool,
        slot_index,
        overlap_samples,
        emitted_samples,
        channels,
        *output.stride(),
        *overlap_pool.stride(),
        pad_slot,
        IS_COLD=is_cold,
        OVERLAP_BLOCK=triton.next_power_of_2(max(overlap_samples, 1)),
        CHANNEL_BLOCK=CHANNEL_BLOCK,
        num_warps=4,
    )
