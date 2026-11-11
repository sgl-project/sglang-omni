# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of the w2v-BERT teacher layer: relative-key attention and the gated causal depthwise convolution."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def attention_key_block(
    query,
    key_pointers,
    value_pointers,
    key_positions,
    frame_count,
    position_bias,
    running_max,
    running_sum,
    accumulator,
    scaling,
):
    """One key block of the online softmax; position_bias is the block's unscaled bias, or its per-query column."""
    is_key = key_positions < frame_count
    key = tl.load(key_pointers, mask=is_key[None, :], other=0.0)
    scores = (tl.dot(query, key) + position_bias) * scaling
    scores = tl.where(is_key[None, :], scores, float("-inf"))
    block_max = tl.maximum(running_max, tl.max(scores, axis=1))
    probabilities = tl.exp(scores - block_max[:, None])
    rescale = tl.exp(running_max - block_max)
    running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
    value = tl.load(value_pointers, mask=is_key[:, None], other=0.0)
    accumulator = accumulator * rescale[:, None] + tl.dot(
        probabilities.to(value.dtype), value
    )
    return block_max, running_sum, accumulator


@triton.jit
def relative_key_attention_kernel(
    query_key_value_pointer,
    distance_embedding_pointer,
    output_pointer,
    frame_count,
    states_batch_stride,
    states_frame_stride,
    output_batch_stride,
    output_frame_stride,
    scaling,
    HEAD_COUNT: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    LEFT_DISTANCES: tl.constexpr,
    RIGHT_DISTANCES: tl.constexpr,
    DISTANCE_BLOCK: tl.constexpr,
    BLOCK_QUERIES: tl.constexpr,
    BLOCK_KEYS: tl.constexpr,
):
    """Softmax((q k^T + bias) * scaling) v for one (batch, head, query block) of fused (frames, 3 * hidden) projections.

    The bias of query i and key j is the query's bfloat16 dot product with the distance embedding at clamp(j - i, -LEFT, RIGHT); key blocks wholly past either clamp add the query's edge logit, and only the band's blocks gather.
    """
    query_block = tl.program_id(0)
    batch_head_index = tl.program_id(1)
    batch_index = batch_head_index // HEAD_COUNT
    head_index = batch_head_index % HEAD_COUNT
    query_start = query_block * BLOCK_QUERIES
    query_positions = query_start + tl.arange(0, BLOCK_QUERIES)
    dimensions = tl.arange(0, HEAD_SIZE)
    block_keys = tl.arange(0, BLOCK_KEYS)
    distances = tl.arange(0, DISTANCE_BLOCK)
    is_query = query_positions < frame_count
    head_offset = (
        batch_index.to(tl.int64) * states_batch_stride + head_index * HEAD_SIZE
    )
    query = tl.load(
        query_key_value_pointer
        + head_offset
        + query_positions[:, None] * states_frame_stride
        + dimensions[None, :],
        mask=is_query[:, None],
        other=0.0,
    )
    distance_logits = (
        tl.dot(
            query,
            tl.load(
                distance_embedding_pointer
                + distances[None, :] * HEAD_SIZE
                + dimensions[:, None],
                mask=distances[None, :] <= LEFT_DISTANCES + RIGHT_DISTANCES,
                other=0.0,
            ),
        )
        .to(query.dtype)
        .to(tl.float32)
    )
    left_logits = tl.sum(
        tl.where(distances[None, :] == 0, distance_logits, 0.0), axis=1
    )
    right_logits = tl.sum(
        tl.where(
            distances[None, :] == LEFT_DISTANCES + RIGHT_DISTANCES, distance_logits, 0.0
        ),
        axis=1,
    )
    key_pointers = (
        query_key_value_pointer
        + head_offset
        + HEAD_COUNT * HEAD_SIZE
        + block_keys[None, :] * states_frame_stride
        + dimensions[:, None]
    )
    value_pointers = (
        query_key_value_pointer
        + head_offset
        + 2 * HEAD_COUNT * HEAD_SIZE
        + block_keys[:, None] * states_frame_stride
        + dimensions[None, :]
    )
    running_max = tl.full([BLOCK_QUERIES], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_QUERIES], dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_QUERIES, HEAD_SIZE], dtype=tl.float32)
    # Key blocks wholly LEFT or more before every query, the blocks across the band, then those wholly RIGHT or more after.
    band_start = (
        tl.maximum(query_start - LEFT_DISTANCES + 1, 0) // BLOCK_KEYS * BLOCK_KEYS
    )
    band_end = tl.minimum(
        tl.cdiv(query_start + BLOCK_QUERIES - 1 + RIGHT_DISTANCES, BLOCK_KEYS)
        * BLOCK_KEYS,
        tl.cdiv(frame_count, BLOCK_KEYS) * BLOCK_KEYS,
    )
    for key_start in range(0, band_start, BLOCK_KEYS):
        running_max, running_sum, accumulator = attention_key_block(
            query,
            key_pointers + key_start * states_frame_stride,
            value_pointers + key_start * states_frame_stride,
            key_start + block_keys,
            frame_count,
            left_logits[:, None],
            running_max,
            running_sum,
            accumulator,
            scaling,
        )
    for key_start in range(band_start, band_end, BLOCK_KEYS):
        key_positions = key_start + block_keys
        distance_indices = (
            tl.minimum(
                tl.maximum(
                    key_positions[None, :] - query_positions[:, None], -LEFT_DISTANCES
                ),
                RIGHT_DISTANCES,
            )
            + LEFT_DISTANCES
        )
        running_max, running_sum, accumulator = attention_key_block(
            query,
            key_pointers + key_start * states_frame_stride,
            value_pointers + key_start * states_frame_stride,
            key_positions,
            frame_count,
            tl.gather(distance_logits, distance_indices, 1),
            running_max,
            running_sum,
            accumulator,
            scaling,
        )
    for key_start in range(band_end, frame_count, BLOCK_KEYS):
        running_max, running_sum, accumulator = attention_key_block(
            query,
            key_pointers + key_start * states_frame_stride,
            value_pointers + key_start * states_frame_stride,
            key_start + block_keys,
            frame_count,
            right_logits[:, None],
            running_max,
            running_sum,
            accumulator,
            scaling,
        )
    output = accumulator / running_sum[:, None]
    tl.store(
        output_pointer
        + batch_index.to(tl.int64) * output_batch_stride
        + head_index * HEAD_SIZE
        + query_positions[:, None] * output_frame_stride
        + dimensions[None, :],
        output.to(output_pointer.dtype.element_ty),
        mask=is_query[:, None],
    )


@triton.jit
def gated_causal_depthwise_convolution_kernel(
    projected_pointer,
    weight_pointer,
    output_pointer,
    frame_count,
    CHANNELS: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_FRAMES: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    WINDOW_FRAMES: tl.constexpr,
):
    """Causal depthwise convolution of GLU(projected) for one (frame block, channel block, batch).

    The block's GLU window, its frames and the KERNEL_SIZE - 1 before them, is loaded once and rounded to the input dtype; each tap gathers its shifted frames from it, so no tap rereads memory.
    """
    frame_start = tl.program_id(0) * BLOCK_FRAMES
    channels = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    batch_index = tl.program_id(2).to(tl.int64)
    window_rows = tl.arange(0, WINDOW_FRAMES)
    source_frames = frame_start - (KERNEL_SIZE - 1) + window_rows
    is_source = (
        (source_frames >= 0)
        & (source_frames < frame_count)
        & (window_rows < BLOCK_FRAMES + KERNEL_SIZE - 1)
    )
    projected_pointers = (
        projected_pointer
        + batch_index * frame_count * 2 * CHANNELS
        + source_frames[:, None] * 2 * CHANNELS
        + channels[None, :]
    )
    projected_values = tl.load(projected_pointers, mask=is_source[:, None], other=0.0)
    gates = tl.load(
        projected_pointers + CHANNELS, mask=is_source[:, None], other=0.0
    ).to(tl.float32)
    window = (projected_values.to(tl.float32) * (1.0 / (1.0 + tl.exp(-gates)))).to(
        projected_values.dtype
    )
    block_rows = tl.arange(0, BLOCK_FRAMES)
    accumulator = tl.zeros([BLOCK_FRAMES, BLOCK_CHANNELS], dtype=tl.float32)
    for tap in tl.static_range(KERNEL_SIZE):
        tap_weight = tl.load(weight_pointer + channels * KERNEL_SIZE + tap).to(
            tl.float32
        )
        accumulator += (
            tl.gather(
                window,
                tl.broadcast_to(
                    (block_rows + tap)[:, None], (BLOCK_FRAMES, BLOCK_CHANNELS)
                ),
                0,
            ).to(tl.float32)
            * tap_weight[None, :]
        )
    frame_positions = frame_start + block_rows
    tl.store(
        output_pointer
        + batch_index * frame_count * CHANNELS
        + frame_positions[:, None] * CHANNELS
        + channels[None, :],
        accumulator.to(output_pointer.dtype.element_ty),
        mask=(frame_positions < frame_count)[:, None],
    )
