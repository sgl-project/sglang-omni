# SPDX-License-Identifier: Apache-2.0
"""Fuse tower QKV projection, rotary embeddings and paged cache writes."""

from __future__ import annotations

import torch

from sglang_omni.models.fishaudio_drama3.tower_tiles import (
    FUSED_BLOCK_M,
    MIX_BLOCK_N,
    PIPELINE_STAGES,
    QKV_BLOCK_K,
    QKV_WARPS,
    SplitScratch,
    TowerLayerShape,
    TowerLayerWeights,
)

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    libdevice = None
    HAS_TRITON = False

if HAS_TRITON:

    from sglang_omni.models.fishaudio_drama3.tower_tiles import (
        normed_rotated,
        row_tile,
        sigmoid,
        tile_sums,
        weight_tile,
    )

    @triton.jit(do_not_specialize=["row_count"])
    def norm_qkv_rope_store_kernel(
        scaled_pointer: tl.tensor,
        norm_sums_pointer: tl.tensor,
        weight_pointer: tl.tensor,
        q_norm_weight_pointer: tl.tensor,
        k_norm_weight_pointer: tl.tensor,
        cos_sin_pointer: tl.tensor,
        positions_pointer: tl.tensor,
        cache_slots_pointer: tl.tensor,
        query_pointer: tl.tensor,
        gate_scale_pointer: tl.tensor,
        key_cache_pointer: tl.tensor,
        value_cache_pointer: tl.tensor,
        cache_slot_stride: tl.tensor,
        cache_head_stride: tl.tensor,
        partials_pointer: tl.tensor,
        counters_pointer: tl.tensor,
        row_count: tl.tensor,
        input_norm_eps: tl.tensor,
        q_norm_eps: tl.tensor,
        k_norm_eps: tl.tensor,
        DIM: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        HEAD_COUNT: tl.constexpr,
        KV_HEAD_COUNT: tl.constexpr,
        NORM_TILES: tl.constexpr,
        NORM_TILE_BLOCK: tl.constexpr,
        SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        projection_part_index = tl.program_id(0)
        reduction_split_index = tl.program_id(1)
        PAIRS: tl.constexpr = HEAD_DIM // 2
        PART_COUNT: tl.constexpr = 2 * (HEAD_COUNT + KV_HEAD_COUNT)
        QUERY_WIDTH: tl.constexpr = HEAD_COUNT * HEAD_DIM
        SPLIT_WIDTH: tl.constexpr = DIM // SPLIT
        row_indices = tl.arange(0, BLOCK_M)
        rotary_pair_indices = tl.arange(0, PAIRS)
        is_row = row_indices < row_count
        if projection_part_index < HEAD_COUNT:
            first_weight_row = 2 * projection_part_index * HEAD_DIM
        elif projection_part_index < 2 * HEAD_COUNT:
            first_weight_row = (2 * (projection_part_index - HEAD_COUNT) + 1) * HEAD_DIM
        else:
            first_weight_row = projection_part_index * HEAD_DIM
        even_weight_rows = first_weight_row + 2 * rotary_pair_indices
        odd_weight_rows = even_weight_rows + 1
        even_product = tl.zeros([BLOCK_M, PAIRS], dtype=tl.float32)
        odd_product = tl.zeros([BLOCK_M, PAIRS], dtype=tl.float32)
        for first_column_index in range(
            reduction_split_index * SPLIT_WIDTH,
            (reduction_split_index + 1) * SPLIT_WIDTH,
            BLOCK_K,
        ):
            scaled = row_tile(
                scaled_pointer, DIM, is_row, first_column_index, BLOCK_M, BLOCK_K
            )
            even_product = tl.dot(
                scaled,
                tl.trans(
                    weight_tile(
                        weight_pointer,
                        even_weight_rows,
                        first_column_index,
                        DIM,
                        BLOCK_K,
                    )
                ),
                even_product,
            )
            odd_product = tl.dot(
                scaled,
                tl.trans(
                    weight_tile(
                        weight_pointer,
                        odd_weight_rows,
                        first_column_index,
                        DIM,
                        BLOCK_K,
                    )
                ),
                odd_product,
            )
        partial_rows = (
            (reduction_split_index * BLOCK_M + row_indices) * PART_COUNT
            + projection_part_index
        ) * HEAD_DIM
        tl.store(
            partials_pointer + partial_rows[:, None] + rotary_pair_indices[None, :],
            even_product,
            mask=is_row[:, None],
        )
        tl.store(
            partials_pointer
            + partial_rows[:, None]
            + PAIRS
            + rotary_pair_indices[None, :],
            odd_product,
            mask=is_row[:, None],
        )
        tl.debug_barrier()
        arrived = tl.atomic_add(
            counters_pointer + projection_part_index, 1, sem="acq_rel", scope="gpu"
        )
        if arrived == SPLIT - 1:
            even = tl.zeros([BLOCK_M, PAIRS], dtype=tl.float32)
            odd = tl.zeros([BLOCK_M, PAIRS], dtype=tl.float32)
            for other_split in tl.static_range(SPLIT):
                partial_rows = (
                    (other_split * BLOCK_M + row_indices) * PART_COUNT
                    + projection_part_index
                ) * HEAD_DIM
                even += tl.load(
                    partials_pointer
                    + partial_rows[:, None]
                    + rotary_pair_indices[None, :],
                    mask=is_row[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )
                odd += tl.load(
                    partials_pointer
                    + partial_rows[:, None]
                    + PAIRS
                    + rotary_pair_indices[None, :],
                    mask=is_row[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )
            tl.atomic_xchg(counters_pointer + projection_part_index, 0)
            input_rstd = libdevice.rsqrt(
                tile_sums(norm_sums_pointer, NORM_TILES, NORM_TILE_BLOCK, BLOCK_M) / DIM
                + input_norm_eps
            )
            even = (even * input_rstd[:, None]).to(tl.bfloat16).to(tl.float32)
            odd = (odd * input_rstd[:, None]).to(tl.bfloat16).to(tl.float32)
            positions = tl.load(positions_pointer + row_indices, mask=is_row, other=0)
            cache_slots = tl.load(
                cache_slots_pointer + row_indices, mask=is_row, other=0
            )
            is_cached_row = is_row & (cache_slots != 0)
            cache_rows = (
                cache_slots[:, None] * cache_slot_stride
                + 2 * rotary_pair_indices[None, :]
            )
            if projection_part_index < HEAD_COUNT:
                query_even, query_odd = normed_rotated(
                    even,
                    odd,
                    q_norm_weight_pointer,
                    q_norm_eps,
                    cos_sin_pointer,
                    positions,
                    is_row,
                    HEAD_DIM,
                    ROTARY_DIM,
                )
                query_columns = (
                    row_indices[:, None] * QUERY_WIDTH
                    + projection_part_index * HEAD_DIM
                    + 2 * rotary_pair_indices[None, :]
                )
                tl.store(
                    query_pointer + query_columns, query_even, mask=is_row[:, None]
                )
                tl.store(
                    query_pointer + query_columns + 1, query_odd, mask=is_row[:, None]
                )
            elif projection_part_index < 2 * HEAD_COUNT:
                gate_columns = (
                    row_indices[:, None] * QUERY_WIDTH
                    + (projection_part_index - HEAD_COUNT) * HEAD_DIM
                    + 2 * rotary_pair_indices[None, :]
                )
                tl.store(
                    gate_scale_pointer + gate_columns,
                    sigmoid(even).to(tl.bfloat16),
                    mask=is_row[:, None],
                )
                tl.store(
                    gate_scale_pointer + gate_columns + 1,
                    sigmoid(odd).to(tl.bfloat16),
                    mask=is_row[:, None],
                )
            elif projection_part_index < 2 * HEAD_COUNT + KV_HEAD_COUNT:
                key_even, key_odd = normed_rotated(
                    even,
                    odd,
                    k_norm_weight_pointer,
                    k_norm_eps,
                    cos_sin_pointer,
                    positions,
                    is_row,
                    HEAD_DIM,
                    ROTARY_DIM,
                )
                key_columns = (
                    cache_rows
                    + (projection_part_index - 2 * HEAD_COUNT) * cache_head_stride
                )
                tl.store(
                    key_cache_pointer + key_columns,
                    key_even,
                    mask=is_cached_row[:, None],
                )
                tl.store(
                    key_cache_pointer + key_columns + 1,
                    key_odd,
                    mask=is_cached_row[:, None],
                )
            else:
                value_columns = (
                    cache_rows
                    + (projection_part_index - 2 * HEAD_COUNT - KV_HEAD_COUNT)
                    * cache_head_stride
                )
                tl.store(
                    value_cache_pointer + value_columns,
                    even.to(tl.bfloat16),
                    mask=is_cached_row[:, None],
                )
                tl.store(
                    value_cache_pointer + value_columns + 1,
                    odd.to(tl.bfloat16),
                    mask=is_cached_row[:, None],
                )
        else:
            pass

else:
    pass


def norm_qkv_rope_store(
    *,
    weights: TowerLayerWeights,
    scaled: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    positions: torch.Tensor,
    cache_slots: torch.Tensor,
    query: torch.Tensor,
    gate_scale: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scratch: SplitScratch,
    shape: TowerLayerShape,
) -> None:
    norm_tiles = shape.dim // MIX_BLOCK_N
    norm_qkv_rope_store_kernel[(shape.qkv_part_count, shape.qkv_split)](
        scaled,
        scratch.norm_sums,
        weights.qkv_proj.weight,
        weights.q_norm.weight,
        weights.k_norm.weight,
        rotary_cos_sin,
        positions,
        cache_slots,
        query,
        gate_scale,
        key_cache,
        value_cache,
        key_cache.stride(0),
        key_cache.stride(1),
        scratch.partials,
        scratch.counters,
        scaled.shape[0],
        weights.input_norm.eps,
        weights.q_norm.eps,
        weights.k_norm.eps,
        DIM=shape.dim,
        HEAD_DIM=shape.head_dim,
        ROTARY_DIM=shape.rotary_dim,
        HEAD_COUNT=shape.head_count,
        KV_HEAD_COUNT=shape.kv_head_count,
        NORM_TILES=norm_tiles,
        NORM_TILE_BLOCK=triton.next_power_of_2(norm_tiles),
        SPLIT=shape.qkv_split,
        BLOCK_M=FUSED_BLOCK_M,
        BLOCK_K=QKV_BLOCK_K,
        num_warps=QKV_WARPS,
        num_stages=PIPELINE_STAGES,
    )
