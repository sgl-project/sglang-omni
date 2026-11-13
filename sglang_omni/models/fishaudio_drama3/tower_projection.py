# SPDX-License-Identifier: Apache-2.0
"""Fuse code embeddings, normalization and the hidden-state projection."""

from __future__ import annotations

import torch

from sglang_omni.models.fishaudio_drama3.tower_tiles import (
    FUSED_BLOCK_M,
    GEMV_BLOCK_K,
    GEMV_WARPS,
    MIX_BLOCK_N,
    PIPELINE_STAGES,
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
        gemma_scaled,
        row_tile,
        weight_tile,
    )

    @triton.jit(do_not_specialize=["row_count"])
    def mix_eh_proj_kernel(
        code_ids_pointer: tl.tensor,
        label_is_semantic_pointer: tl.tensor,
        token_embedding_pointer: tl.tensor,
        shared_input_pointer: tl.tensor,
        hidden_states_pointer: tl.tensor,
        enorm_weight_pointer: tl.tensor,
        hnorm_weight_pointer: tl.tensor,
        weight_pointer: tl.tensor,
        input_norm_weight_pointer: tl.tensor,
        mixed_pointer: tl.tensor,
        scaled_pointer: tl.tensor,
        partials_pointer: tl.tensor,
        sum_of_squares_pointer: tl.tensor,
        norm_sums_pointer: tl.tensor,
        counters_pointer: tl.tensor,
        row_count: tl.tensor,
        enorm_eps: tl.tensor,
        hnorm_eps: tl.tensor,
        CODEBOOK_SIZE: tl.constexpr,
        DIM: tl.constexpr,
        HAS_SHARED_INPUT: tl.constexpr,
        HALF_SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        output_tile_index = tl.program_id(0)
        reduction_split_index = tl.program_id(1)
        SPLIT: tl.constexpr = 2 * HALF_SPLIT
        SPLIT_WIDTH: tl.constexpr = DIM // HALF_SPLIT
        row_indices = tl.arange(0, BLOCK_M)
        block_columns = tl.arange(0, BLOCK_K)
        output_columns = output_tile_index * BLOCK_N + tl.arange(0, BLOCK_N)
        is_row = row_indices < row_count
        first_column = reduction_split_index % HALF_SPLIT * SPLIT_WIDTH
        square = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)
        product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if reduction_split_index < HALF_SPLIT:
            code_ids = tl.load(code_ids_pointer + row_indices, mask=is_row, other=0)
            table_rows = tl.minimum(tl.maximum(code_ids, 0), CODEBOOK_SIZE - 1)
            is_semantic = (
                tl.load(label_is_semantic_pointer + row_indices, mask=is_row, other=0)
                != 0
            )
            for first_column_index in range(
                first_column, first_column + SPLIT_WIDTH, BLOCK_K
            ):
                embeddings = tl.load(
                    token_embedding_pointer
                    + table_rows[:, None] * DIM
                    + (first_column_index + block_columns)[None, :],
                    mask=(is_row & is_semantic)[:, None],
                    other=0.0,
                ).to(tl.float32)
                if HAS_SHARED_INPUT:
                    embeddings += row_tile(
                        shared_input_pointer,
                        DIM,
                        is_row,
                        first_column_index,
                        BLOCK_M,
                        BLOCK_K,
                    ).to(tl.float32)
                else:
                    pass
                input_states = embeddings.to(tl.bfloat16)
                square = tl.dot(input_states, tl.trans(input_states), square)
                product = tl.dot(
                    gemma_scaled(
                        input_states, enorm_weight_pointer, first_column_index, BLOCK_K
                    ),
                    tl.trans(
                        weight_tile(
                            weight_pointer,
                            output_columns,
                            first_column_index,
                            2 * DIM,
                            BLOCK_K,
                        )
                    ),
                    product,
                )
        else:
            for first_column_index in range(
                first_column, first_column + SPLIT_WIDTH, BLOCK_K
            ):
                input_states = row_tile(
                    hidden_states_pointer,
                    DIM,
                    is_row,
                    first_column_index,
                    BLOCK_M,
                    BLOCK_K,
                )
                square = tl.dot(input_states, tl.trans(input_states), square)
                product = tl.dot(
                    gemma_scaled(
                        input_states, hnorm_weight_pointer, first_column_index, BLOCK_K
                    ),
                    tl.trans(
                        weight_tile(
                            weight_pointer,
                            output_columns,
                            DIM + first_column_index,
                            2 * DIM,
                            BLOCK_K,
                        )
                    ),
                    product,
                )
        partial_rows = (reduction_split_index * BLOCK_M + row_indices)[:, None] * DIM
        tl.store(
            partials_pointer + partial_rows + output_columns[None, :],
            product,
            mask=is_row[:, None],
        )
        tl.store(
            sum_of_squares_pointer
            + (output_tile_index * SPLIT + reduction_split_index) * BLOCK_M
            + row_indices,
            tl.sum(
                tl.where(row_indices[:, None] == row_indices[None, :], square, 0.0), 1
            ),
        )
        tl.debug_barrier()
        arrived = tl.atomic_add(
            counters_pointer + output_tile_index, 1, sem="acq_rel", scope="gpu"
        )
        if arrived == SPLIT - 1:
            embedding_product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            hidden_product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            embedding_sum_of_squares = tl.zeros([BLOCK_M], dtype=tl.float32)
            hidden_sum_of_squares = tl.zeros([BLOCK_M], dtype=tl.float32)
            for projection_part_index in tl.static_range(SPLIT):
                partial_rows = (projection_part_index * BLOCK_M + row_indices)[
                    :, None
                ] * DIM
                partial = tl.load(
                    partials_pointer + partial_rows + output_columns[None, :],
                    mask=is_row[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )
                partial_sum_of_squares = tl.load(
                    sum_of_squares_pointer
                    + (output_tile_index * SPLIT + projection_part_index) * BLOCK_M
                    + row_indices,
                    cache_modifier=".cg",
                )
                if projection_part_index < HALF_SPLIT:
                    embedding_product += partial
                    embedding_sum_of_squares += partial_sum_of_squares
                else:
                    hidden_product += partial
                    hidden_sum_of_squares += partial_sum_of_squares
            tl.atomic_xchg(counters_pointer + output_tile_index, 0)
            embedding_rstd = libdevice.rsqrt(embedding_sum_of_squares / DIM + enorm_eps)
            hidden_rstd = libdevice.rsqrt(hidden_sum_of_squares / DIM + hnorm_eps)
            mixed = (
                embedding_product * embedding_rstd[:, None]
                + hidden_product * hidden_rstd[:, None]
            ).to(tl.bfloat16)
            row_columns = row_indices[:, None] * DIM + output_columns[None, :]
            tl.store(mixed_pointer + row_columns, mixed, mask=is_row[:, None])
            mixed = mixed.to(tl.float32)
            gains = 1.0 + tl.load(input_norm_weight_pointer + output_columns).to(
                tl.float32
            )
            tl.store(
                scaled_pointer + row_columns,
                (mixed * gains[None, :]).to(tl.bfloat16),
                mask=is_row[:, None],
            )
            tl.store(
                norm_sums_pointer + output_tile_index * BLOCK_M + row_indices,
                tl.sum(mixed * mixed, 1),
            )
        else:
            pass

else:
    pass


def mix_eh_proj(
    *,
    weights: TowerLayerWeights,
    hidden_states: torch.Tensor,
    code_ids: torch.Tensor,
    label_is_semantic: torch.Tensor,
    shared_input: torch.Tensor | None,
    mixed: torch.Tensor,
    scaled: torch.Tensor,
    scratch: SplitScratch,
    shape: TowerLayerShape,
) -> None:
    row_count = hidden_states.shape[0]
    mix_eh_proj_kernel[(shape.dim // MIX_BLOCK_N, 2 * shape.mix_half_split)](
        code_ids,
        label_is_semantic,
        weights.token_embedding,
        hidden_states if shared_input is None else shared_input,
        hidden_states,
        weights.enorm.weight,
        weights.hnorm.weight,
        weights.eh_proj.weight,
        weights.input_norm.weight,
        mixed,
        scaled,
        scratch.partials,
        scratch.sum_of_squares,
        scratch.norm_sums,
        scratch.counters,
        row_count,
        weights.enorm.eps,
        weights.hnorm.eps,
        CODEBOOK_SIZE=shape.codebook_size,
        DIM=shape.dim,
        HAS_SHARED_INPUT=shared_input is not None,
        HALF_SPLIT=shape.mix_half_split,
        BLOCK_M=FUSED_BLOCK_M,
        BLOCK_N=MIX_BLOCK_N,
        BLOCK_K=GEMV_BLOCK_K,
        num_warps=GEMV_WARPS,
        num_stages=PIPELINE_STAGES,
    )
