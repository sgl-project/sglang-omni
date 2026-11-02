# SPDX-License-Identifier: Apache-2.0
"""Fuse the attention output projection and gated tower activation."""

from __future__ import annotations

import torch

from sglang_omni.models.fishaudio_drama3.tower_tiles import (
    FUSED_BLOCK_M,
    GATE_UP_BLOCK_N,
    GEMV_BLOCK_K,
    GEMV_WARPS,
    O_PROJ_BLOCK_N,
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
        row_tile,
        tile_sums,
        weight_tile,
    )

    @triton.jit(do_not_specialize=["row_count"])
    def gate_o_proj_add_kernel(
        attended_pointer: tl.tensor,
        attended_row_stride: tl.tensor,
        gate_scale_pointer: tl.tensor,
        weight_pointer: tl.tensor,
        residual_in_pointer: tl.tensor,
        norm_weight_pointer: tl.tensor,
        residual_out_pointer: tl.tensor,
        scaled_out_pointer: tl.tensor,
        partials_pointer: tl.tensor,
        norm_sums_pointer: tl.tensor,
        counters_pointer: tl.tensor,
        row_count: tl.tensor,
        DIM: tl.constexpr,
        QUERY_WIDTH: tl.constexpr,
        SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        output_tile_index = tl.program_id(0)
        reduction_split_index = tl.program_id(1)
        SPLIT_WIDTH: tl.constexpr = QUERY_WIDTH // SPLIT
        row_indices = tl.arange(0, BLOCK_M)
        output_columns = output_tile_index * BLOCK_N + tl.arange(0, BLOCK_N)
        is_row = row_indices < row_count
        product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for first_column_index in range(
            reduction_split_index * SPLIT_WIDTH,
            (reduction_split_index + 1) * SPLIT_WIDTH,
            BLOCK_K,
        ):
            attended = row_tile(
                attended_pointer,
                attended_row_stride,
                is_row,
                first_column_index,
                BLOCK_M,
                BLOCK_K,
            ).to(tl.float32)
            gate_scale = row_tile(
                gate_scale_pointer,
                QUERY_WIDTH,
                is_row,
                first_column_index,
                BLOCK_M,
                BLOCK_K,
            ).to(tl.float32)
            input_states = (attended * gate_scale).to(tl.bfloat16)
            product = tl.dot(
                input_states,
                tl.trans(
                    weight_tile(
                        weight_pointer,
                        output_columns,
                        first_column_index,
                        QUERY_WIDTH,
                        BLOCK_K,
                    )
                ),
                product,
            )
        partial_rows = (reduction_split_index * BLOCK_M + row_indices)[:, None] * DIM
        if SPLIT > 1:
            tl.store(
                partials_pointer + partial_rows + output_columns[None, :],
                product,
                mask=is_row[:, None],
            )
            tl.debug_barrier()
            arrived = tl.atomic_add(
                counters_pointer + output_tile_index, 1, sem="acq_rel", scope="gpu"
            )
            is_last = arrived == SPLIT - 1
        else:
            is_last = reduction_split_index == 0
        if is_last:
            if SPLIT > 1:
                product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                for other_split in tl.static_range(SPLIT):
                    partial_rows = (other_split * BLOCK_M + row_indices)[:, None] * DIM
                    product += tl.load(
                        partials_pointer + partial_rows + output_columns[None, :],
                        mask=is_row[:, None],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                tl.atomic_xchg(counters_pointer + output_tile_index, 0)
            else:
                pass
            residual_columns = row_indices[:, None] * DIM + output_columns[None, :]
            residual = tl.load(
                residual_in_pointer + residual_columns,
                mask=is_row[:, None],
                other=0.0,
            ).to(tl.float32)
            summed = product.to(tl.bfloat16).to(tl.float32) + residual
            gains = 1.0 + tl.load(norm_weight_pointer + output_columns).to(tl.float32)
            tl.store(
                residual_out_pointer + residual_columns,
                summed.to(tl.bfloat16),
                mask=is_row[:, None],
            )
            tl.store(
                scaled_out_pointer + residual_columns,
                (summed * gains[None, :]).to(tl.bfloat16),
                mask=is_row[:, None],
            )
            tl.store(
                norm_sums_pointer + output_tile_index * BLOCK_M + row_indices,
                tl.sum(summed * summed, 1),
            )
        else:
            pass

    @triton.jit(do_not_specialize=["row_count"])
    def norm_gate_up_silu_kernel(
        scaled_pointer: tl.tensor,
        norm_sums_pointer: tl.tensor,
        weight_pointer: tl.tensor,
        activated_pointer: tl.tensor,
        partials_pointer: tl.tensor,
        counters_pointer: tl.tensor,
        row_count: tl.tensor,
        norm_eps: tl.tensor,
        DIM: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        NORM_TILES: tl.constexpr,
        NORM_TILE_BLOCK: tl.constexpr,
        SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        output_tile_index = tl.program_id(0)
        reduction_split_index = tl.program_id(1)
        SPLIT_WIDTH: tl.constexpr = DIM // SPLIT
        row_indices = tl.arange(0, BLOCK_M)
        gate_rows = output_tile_index * BLOCK_N + tl.arange(0, BLOCK_N)
        up_rows = INTERMEDIATE + gate_rows
        is_row = row_indices < row_count
        gate_product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        up_product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for first_column_index in range(
            reduction_split_index * SPLIT_WIDTH,
            (reduction_split_index + 1) * SPLIT_WIDTH,
            BLOCK_K,
        ):
            input_states = row_tile(
                scaled_pointer, DIM, is_row, first_column_index, BLOCK_M, BLOCK_K
            )
            gate_product = tl.dot(
                input_states,
                tl.trans(
                    weight_tile(
                        weight_pointer, gate_rows, first_column_index, DIM, BLOCK_K
                    )
                ),
                gate_product,
            )
            up_product = tl.dot(
                input_states,
                tl.trans(
                    weight_tile(
                        weight_pointer, up_rows, first_column_index, DIM, BLOCK_K
                    )
                ),
                up_product,
            )
        partial_rows = (reduction_split_index * BLOCK_M + row_indices)[:, None] * (
            2 * INTERMEDIATE
        )
        if SPLIT > 1:
            tl.store(
                partials_pointer + partial_rows + gate_rows[None, :],
                gate_product,
                mask=is_row[:, None],
            )
            tl.store(
                partials_pointer + partial_rows + up_rows[None, :],
                up_product,
                mask=is_row[:, None],
            )
            tl.debug_barrier()
            arrived = tl.atomic_add(
                counters_pointer + output_tile_index, 1, sem="acq_rel", scope="gpu"
            )
            is_last = arrived == SPLIT - 1
        else:
            is_last = reduction_split_index == 0
        if is_last:
            if SPLIT > 1:
                gate_product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                up_product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                for other_split in tl.static_range(SPLIT):
                    partial_rows = (other_split * BLOCK_M + row_indices)[:, None] * (
                        2 * INTERMEDIATE
                    )
                    gate_product += tl.load(
                        partials_pointer + partial_rows + gate_rows[None, :],
                        mask=is_row[:, None],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                    up_product += tl.load(
                        partials_pointer + partial_rows + up_rows[None, :],
                        mask=is_row[:, None],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                tl.atomic_xchg(counters_pointer + output_tile_index, 0)
            else:
                pass
            rstd = libdevice.rsqrt(
                tile_sums(norm_sums_pointer, NORM_TILES, NORM_TILE_BLOCK, BLOCK_M) / DIM
                + norm_eps
            )
            gate = (gate_product * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
            up = (up_product * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
            activated = tl.math.div_rn(gate, 1.0 + libdevice.exp(-gate)) * up
            tl.store(
                activated_pointer
                + row_indices[:, None] * INTERMEDIATE
                + gate_rows[None, :],
                activated.to(tl.bfloat16),
                mask=is_row[:, None],
            )
        else:
            pass

else:
    pass


def gate_o_proj_add(
    *,
    weights: TowerLayerWeights,
    attended: torch.Tensor,
    gate_scale: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    scaled_out: torch.Tensor,
    scratch: SplitScratch,
    shape: TowerLayerShape,
) -> None:
    gate_o_proj_add_kernel[(shape.dim // O_PROJ_BLOCK_N, shape.o_proj_split)](
        attended,
        attended.stride(0),
        gate_scale,
        weights.o_proj.weight,
        residual_in,
        weights.post_norm.weight,
        residual_out,
        scaled_out,
        scratch.partials,
        scratch.norm_sums,
        scratch.counters,
        residual_out.shape[0],
        DIM=shape.dim,
        QUERY_WIDTH=shape.query_width,
        SPLIT=shape.o_proj_split,
        BLOCK_M=FUSED_BLOCK_M,
        BLOCK_N=O_PROJ_BLOCK_N,
        BLOCK_K=GEMV_BLOCK_K,
        num_warps=GEMV_WARPS,
        num_stages=PIPELINE_STAGES,
    )


def norm_gate_up_silu(
    *,
    weights: TowerLayerWeights,
    scaled: torch.Tensor,
    activated: torch.Tensor,
    scratch: SplitScratch,
    shape: TowerLayerShape,
) -> None:
    norm_tiles = shape.dim // O_PROJ_BLOCK_N
    norm_gate_up_silu_kernel[
        (shape.intermediate_size // GATE_UP_BLOCK_N, shape.gate_up_split)
    ](
        scaled,
        scratch.norm_sums,
        weights.gate_up_proj.weight,
        activated,
        scratch.partials,
        scratch.counters,
        scaled.shape[0],
        weights.post_norm.eps,
        DIM=shape.dim,
        INTERMEDIATE=shape.intermediate_size,
        NORM_TILES=norm_tiles,
        NORM_TILE_BLOCK=triton.next_power_of_2(norm_tiles),
        SPLIT=shape.gate_up_split,
        BLOCK_M=FUSED_BLOCK_M,
        BLOCK_N=GATE_UP_BLOCK_N,
        BLOCK_K=GEMV_BLOCK_K,
        num_warps=GEMV_WARPS,
        num_stages=PIPELINE_STAGES,
    )
