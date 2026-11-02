# SPDX-License-Identifier: Apache-2.0
"""Shared split-reduction tiles, weight views and launch geometry for tower kernels."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

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

MAX_FUSED_ROWS = 16

FUSED_BLOCK_M = MAX_FUSED_ROWS

MIX_BLOCK_N = 64

O_PROJ_BLOCK_N = 64

GATE_UP_BLOCK_N = 32

GEMV_BLOCK_K = 128

QKV_BLOCK_K = 64

SPLIT_CANDIDATES = (1, 2, 4, 8, 16)

GEMV_WARPS = 4

QKV_WARPS = 8

PIPELINE_STAGES = 3


@dataclasses.dataclass(frozen=True, kw_only=True)
class NormWeights:
    weight: torch.Tensor
    eps: float


@dataclasses.dataclass(frozen=True, kw_only=True)
class TowerLayerWeights:
    token_embedding: torch.Tensor
    enorm: NormWeights
    hnorm: NormWeights
    eh_proj: nn.Linear
    input_norm: NormWeights
    qkv_proj: nn.Linear
    q_norm: NormWeights
    k_norm: NormWeights
    o_proj: nn.Linear
    post_norm: NormWeights
    gate_up_proj: nn.Linear


@dataclasses.dataclass(frozen=True, kw_only=True)
class TowerLayerShape:
    dim: int
    head_count: int
    kv_head_count: int
    head_dim: int
    rotary_dim: int
    intermediate_size: int
    codebook_size: int
    mix_half_split: int
    qkv_split: int
    o_proj_split: int
    gate_up_split: int

    @property
    def query_width(self) -> int:
        return self.head_count * self.head_dim

    @property
    def qkv_part_count(self) -> int:
        return 2 * (self.head_count + self.kv_head_count)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SplitScratch:
    partials: torch.Tensor
    sum_of_squares: torch.Tensor
    norm_sums: torch.Tensor
    counters: torch.Tensor


def split_count(
    output_tiles: int, reduction_blocks: int, multiprocessor_count: int
) -> int:
    candidates = [
        reduction_split_count
        for reduction_split_count in SPLIT_CANDIDATES
        if reduction_blocks % reduction_split_count == 0
    ]
    return min(
        candidates,
        key=lambda reduction_split_count: abs(
            output_tiles * reduction_split_count - multiprocessor_count
        ),
    )


def tower_layer_shape(
    *,
    weights: TowerLayerWeights,
    head_count: int,
    kv_head_count: int,
    rotary_dim: int,
    multiprocessor_count: int,
) -> TowerLayerShape:
    hidden_size = weights.eh_proj.weight.shape[0]
    head_width = weights.q_norm.weight.shape[0]
    intermediate_size = weights.gate_up_proj.weight.shape[0] // 2
    return TowerLayerShape(
        dim=hidden_size,
        head_count=head_count,
        kv_head_count=kv_head_count,
        head_dim=head_width,
        rotary_dim=rotary_dim,
        intermediate_size=intermediate_size,
        codebook_size=weights.token_embedding.shape[0],
        mix_half_split=split_count(
            2 * (hidden_size // MIX_BLOCK_N),
            hidden_size // GEMV_BLOCK_K,
            multiprocessor_count,
        ),
        qkv_split=split_count(
            2 * (head_count + kv_head_count),
            hidden_size // QKV_BLOCK_K,
            multiprocessor_count,
        ),
        o_proj_split=split_count(
            hidden_size // O_PROJ_BLOCK_N,
            head_count * head_width // GEMV_BLOCK_K,
            multiprocessor_count,
        ),
        gate_up_split=split_count(
            intermediate_size // GATE_UP_BLOCK_N,
            hidden_size // GEMV_BLOCK_K,
            multiprocessor_count,
        ),
    )


if HAS_TRITON:

    @triton.jit
    def gemma_scaled(
        input_states: tl.tensor,
        norm_weight_pointer: tl.tensor,
        first_column: tl.tensor,
        BLOCK_K: tl.constexpr,
    ) -> tl.tensor:
        column_indices = first_column + tl.arange(0, BLOCK_K)
        norm_gains = 1.0 + tl.load(norm_weight_pointer + column_indices).to(tl.float32)
        return (input_states.to(tl.float32) * norm_gains[None, :]).to(tl.bfloat16)

    @triton.jit
    def weight_tile(
        weight_pointer: tl.tensor,
        weight_row_indices: tl.tensor,
        first_column: tl.tensor,
        ROW_WIDTH: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> tl.tensor:
        column_indices = first_column + tl.arange(0, BLOCK_K)
        return tl.load(
            weight_pointer
            + weight_row_indices[:, None] * ROW_WIDTH
            + column_indices[None, :]
        )

    @triton.jit
    def row_tile(
        values_pointer: tl.tensor,
        row_stride: tl.tensor,
        is_row: tl.tensor,
        first_column: tl.tensor,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> tl.tensor:
        row_indices = tl.arange(0, BLOCK_M)
        column_indices = first_column + tl.arange(0, BLOCK_K)
        return tl.load(
            values_pointer
            + row_indices[:, None] * row_stride
            + column_indices[None, :],
            mask=is_row[:, None],
            other=0.0,
        )

    @triton.jit
    def tile_sums(
        norm_sums_pointer: tl.tensor,
        TILES: tl.constexpr,
        TILE_BLOCK: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ) -> tl.tensor:
        tiles = tl.arange(0, TILE_BLOCK)
        row_indices = tl.arange(0, BLOCK_M)
        return tl.sum(
            tl.load(
                norm_sums_pointer + tiles[:, None] * BLOCK_M + row_indices[None, :],
                mask=(tiles < TILES)[:, None],
                other=0.0,
                cache_modifier=".cg",
            ),
            0,
        )

    @triton.jit
    def sigmoid(values: tl.tensor) -> tl.tensor:
        return tl.math.div_rn(
            tl.full(values.shape, 1.0, tl.float32), 1.0 + libdevice.exp(-values)
        )

    @triton.jit
    def normed_rotated(
        even: tl.tensor,
        odd: tl.tensor,
        norm_weight_pointer: tl.tensor,
        norm_eps: tl.tensor,
        cos_sin_pointer: tl.tensor,
        positions: tl.tensor,
        is_row: tl.tensor,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
    ) -> tuple[tl.tensor, ...]:
        pairs = tl.arange(0, HEAD_DIM // 2)
        head_rstd = libdevice.rsqrt(
            (tl.sum(even * even, 1) + tl.sum(odd * odd, 1)) / HEAD_DIM + norm_eps
        )
        even_gains = 1.0 + tl.load(norm_weight_pointer + 2 * pairs).to(tl.float32)
        odd_gains = 1.0 + tl.load(norm_weight_pointer + 2 * pairs + 1).to(tl.float32)
        even = (even * head_rstd[:, None] * even_gains[None, :]).to(tl.bfloat16)
        odd = (odd * head_rstd[:, None] * odd_gains[None, :]).to(tl.bfloat16)
        even = even.to(tl.float32)
        odd = odd.to(tl.float32)
        is_rotary = pairs < ROTARY_DIM // 2
        table_mask = is_row[:, None] & is_rotary[None, :]
        table_rows = positions[:, None] * ROTARY_DIM
        cosine = tl.load(
            cos_sin_pointer + table_rows + pairs[None, :], mask=table_mask, other=1.0
        )
        sine = tl.load(
            cos_sin_pointer + table_rows + ROTARY_DIM // 2 + pairs[None, :],
            mask=table_mask,
            other=0.0,
        )
        rotated_even = tl.where(is_rotary[None, :], even * cosine - odd * sine, even)
        rotated_odd = tl.where(is_rotary[None, :], odd * cosine + even * sine, odd)
        return rotated_even.to(tl.bfloat16), rotated_odd.to(tl.bfloat16)

else:
    pass
