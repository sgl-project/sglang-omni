# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Predictor square sums with Tensor Core accumulation order."""

import cutlass
from cutlass import cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def fma(a, b, c, *, loc=None, ip=None):
    """Round a multiply-add once, including when contraction is disabled."""
    return cutlass.Float32(
        llvm.intr_fma(a.ir_value(), b.ir_value(), c.ir_value(), loc=loc, ip=ip)
    )


@dsl_user_op
def diagonal_mma(low, high, first, second, *, loc=None, ip=None):
    """Accumulate eight rows' square sums on an M16N8 diagonal."""
    low, high = low.ir_value(), high.ir_value()
    value = nvvm.mma_sync(
        ir.Type.parse("!llvm.struct<(f32,f32,f32,f32)>"),
        ir.Attribute.parse("#nvvm.shape<m = 16, n = 8, k = 16>"),
        nvvm.MMALayout.row,
        nvvm.MMALayout.col,
        [low, low, high, high],
        [low, high],
        [first.ir_value(), second.ir_value(), first.ir_value(), second.ir_value()],
        multiplicand_a_ptx_type=nvvm.MMATypes.bf16,
        multiplicand_b_ptx_type=nvvm.MMATypes.bf16,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Float32(llvm.extractvalue(cutlass.Float32.mlir_type, value, [0])),
        cutlass.Float32(llvm.extractvalue(cutlass.Float32.mlir_type, value, [1])),
    )


@cute.jit
def square_sum(
    raw,
    row,
    lane,
    batch_tile: cutlass.Constexpr,
    begin_tile=0,
    tiles: cutlass.Constexpr = 64,
    packed: cutlass.Constexpr = False,
):
    """Reduce K16 tiles in order; diagonal lanes own the eight row sums."""
    first, second = cutlass.Float32(0), cutlass.Float32(0)
    for tile in cutlass.range_constexpr(tiles):
        column = (begin_tile + tile) * 16 + (lane % 4) * 2
        offset = (
            (column // 64) * batch_tile * 64
            + row * 64
            + ((column % 64) ^ ((row % 8) * 8))
        )
        if cutlass.const_expr(packed):
            low = cutlass.Float32(raw.load(idx=offset // 2)).bitcast(cutlass.Int32)
            high = cutlass.Float32(raw.load(idx=(offset ^ 8) // 2)).bitcast(
                cutlass.Int32
            )
        else:
            low = raw.load(idx=offset, vector_size=2, alignment=4).bitcast(
                cutlass.Int32
            )[0]
            high = raw.load(idx=offset ^ 8, vector_size=2, alignment=4).bitcast(
                cutlass.Int32
            )[0]
        first, second = diagonal_mma(low, high, first, second)
    return cutlass.select_(lane // 4 == (lane % 4) * 2, first, second)
