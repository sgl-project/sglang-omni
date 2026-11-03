# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Hopper WGMMA descriptors and FP32 accumulator fragments."""

import cutlass
from cutlass import cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import dsl_user_op


@cute.jit
def swizzle(row, k):
    """128-byte swizzle for a row of 64 BF16 elements."""
    return row * 64 + (k ^ ((row % 8) * 8))


@cute.jit
def descriptor(array, offset=0):
    """K-major descriptor for a 128-byte-swizzled operand with a 1-KiB-aligned base."""
    address = cutlass.Int64(array.data_ptr(offset).toint()) >> 4
    return address | (1 << 16) | (64 << 32) | (1 << 62)


@dsl_user_op
def mma(accumulator, weight_descriptor, input_descriptor, n, *, loc=None, ip=None):
    """Accumulate M64 x N x K16; each warpgroup thread owns N/2 values."""
    count = n // 2
    ty = ir.Type.parse("!llvm.struct<(" + ",".join(["f32"] * count) + ")>")
    value = llvm.UndefOp(ty).result
    for i in range(count):
        value = llvm.insertvalue(value, accumulator[i].ir_value(), [i])
    # The 4.8 experimental wrapper passes Numeric objects to this raw op;
    # use the NVVM binding with MLIR values until that wrapper is corrected.
    value = nvvm.wgmma_mma_async(
        ty,
        value,
        weight_descriptor.ir_value(),
        input_descriptor.ir_value(),
        ir.Attribute.parse(f"#nvvm.shape<m = 64, n = {n}, k = 16>"),
        nvvm.WGMMATypes.bf16,
        nvvm.WGMMATypes.bf16,
        nvvm.WGMMATypes.f32,
        nvvm.WGMMAScaleOut.one,
        nvvm.WGMMAScaleIn.one,
        nvvm.WGMMAScaleIn.one,
        nvvm.MMALayout.row,
        nvvm.MMALayout.col,
        loc=loc,
        ip=ip,
    )
    for i in range(count):
        accumulator[i] = cutlass.Float32(
            llvm.extractvalue(cutlass.Float32.mlir_type, value, [i])
        )


@cute.jit
def fragment_row(thread, index):
    return (thread // 32) * 16 + (thread % 32) // 4 + ((index % 4) // 2) * 8


@cute.jit
def fragment_column(thread, index):
    return (thread % 4) * 2 + index % 2 + (index // 4) * 8
