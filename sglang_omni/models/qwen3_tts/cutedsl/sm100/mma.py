# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Blackwell M64 GEMM and TMEM accumulator fragments."""

import cutlass
from cutlass import cute
from cutlass.experimental import primitives as p


@cute.jit
def swizzle(row, k):
    """128-byte swizzle for a row of 64 BF16 elements."""
    return row * 64 + (k ^ ((row % 8) * 8))


@cute.jit
def descriptor(array, offset=0):
    """K-major descriptor for 128-byte-swizzled BF16 operands."""
    return p.Tcgen05SmemDesc.build(
        start_address=array.subview(offset),
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=p.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )


@cute.jit
def mma(tmem, weight_descriptor, input_descriptor, n: cutlass.Constexpr, accumulate):
    """Issue M64 x N x K16 from one thread into TMEM."""
    instruction = p.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        m_dim=64,
        n_dim=n,
    )
    p.tcgen05_mma(
        p.Tcgen05MMAKind.F16,
        p.CTAGroup.CTA_1,
        cutlass.inttoptr(tmem, 6, cutlass.Int32),
        weight_descriptor,
        input_descriptor,
        instruction,
        accumulate,
    )


@cute.jit
def load_accumulator(accumulator, tmem, thread, n: cutlass.Constexpr):
    """Read M64's four 16-row bands into the epilogue's per-thread fragments."""
    p.tcgen05_fence(p.Tcgen05Fence.AFTER_THREAD_SYNC)
    address = tmem + ((thread // 32) * 32 << 16)
    values = p.tcgen05_ld(
        "16x256b", cutlass.inttoptr(address, 6, cutlass.Float32), num=n // 8
    )
    p.tcgen05_wait(p.Tcgen05Wait.LOAD)
    for i in cutlass.range_constexpr(n // 2):
        accumulator[i] = values[i]
    p.tcgen05_fence(p.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def fragment_row(thread, index):
    return (thread // 32) * 16 + (thread % 32) // 4 + ((index % 4) // 2) * 8


@cute.jit
def fragment_column(thread, index):
    return (thread % 4) * 2 + index % 2 + (index // 4) * 8
