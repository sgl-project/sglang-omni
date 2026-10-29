# SPDX-License-Identifier: Apache-2.0
"""Optional BF16 kernels for selected FFN1 and packed QKV shapes."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def ffn_gemm_kernel(
    X,
    W,
    BIAS,
    Y,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tiles_m = tl.cdiv(M, BLOCK_M)
    rows = tl.program_id(0) % tiles_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) // tiles_m * BLOCK_N + tl.arange(0, BLOCK_N)
    reductions = tl.arange(0, BLOCK_K)
    state = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)
    bias = tl.load(BIAS + cols, mask=cols < 3072, other=0).to(tl.float32)
    for chunk in range(8):
        partial = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for step in tl.static_range(128 // BLOCK_K):
            offsets_k = chunk * 128 + step * BLOCK_K + reductions
            left = tl.load(
                X + rows[:, None] * 1024 + offsets_k[None, :],
                mask=rows[:, None] < M,
                other=0,
            )
            right = tl.load(
                W + offsets_k[:, None] + cols[None, :] * 1024,
                mask=cols[None, :] < 3072,
                other=0,
            )
            partial = tl.dot(left, right, partial, input_precision="ieee")
        # note (Codex): keep prior BF16 state outside the FP32 dot accumulation.
        accumulator = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[state.to(tl.float32), partial],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        if chunk == 7:
            accumulator = tl.inline_asm_elementwise(
                "add.rn.f32 $0, $1, $2;",
                constraints="=f,f,f",
                args=[accumulator, tl.broadcast_to(bias[None, :], (BLOCK_M, BLOCK_N))],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        else:
            pass
        state = accumulator.to(tl.bfloat16)
    tl.store(
        Y + rows[:, None] * 3072 + cols[None, :],
        state,
        (rows[:, None] < M) & (cols[None, :] < 3072),
    )


@triton.jit
def packed_fp32_kernel(
    X,
    W,
    BIAS,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tiles_m = tl.cdiv(M, BLOCK_M)
    tile_m = tl.program_id(0) % tiles_m
    tile_n = tl.program_id(0) // tiles_m
    split = tl.program_id(1)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    chunk_length: tl.constexpr = tl.cdiv(K, BLOCK_K) * BLOCK_K
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for chunk in range(tl.cdiv(chunk_length, BLOCK_K)):
        offsets_k = split * chunk_length + chunk * BLOCK_K + reduction
        left = tl.load(
            X + rows[:, None] * K + offsets_k[None, :],
            (rows[:, None] < M) & (offsets_k[None, :] < K),
            other=0,
        )
        right = tl.load(
            W + offsets_k[:, None] + cols[None, :] * K,
            (offsets_k[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        accumulator = tl.dot(left, right, accumulator, input_precision="ieee")
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    bias = tl.load(BIAS + cols, cols < N, other=0).to(tl.float32)
    result = accumulator + bias[None, :]
    tl.store(Y + offsets, result, mask)


def gemm_out(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    config: tuple[int, int, int, int, int],
) -> torch.Tensor:
    """Write a checked BF16 1024-to-3072 projection without allocating workspace."""
    block_m, block_n, block_k, warps, stages = config
    rows = inputs.shape[0]
    grid = triton.cdiv(rows, block_m) * triton.cdiv(3072, block_n)
    if rows in (24, 48, 96, 192):
        ffn_gemm_kernel[(grid,)](
            inputs,
            weight,
            bias,
            out,
            rows,
            block_m,
            block_n,
            block_k,
            num_warps=warps,
            num_stages=stages,
            enable_fp_fusion=True,
        )
    elif rows == 384:
        assert config == (64, 64, 64, 4, 3)
        # note (Codex): retain the measured single-partition launch geometry.
        packed_fp32_kernel[(grid, 1)](
            inputs,
            weight,
            bias,
            out,
            rows,
            3072,
            1024,
            block_m,
            block_n,
            block_k,
            num_warps=warps,
            num_stages=stages,
            enable_fp_fusion=True,
        )
    else:
        raise ValueError(f"Unsupported FFN1/packed QKV row count: {rows}")
    return out
