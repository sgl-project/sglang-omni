# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Greedy codec selection, embedding lookup and accumulation."""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.base_dsl.jit_executor import JitCompiledFunction
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
    make_fake_tensor,
)
from cutlass.experimental import primitives as p

from sglang_omni.models.qwen3_tts.cutedsl.runtime import device_arch


@cute.jit
def prefer_candidate(
    value: cutlass.Numeric,
    index: cutlass.Numeric,
    best: cutlass.Numeric,
    best_index: cutlass.Numeric,
) -> cutlass.Numeric:
    """Order NaNs first, then descending values, then ascending indices."""
    is_nan = value != value  # noqa: PLR0124 - IEEE NaN test in device code.
    best_is_number = best == best  # noqa: PLR0124 - IEEE NaN test in device code.
    return (is_nan & (best_is_number | (index < best_index))) | (
        best_is_number & ((value > best) | ((value == best) & (index < best_index)))
    )


@cute.jit
def warp_argmax(
    value: cutlass.Numeric, index: cutlass.Numeric
) -> tuple[cutlass.Numeric, cutlass.Numeric]:
    for shift in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_bfly(value, 1 << shift)
        other_index = cute.arch.shuffle_sync_bfly(index, 1 << shift)
        if prefer_candidate(other, other_index, value, index):
            value, index = other, other_index
        else:
            pass
    return value, index


@cute.jit
def gather_row(
    table: cute.Tensor,
    output: cute.Tensor,
    row: cutlass.Int32,
    token: cutlass.Numeric,
    width: cutlass.Constexpr,
    accumulated: cute.Tensor | None = None,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(width % 8 == 0):
        for block in range(cute.ceil_div(width, 128 * 8)):
            col = block * 128 * 8 + tid * 8
            if col < width:
                values = (table.iterator.raw_ptr() + token * width + col).load(
                    count=8, alignment=16
                )
                if cutlass.const_expr(accumulated is not None):
                    pointer = accumulated.iterator.raw_ptr() + row * width + col
                    previous = pointer.load(count=8, alignment=16)
                else:
                    pass
                (output.iterator.raw_ptr() + row * width + col).store(
                    values, alignment=16
                )
                if cutlass.const_expr(accumulated is not None):
                    pointer.store(previous + values, alignment=16)
                else:
                    pass
            else:
                pass
    else:
        for block in range(cute.ceil_div(width, 128)):
            col = block * 128 + tid
            if col < width:
                output[row, col] = table[token, col]
                if cutlass.const_expr(accumulated is not None):
                    accumulated[row, col] = table.element_type(
                        cutlass.Float32(accumulated[row, col])
                        + cutlass.Float32(table[token, col])
                    )
                else:
                    pass
            else:
                pass


@cute.kernel
def argmax_embedding_kernel(
    logits: cute.Tensor,
    table: cute.Tensor,
    projected_table: cute.Tensor | None,
    codes: cute.Tensor,
    embeddings: cute.Tensor,
    projected: cute.Tensor | None,
    accumulated: cute.Tensor | None,
    V: cutlass.Constexpr,
    D: cutlass.Constexpr,
    DP: cutlass.Constexpr,
) -> None:
    row, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    partial_values = cutlass.Array(
        cutlass.Float32, 4, space=cutlass.AddressSpace.smem, alignment=16
    )
    partial_indices = cutlass.Array(
        cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16
    )
    winner = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
    )
    best, index = cutlass.Float32(float("-inf")), cutlass.Int32(0x7FFFFFFF)
    if cutlass.const_expr(V % 1024 == 0):
        for block in cutlass.range_constexpr(V // 1024):
            start = block * 1024 + tid * 8
            values = (logits.iterator.raw_ptr() + row * V + start).load(
                count=8, alignment=16
            )
            for item in cutlass.range_constexpr(8):
                col = start + item
                value = cutlass.Float32(values[item])
                if prefer_candidate(value, col, best, index):
                    best, index = value, col
                else:
                    pass
    else:
        for block in cutlass.range_constexpr(cute.ceil_div(V, 128)):
            col = block * 128 + tid
            if col < V:
                value = cutlass.Float32(logits[row, col])
                if prefer_candidate(value, col, best, index):
                    best, index = value, col
                else:
                    pass
            else:
                pass
    best, index = warp_argmax(best, index)
    if tid % 32 == 0:
        partial_values[tid // 32], partial_indices[tid // 32] = best, index
    else:
        pass
    p.barrier_cta_sync()
    if tid < 32:
        best, index = cutlass.Float32(float("-inf")), cutlass.Int32(0x7FFFFFFF)
        if tid < 4:
            best, index = partial_values[tid], partial_indices[tid]
        else:
            pass
        best, index = warp_argmax(best, index)
        if tid == 0:
            winner[0] = index
            codes[row] = cutlass.Int64(index)
        else:
            pass
    else:
        pass
    p.barrier_cta_sync()
    index = winner[0]
    gather_row(table, embeddings, row, index, D, accumulated)
    if cutlass.const_expr(DP > 0):
        gather_row(projected_table, projected, row, index, DP)
    else:
        pass


@cute.jit
def argmax_embedding_host(
    logits: cute.Tensor,
    table: cute.Tensor,
    projected_table: cute.Tensor,
    codes: cute.Tensor,
    embeddings: cute.Tensor,
    projected: cute.Tensor,
    accumulated: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    argmax_embedding_kernel(
        logits,
        table,
        projected_table,
        codes,
        embeddings,
        projected,
        accumulated,
        logits.shape[1],
        table.shape[1],
        projected_table.shape[1] if projected_table is not None else 0,
    ).launch(grid=(logits.shape[0], 1, 1), block=(128, 1, 1), stream=stream)


@cache
def compile_kernel(
    vocab: int,
    width: int,
    projected_width: int,
    logits_dtype: type[cutlass.Numeric],
    embedding_dtype: type[cutlass.Numeric],
    arch: str,
    accumulate: bool,
    code_stride: int,
) -> JitCompiledFunction:
    batch = cutlass.sym_int64()

    def fake(dtype: type[cutlass.Numeric], *shape: int | cute.SymInt) -> cute.Tensor:
        return make_fake_compact_tensor(
            dtype, shape, stride_order=tuple(reversed(range(len(shape))))
        )

    return cute.compile(
        argmax_embedding_host,
        fake(logits_dtype, batch, vocab),
        fake(embedding_dtype, vocab, width),
        fake(embedding_dtype, vocab, projected_width) if projected_width else None,
        make_fake_tensor(
            cutlass.Int64, (batch,), stride=(code_stride,), assumed_align=8
        ),
        fake(embedding_dtype, batch, width),
        fake(embedding_dtype, batch, projected_width) if projected_width else None,
        fake(embedding_dtype, batch, width) if accumulate else None,
        make_fake_stream(),
        options=f"--enable-tvm-ffi --gpu-arch={arch}",
    )


@torch.compiler.disable
def argmax_embedding(
    logits: torch.Tensor,
    table: torch.Tensor,
    projected_table: torch.Tensor | None = None,
    *,
    accumulated: torch.Tensor | None = None,
    out_codes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Select and gather, optionally updating the embedding sum and strided output-code slots."""
    types = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    if logits.ndim != 2 or table.ndim != 2 or logits.shape[1] != table.shape[0]:
        raise ValueError("Expected logits [B,V] and embedding table [V,D]")
    else:
        pass
    batch, vocab = logits.shape
    if batch < 1 or not 1 <= vocab <= 8192 or table.shape[1] < 1:
        raise ValueError("Expected B,D >= 1 and 1 <= V <= 8192")
    else:
        pass
    if logits.dtype not in types or table.dtype not in types:
        raise ValueError("Expected FP16, BF16, or FP32 tensors")
    else:
        pass
    tensors = [logits, table]
    if projected_table is not None:
        if (
            projected_table.ndim != 2
            or projected_table.shape[0] != vocab
            or projected_table.shape[1] < 1
            or projected_table.dtype != table.dtype
        ):
            raise ValueError("Expected projected table [V,DP] with the embedding dtype")
        else:
            pass
        tensors.append(projected_table)
    else:
        pass
    if accumulated is not None:
        if (
            accumulated.shape != (batch, table.shape[1])
            or accumulated.dtype != table.dtype
        ):
            raise ValueError("Expected accumulator [B,D] with the embedding dtype")
        else:
            pass
        tensors.append(accumulated)
    else:
        pass
    if any(
        not t.is_cuda or t.device != logits.device or not t.is_contiguous()
        for t in tensors
    ):
        raise ValueError("Expected contiguous tensors on the same CUDA device")
    else:
        pass
    arch = device_arch(logits.device)
    if out_codes is not None and (
        out_codes.shape != (batch,)
        or out_codes.dtype != torch.int64
        or out_codes.device != logits.device
        or out_codes.stride(0) < 1
    ):
        raise ValueError("Expected INT64 output-code slots [B] on the logits device")
    else:
        pass
    codes = (
        out_codes
        if out_codes is not None
        else torch.empty(batch, device=logits.device, dtype=torch.int64)
    )
    embeddings = table.new_empty(batch, table.shape[1])
    projected = (
        projected_table.new_empty(batch, projected_table.shape[1])
        if projected_table is not None
        else None
    )
    compile_kernel(
        vocab,
        table.shape[1],
        projected_table.shape[1] if projected_table is not None else 0,
        types[logits.dtype],
        types[table.dtype],
        arch,
        accumulated is not None,
        codes.stride(0),
    )(
        logits,
        table,
        projected_table,
        codes,
        embeddings,
        projected,
        accumulated,
        cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream),
    )
    return codes, embeddings, projected
