# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Codec embedding lookup and accumulation with optional projected lookup."""

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

from sglang_omni.models.qwen3_tts.cutedsl.runtime import device_arch


@cute.jit
def gather_update_row(
    table: cute.Tensor,
    gathered: cute.Tensor,
    accumulated: cute.Tensor,
    row: cutlass.Int32,
    token: cutlass.Numeric,
    start: cutlass.Int32,
    width: cutlass.Constexpr,
) -> None:
    """Use 128-bit transfers and BF16 additions with round-to-nearest."""
    values = (table.iterator.raw_ptr() + token * width + start).load(
        count=8, alignment=16
    )
    if cutlass.const_expr(accumulated is not None):
        pointer = accumulated.iterator.raw_ptr() + row * width + start
        previous = pointer.load(count=8, alignment=16)
    else:
        pass
    (gathered.iterator.raw_ptr() + row * width + start).store(values, alignment=16)
    if cutlass.const_expr(accumulated is not None):
        pointer.store(previous + values, alignment=16)
    else:
        pass


@cute.kernel
def embedding_update_kernel(
    tokens: cute.Tensor,
    table: cute.Tensor,
    accumulated: cute.Tensor,
    gathered: cute.Tensor,
    projected_table: cute.Tensor,
    projected: cute.Tensor,
    width: cutlass.Constexpr,
    projected_width: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    row, tile, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    token = tokens[row]
    embedding_tiles = cute.ceil_div(width, threads * 8)
    if tile < embedding_tiles:
        start = (tile * threads + tid) * 8
        if start < width:
            gather_update_row(table, gathered, accumulated, row, token, start, width)
        else:
            pass
    else:
        if cutlass.const_expr(projected_width > 0):
            start = ((tile - embedding_tiles) * threads + tid) * 8
            if start < projected_width:
                gather_update_row(
                    projected_table, projected, None, row, token, start, projected_width
                )
            else:
                pass
        else:
            pass


@cute.jit
def embedding_update_host(
    tokens: cute.Tensor,
    table: cute.Tensor,
    accumulated: cute.Tensor,
    gathered: cute.Tensor,
    projected_table: cute.Tensor,
    projected: cute.Tensor,
    stream: cuda.CUstream,
    threads: cutlass.Constexpr,
) -> None:
    width = table.shape[1]
    projected_width = projected_table.shape[1] if projected_table is not None else 0
    embedding_update_kernel(
        tokens,
        table,
        accumulated,
        gathered,
        projected_table,
        projected,
        width,
        projected_width,
        threads,
    ).launch(
        grid=(
            tokens.shape[0],
            cute.ceil_div(width, threads * 8)
            + cute.ceil_div(projected_width, threads * 8),
            1,
        ),
        block=(threads, 1, 1),
        stream=stream,
    )


@cache
def compile_kernel(
    width: int,
    projected_width: int,
    token_type: type[cutlass.Numeric],
    arch: str,
    threads: int,
    token_stride: int = 1,
) -> JitCompiledFunction:
    batch, vocab = cutlass.sym_int64(), cutlass.sym_int64()

    def fake(dtype: type[cutlass.Numeric], *shape: int | cute.SymInt) -> cute.Tensor:
        return make_fake_compact_tensor(
            dtype, shape, stride_order=tuple(reversed(range(len(shape))))
        )

    return cute.compile(
        embedding_update_host,
        make_fake_tensor(
            token_type,
            (batch,),
            stride=(token_stride,),
            assumed_align=token_type.width // 8,
        ),
        fake(cutlass.BFloat16, vocab, width),
        fake(cutlass.BFloat16, batch, width),
        fake(cutlass.BFloat16, batch, width),
        fake(cutlass.BFloat16, vocab, projected_width) if projected_width else None,
        fake(cutlass.BFloat16, batch, projected_width) if projected_width else None,
        make_fake_stream(),
        threads,
        options=f"--enable-tvm-ffi --gpu-arch={arch}",
    )


@torch.compiler.disable
def embedding_update(
    tokens: torch.Tensor,
    table: torch.Tensor,
    accumulated: torch.Tensor,
    projected_table: torch.Tensor | None = None,
    *,
    threads: int = 128,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return gathered rows and optional projected rows; update accumulated in place."""
    if (
        tokens.ndim != 1
        or table.ndim != 2
        or accumulated.shape != (tokens.numel(), table.shape[1])
    ):
        raise ValueError("Expected token IDs [B], table [V,D], and accumulator [B,D]")
    else:
        pass
    if tokens.dtype not in (torch.int32, torch.int64) or table.dtype != torch.bfloat16:
        raise ValueError("Expected INT32/INT64 IDs and BF16 embeddings")
    else:
        pass
    if table.shape[1] % 8 or table.shape[1] == 0 or tokens.numel() == 0:
        raise ValueError("Expected B > 0 and positive D divisible by eight")
    else:
        pass
    tensors = [table, accumulated]
    if projected_table is not None:
        if (
            projected_table.ndim != 2
            or projected_table.shape[0] != table.shape[0]
            or projected_table.shape[1] % 8
        ):
            raise ValueError(
                "Expected projected table [V,DP] with DP divisible by eight"
            )
        else:
            pass
        tensors.append(projected_table)
    else:
        pass
    if (
        not tokens.is_cuda
        or tokens.stride(0) < 1
        or any(
            t.device != tokens.device or t.dtype != table.dtype or not t.is_contiguous()
            for t in tensors
        )
    ):
        raise ValueError(
            "Expected contiguous tensors on the same CUDA device, with BF16 tables and accumulator"
        )
    else:
        pass
    if threads not in (32, 64, 128):
        raise ValueError("Expected 32, 64 or 128 threads")
    else:
        pass
    batch, width = accumulated.shape
    projected_width = projected_table.shape[1] if projected_table is not None else 0
    gathered = torch.empty_like(accumulated)
    projected = table.new_empty(batch, projected_width) if projected_width else None
    compile_kernel(
        width,
        projected_width,
        cutlass.Int32 if tokens.dtype == torch.int32 else cutlass.Int64,
        device_arch(tokens.device),
        threads,
        tokens.stride(0),
    )(
        tokens,
        table,
        accumulated,
        gathered,
        projected_table,
        projected,
        cuda.CUstream(torch.cuda.current_stream(tokens.device).cuda_stream),
    )
    return gathered, projected
