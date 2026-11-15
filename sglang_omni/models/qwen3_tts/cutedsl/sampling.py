# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Seeded top-k/top-p codec sampling with optional embedding lookup."""

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

from sglang_omni.models.qwen3_tts.cutedsl.embedding_update import gather_update_row
from sglang_omni.models.qwen3_tts.cutedsl.runtime import device_arch


@cute.jit
def select(
    condition: cutlass.Numeric, first: cutlass.Numeric, second: cutlass.Numeric
) -> cutlass.Numeric:
    return first if condition else second


@cute.jit
def warp_sort(keys: cutlass.Array, items: cutlass.Constexpr) -> cutlass.Array:
    """Descending bitonic sort of lane-interleaved register keys."""
    lane = cute.arch.lane_idx()
    for stage in cutlass.range_constexpr((32 * items).bit_length() - 1):
        for step in cutlass.range_constexpr(stage + 1):
            distance = 1 << (stage - step)
            if cutlass.const_expr(distance < 32):
                for item in cutlass.range_constexpr(items):
                    value = keys[item]
                    other = cute.arch.shuffle_sync_bfly(value, distance)
                    maximum = ((lane + item * 32) & (2 << stage) == 0) == (
                        lane & distance == 0
                    )
                    keys[item] = select((value > other) == maximum, value, other)
            else:
                for item in cutlass.range_constexpr(items):
                    partner = item ^ (distance // 32)
                    if cutlass.const_expr(item < partner):
                        first, second = keys[item], keys[partner]
                        descending = item * 32 & (2 << stage) == 0
                        keys[item] = select(
                            (first > second) == descending, first, second
                        )
                        keys[partner] = select(
                            (first > second) == descending, second, first
                        )
                    else:
                        pass
    return keys


@cute.jit
def warp_topk(
    keys: cutlass.Array,
    items: cutlass.Constexpr,
    keep: cutlass.Constexpr,
    presorted: cutlass.Constexpr = False,
) -> cutlass.Array:
    """Merge sorted runs, discarding the bottom half after each merge."""
    lane = cute.arch.lane_idx()
    if cutlass.const_expr(not presorted):
        for group in cutlass.range_constexpr(items // keep):
            run = cutlass.Array(cutlass.Uint32, keep)
            for item in cutlass.range_constexpr(keep):
                run[item] = keys[group * keep + item]
            run = warp_sort(run, keep)
            for item in cutlass.range_constexpr(keep):
                keys[group * keep + item] = run[item]
    else:
        pass
    for level in cutlass.range_constexpr((items // keep).bit_length() - 1):
        for group in cutlass.range_constexpr(items // (keep * (2 << level))):
            run = cutlass.Array(cutlass.Uint32, keep)
            for item in cutlass.range_constexpr(keep):
                first = keys[group * 2 * keep + item]
                second = cute.arch.shuffle_sync_bfly(
                    keys[(group * 2 + 2) * keep - item - 1], 31
                )
                run[item] = select(first > second, first, second)
            # note (Hao Wang): pairwise maxima form a bitonic run; only its merge remains.
            for step in cutlass.range_constexpr((32 * keep).bit_length() - 1):
                distance = (32 * keep) >> (step + 1)
                if cutlass.const_expr(distance < 32):
                    for item in cutlass.range_constexpr(keep):
                        value = run[item]
                        other = cute.arch.shuffle_sync_bfly(value, distance)
                        run[item] = select(
                            (value > other) == (lane & distance == 0), value, other
                        )
                else:
                    for item in cutlass.range_constexpr(keep):
                        partner = item ^ (distance // 32)
                        if cutlass.const_expr(item < partner):
                            first, second = run[item], run[partner]
                            run[item] = select(first > second, first, second)
                            run[partner] = select(first > second, second, first)
                        else:
                            pass
            for item in cutlass.range_constexpr(keep):
                keys[group * keep + item] = run[item]
    return keys


@cute.jit
def mix_hash(h: cutlass.Numeric, word: cutlass.Numeric) -> cutlass.Numeric:
    word *= cutlass.Uint32(0xCC9E2D51)
    word = (word << 15) | (word >> 17)
    word *= cutlass.Uint32(0x1B873593)
    h ^= word
    h = (h << 13) | (h >> 19)
    return h * cutlass.Uint32(5) + cutlass.Uint32(0xE6546B64)


@cute.jit
def gumbel(
    seed: cutlass.Numeric, position: cutlass.Numeric, rank: cutlass.Numeric
) -> cutlass.Float64:
    """Murmur3 counter-based draw, including the zero/one endpoint behavior."""
    h = mix_hash(cutlass.Uint32(0), cutlass.Uint32(seed))
    h = mix_hash(h, cutlass.Uint32(cutlass.Uint64(seed) >> 32))
    h = mix_hash(h, cutlass.Uint32(position))
    h = mix_hash(h, cutlass.Uint32(rank))
    h ^= cutlass.Uint32(16)
    h ^= h >> 16
    h *= cutlass.Uint32(0x85EBCA6B)
    h ^= h >> 13
    h *= cutlass.Uint32(0xC2B2AE35)
    h ^= h >> 16
    u = cutlass.Float64(h) / cutlass.Float64(4294967295.0)
    log_u = cute.math.log(u)
    log_u = cute.math.max(log_u, cutlass.Float64(-1.7976931348623157e308))
    log_u = cute.math.min(log_u, cutlass.Float64(-2.3283064365386963e-10))
    return -cute.math.log(-log_u)


@cute.jit
def warp_sum(value: cutlass.Numeric) -> cutlass.Numeric:
    for step in cutlass.range_constexpr(5):
        value += cute.arch.shuffle_sync_bfly(value, 1 << step)
    return value


@cute.jit
def warp_winner(score: cutlass.Numeric, rank: cutlass.Numeric) -> cutlass.Numeric:
    for step in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_bfly(score, 1 << step)
        other_rank = cute.arch.shuffle_sync_bfly(rank, 1 << step)
        take = (other > score) | ((other == score) & (other_rank < rank))
        score, rank = select(take, other, score), select(take, other_rank, rank)
    return rank


@cute.jit
def key_score(key: cutlass.Numeric) -> cutlass.Numeric:
    ordered = key >> 16
    bits = ordered ^ select(
        (ordered & 0x8000) != 0, cutlass.Uint32(0x8000), cutlass.Uint32(0xFFFF)
    )
    return cutlass.Float32(cutlass.Uint16(bits).bitcast(cutlass.BFloat16))


@cute.kernel
def sample_kernel(
    logits: cute.Tensor,
    temperatures: cute.Tensor,
    top_ks: cute.Tensor,
    top_ps: cute.Tensor,
    seeds: cute.Tensor,
    positions: cute.Tensor,
    codes: cute.Tensor,
    table: cute.Tensor,
    accumulated: cute.Tensor,
    gathered: cute.Tensor,
    projected_table: cute.Tensor,
    projected: cute.Tensor,
    top_k: cutlass.Constexpr,
    block_k: cutlass.Constexpr,
    warps: cutlass.Constexpr,
) -> None:
    row, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    lane, warp = tid % 32, tid // 32
    candidates = cutlass.Array(
        cutlass.Uint32, warps * block_k, space=cutlass.AddressSpace.smem, alignment=16
    )
    winner = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
    )
    noise = cutlass.Array(
        cutlass.Float64, block_k, space=cutlass.AddressSpace.smem, alignment=16
    )
    keys = cutlass.Array(cutlass.Uint32, 2048 // (warps * 32))
    start = tid * (2048 // (warps * 32))
    values = (logits.iterator.raw_ptr() + row * 2048 + start).load(
        count=2048 // (warps * 32), alignment=min(16, 4096 // (warps * 32))
    )
    for item in cutlass.range_constexpr(2048 // (warps * 32)):
        token = start + item
        bits = cutlass.Uint32(values[item].bitcast(cutlass.Uint16))
        ordered = bits ^ select(
            (bits & 0x8000) != 0, cutlass.Uint32(0xFFFF), cutlass.Uint32(0x8000)
        )
        keys[item] = (ordered << 16) | (cutlass.Uint32(0xFFFF) - cutlass.Uint32(token))
    keys = warp_topk(keys, 2048 // (warps * 32), block_k // 32)
    for item in cutlass.range_constexpr(cute.ceil_div(block_k, 32)):
        if item * 32 + lane < block_k:
            candidates[warp * block_k + item * 32 + lane] = keys[item]
        else:
            pass
    p.barrier_cta_sync()
    merged = cutlass.Array(cutlass.Uint32, cute.ceil_div(warps * block_k, 32))
    probabilities = cutlass.Array(cutlass.Float32, cute.ceil_div(block_k, 32))
    count, total = cutlass.Int32(0), cutlass.Float32(0)
    # note (Hao Wang): noise depends on rank, so it can overlap candidate selection.
    if warp > 0:
        for item in cutlass.range_constexpr(cute.ceil_div(block_k, (warps - 1) * 32)):
            rank = item * (warps - 1) * 32 + (warp - 1) * 32 + lane
            if rank < top_k:
                noise[rank] = gumbel(seeds[row], positions[row], rank)
            else:
                pass
    else:
        pass
    if warp == 0:
        for item in cutlass.range_constexpr(cute.ceil_div(warps * block_k, 32)):
            merged[item] = cutlass.Uint32(0)
            if item * 32 + lane < warps * block_k:
                merged[item] = candidates[item * 32 + lane]
            else:
                pass
        merged = warp_topk(merged, warps * block_k // 32, block_k // 32, True)
        count = cute.math.min(cutlass.Int32(top_ks[row]), cutlass.Int32(top_k))
        temperature = cute.math.max(temperatures[row], cutlass.Float32(1e-5))
        maximum = cute.arch.shuffle_sync(key_score(merged[0]) / temperature, 0)
        total = cutlass.Float32(0)
        for item in cutlass.range_constexpr(cute.ceil_div(block_k, 32)):
            probability = cutlass.Float32(0)
            if item * 32 + lane < count:
                probability = cute.math.exp(
                    key_score(merged[item]) / temperature - maximum
                )
            else:
                pass
            probabilities[item] = probability
            total += probability
        total = warp_sum(total)
    else:
        pass
    p.barrier_cta_sync()
    if warp == 0:
        before = cutlass.Float32(0)
        best, best_rank = cutlass.Float64(float("-inf")), cutlass.Int32(0x7FFFFFFF)
        for item in cutlass.range_constexpr(cute.ceil_div(block_k, 32)):
            rank = item * 32 + lane
            probability = probabilities[item] / total
            cumulative = probability
            for step in cutlass.range_constexpr(5):
                earlier = cute.arch.shuffle_sync_up(cumulative, 1 << step)
                if lane >= (1 << step):
                    cumulative += earlier
                else:
                    pass
            prefix = before + cumulative - probability
            before += cute.arch.shuffle_sync(cumulative, 31)
            cutoff = top_ps[row]
            keep = (rank < count) & (
                (cutoff <= 0) | (cutoff >= 1) | (prefix < cutoff) | (rank == 0)
            )
            score = cutlass.Float64(float("-inf"))
            if keep:
                score = cutlass.Float64(cute.math.log(probability)) + noise[rank]
            else:
                pass
            if (score > best) | ((score == best) & (rank < best_rank)):
                best, best_rank = score, rank
            else:
                pass
        best_rank = warp_winner(best, best_rank)
        token = cutlass.Int32(0)
        for item in cutlass.range_constexpr(cute.ceil_div(block_k, 32)):
            candidate = cute.arch.shuffle_sync(merged[item], best_rank % 32)
            if best_rank // 32 == item:
                token = cutlass.Int32(0xFFFF - (candidate & 0xFFFF))
            else:
                pass
        if lane == 0:
            codes[row] = cutlass.Int64(token)
            winner[0] = token
        else:
            pass
    else:
        pass
    if cutlass.const_expr(table is not None):
        p.barrier_cta_sync()
        token = winner[0]
        for tile in range(cute.ceil_div(table.shape[1], warps * 32 * 8)):
            start = (tile * warps * 32 + tid) * 8
            if start < table.shape[1]:
                gather_update_row(
                    table, gathered, accumulated, row, token, start, table.shape[1]
                )
            else:
                pass
        if cutlass.const_expr(projected_table is not None):
            for tile in range(cute.ceil_div(projected_table.shape[1], warps * 32 * 8)):
                start = (tile * warps * 32 + tid) * 8
                if start < projected_table.shape[1]:
                    gather_update_row(
                        projected_table,
                        projected,
                        None,
                        row,
                        token,
                        start,
                        projected_table.shape[1],
                    )
                else:
                    pass
        else:
            pass
    else:
        pass


@cute.jit
def sample_host(
    logits: cute.Tensor,
    temperatures: cute.Tensor,
    top_ks: cute.Tensor,
    top_ps: cute.Tensor,
    seeds: cute.Tensor,
    positions: cute.Tensor,
    codes: cute.Tensor,
    table: cute.Tensor,
    accumulated: cute.Tensor,
    gathered: cute.Tensor,
    projected_table: cute.Tensor,
    projected: cute.Tensor,
    stream: cuda.CUstream,
    top_k: cutlass.Constexpr,
    block_k: cutlass.Constexpr,
    warps: cutlass.Constexpr,
) -> None:
    sample_kernel(
        logits,
        temperatures,
        top_ks,
        top_ps,
        seeds,
        positions,
        codes,
        table,
        accumulated,
        gathered,
        projected_table,
        projected,
        top_k,
        block_k,
        warps,
    ).launch(grid=(logits.shape[0], 1, 1), block=(warps * 32, 1, 1), stream=stream)


@cache
def compile_kernel(
    top_k: int,
    width: int,
    projected_width: int,
    accumulate: bool,
    code_stride: int,
    warps: int,
    arch: str,
) -> JitCompiledFunction:
    batch = cutlass.sym_int64()

    def fake(dtype: type[cutlass.Numeric], *shape: int | cute.SymInt) -> cute.Tensor:
        return make_fake_compact_tensor(
            dtype, shape, stride_order=tuple(reversed(range(len(shape))))
        )

    return cute.compile(
        sample_host,
        fake(cutlass.BFloat16, batch, 2048),
        fake(cutlass.Float32, batch),
        fake(cutlass.Int64, batch),
        fake(cutlass.Float32, batch),
        fake(cutlass.Int64, batch),
        fake(cutlass.Int64, batch),
        make_fake_tensor(
            cutlass.Int64, (batch,), stride=(code_stride,), assumed_align=8
        ),
        fake(cutlass.BFloat16, 2048, width) if width else None,
        fake(cutlass.BFloat16, batch, width) if accumulate else None,
        fake(cutlass.BFloat16, batch, width) if width else None,
        fake(cutlass.BFloat16, 2048, projected_width) if projected_width else None,
        fake(cutlass.BFloat16, batch, projected_width) if projected_width else None,
        make_fake_stream(),
        top_k,
        max(32, 1 << (top_k - 1).bit_length()),
        warps,
        options=f"--enable-tvm-ffi --gpu-arch={arch} --ptxas-options=--fmad=false",
    )


@cute.kernel
def sorted_sample_kernel(
    logprobs: cute.Tensor,
    indices: cute.Tensor,
    seeds: cute.Tensor,
    positions: cute.Tensor,
    output: cute.Tensor,
    count: cutlass.Constexpr,
    warps: cutlass.Constexpr,
) -> None:
    row, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    lane, warp = tid % 32, tid // 32
    best, best_rank = cutlass.Float64(float("-inf")), cutlass.Int32(0x7FFFFFFF)
    best_token = cutlass.Int64(0)
    seed, position = seeds[row], positions[row]
    for item in cutlass.range_constexpr(cute.ceil_div(count, warps * 32)):
        rank = item * warps * 32 + tid
        score = cutlass.Float64(float("-inf"))
        token = cutlass.Int64(0)
        if rank < count:
            if cutlass.const_expr(warps == 1):
                token = indices[row, rank]
            else:
                pass
            score = cutlass.Float64(logprobs[row, rank]) + gumbel(seed, position, rank)
        else:
            pass
        if (score > best) | ((score == best) & (rank < best_rank)):
            best, best_rank = score, rank
            best_token = token
        else:
            pass
    if cutlass.const_expr(warps > 1):
        scores = cutlass.Array(
            cutlass.Float64, warps, space=cutlass.AddressSpace.smem, alignment=8
        )
        ranks = cutlass.Array(
            cutlass.Int32, warps, space=cutlass.AddressSpace.smem, alignment=4
        )
        selected = warp_winner(best, best_rank)
        selected_score = cute.arch.shuffle_sync(best, selected % 32)
        if lane == 0:
            scores[warp], ranks[warp] = selected_score, selected
        else:
            pass
        p.barrier_cta_sync()
        if warp == 0:
            best, best_rank = cutlass.Float64(float("-inf")), cutlass.Int32(0x7FFFFFFF)
            if lane < warps:
                best, best_rank = scores[lane], ranks[lane]
            else:
                pass
            best_rank = warp_winner(best, best_rank)
            if lane == 0:
                output[row] = indices[row, best_rank]
            else:
                pass
        else:
            pass
    else:
        best_rank = warp_winner(best, best_rank)
        best_token = cute.arch.shuffle_sync(best_token, best_rank % 32)
        if lane == 0:
            output[row] = best_token
        else:
            pass


@cute.jit
def sorted_sample_host(
    logprobs: cute.Tensor,
    indices: cute.Tensor,
    seeds: cute.Tensor,
    positions: cute.Tensor,
    output: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    count = logprobs.shape[1]
    warps = 1 if count <= 32 else (4 if count <= 128 else (8 if count <= 512 else 32))
    sorted_sample_kernel(
        logprobs, indices, seeds, positions, output, logprobs.shape[1], warps
    ).launch(grid=(logprobs.shape[0], 1, 1), block=(warps * 32, 1, 1), stream=stream)


@cache
def compile_sorted(count: int, arch: str) -> JitCompiledFunction:
    batch = cutlass.sym_int64()

    def fake(dtype: type[cutlass.Numeric], *shape: int | cute.SymInt) -> cute.Tensor:
        return make_fake_compact_tensor(
            dtype, shape, stride_order=tuple(reversed(range(len(shape))))
        )

    return cute.compile(
        sorted_sample_host,
        fake(cutlass.Float32, batch, count),
        fake(cutlass.Int64, batch, count),
        fake(cutlass.Int64, batch),
        fake(cutlass.Int64, batch),
        fake(cutlass.Int64, batch),
        make_fake_stream(),
        options=f"--enable-tvm-ffi --gpu-arch={arch} --ptxas-options=--fmad=false",
    )


@torch.compiler.disable
def sample_sorted(
    logprobs: torch.Tensor,
    indices: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Sample prefiltered FP32 log probabilities using their supplied candidate order."""
    if logprobs.ndim != 2 or logprobs.dtype != torch.float32 or not logprobs.is_cuda:
        raise ValueError("Expected CUDA FP32 log probabilities [B,K]")
    else:
        pass
    batch, count = logprobs.shape
    if batch < 1 or not 1 <= count <= 2048 or indices.shape != logprobs.shape:
        raise ValueError(
            "Expected matching [B,K] probabilities/indices, B > 0 and K <= 2048"
        )
    else:
        pass
    if (
        not logprobs.is_contiguous()
        or any(
            t.device != logprobs.device
            or t.dtype != torch.int64
            or not t.is_contiguous()
            for t in (indices, seeds, positions)
        )
        or seeds.shape != (batch,)
        or positions.shape != (batch,)
    ):
        raise ValueError(
            "Expected contiguous INT64 indices, seeds and positions on the same CUDA device"
        )
    else:
        pass
    output = indices.new_empty(batch)
    compile_sorted(count, device_arch(logprobs.device))(
        logprobs,
        indices,
        seeds,
        positions,
        output,
        cuda.CUstream(torch.cuda.current_stream(logprobs.device).cuda_stream),
    )
    return output


@torch.compiler.disable
def sample_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    *,
    max_top_k: int = 50,
    table: torch.Tensor | None = None,
    accumulated: torch.Tensor | None = None,
    projected_table: torch.Tensor | None = None,
    out_codes: torch.Tensor | None = None,
    warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Sample BF16 [B,2048] logits and optionally gather and accumulate embeddings.
    Each row needs top_k in [1,max_top_k], finite temperature, and at least one finite logit; other logits may be -inf.
    Disabled top-k requires full-vocabulary filtering and sample_sorted, not clamping to max_top_k. Equal score bit patterns use ascending token ID.
    """
    if (
        logits.ndim != 2
        or logits.shape[1] != 2048
        or logits.dtype != torch.bfloat16
        or not logits.is_cuda
    ):
        raise ValueError("Expected CUDA BF16 logits [B,2048]")
    else:
        pass
    batch = logits.shape[0]
    if batch == 0 or max_top_k < 1 or max_top_k > 128 or warps not in (4, 8):
        raise ValueError("Expected B > 0, 1 <= max_top_k <= 128, and 4 or 8 warps")
    else:
        pass
    if not logits.is_contiguous() or any(
        t.shape != (batch,)
        or t.device != logits.device
        or t.dtype != dtype
        or not t.is_contiguous()
        for t, dtype in (
            (temperatures, torch.float32),
            (top_ks, torch.int64),
            (top_ps, torch.float32),
            (seeds, torch.int64),
            (positions, torch.int64),
        )
    ):
        raise ValueError(
            "Expected contiguous logits and per-row FP32 temperature/top_p, INT64 top_k/seed/position"
        )
    else:
        pass
    width = 0 if table is None else table.shape[1]
    projected_width = 0 if projected_table is None else projected_table.shape[1]
    if (accumulated is not None or projected_table is not None) and table is None:
        raise ValueError("Embedding outputs require an embedding table")
    else:
        pass
    for tensor, shape in (
        (table, (2048, width)),
        (accumulated, (batch, width)),
        (projected_table, (2048, projected_width)),
    ):
        if tensor is not None and (
            tensor.shape != shape
            or tensor.dtype != torch.bfloat16
            or tensor.device != logits.device
            or not tensor.is_contiguous()
            or shape[1] < 8
            or shape[1] % 8
        ):
            raise ValueError(
                "Expected contiguous BF16 tables/accumulator with widths divisible by eight"
            )
        else:
            pass
    if out_codes is not None and (
        out_codes.shape != (batch,)
        or out_codes.dtype != torch.int64
        or out_codes.device != logits.device
        or out_codes.stride(0) < 1
    ):
        raise ValueError("Expected INT64 output slots [B] on the logits device")
    else:
        pass
    codes = (
        logits.new_empty(batch, dtype=torch.int64) if out_codes is None else out_codes
    )
    gathered = logits.new_empty(batch, width) if width else None
    projected = logits.new_empty(batch, projected_width) if projected_width else None
    compile_kernel(
        max_top_k,
        width,
        projected_width,
        accumulated is not None,
        codes.stride(0),
        warps,
        device_arch(logits.device),
    )(
        logits,
        temperatures,
        top_ks,
        top_ps,
        seeds,
        positions,
        codes,
        table,
        accumulated,
        gathered,
        projected_table,
        projected,
        cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream),
    )
    return codes, gathered, projected
