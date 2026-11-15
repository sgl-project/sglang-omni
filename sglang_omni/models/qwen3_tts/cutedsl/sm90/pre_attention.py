# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Hopper fused QKV preparation."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.experimental import cuda as tensor_map
from cutlass.experimental import primitives as p

from sglang_omni.models.qwen3_tts.cutedsl.normalization import fma, square_sum
from sglang_omni.models.qwen3_tts.cutedsl.sm90.mma import (
    descriptor,
    fragment_column,
    fragment_row,
    mma,
    swizzle,
)


@cute.kernel
def pre_attention_kernel(
    x: cute.Tensor,
    gamma: cute.Tensor,
    weight_map: cutlass.GridConstant[tensor_map.TensorMap],
    qgamma: cute.Tensor,
    kgamma: cute.Tensor,
    cos: cute.Tensor,
    positions: cute.Tensor,
    kv: cute.Tensor,
    q: cute.Tensor,
    batch_size: cutlass.Int32,
    batch_tile: cutlass.Constexpr,
    norm_epsilon: cutlass.Constexpr,
    hidden_size: cutlass.Constexpr,
    cache_slots: cute.Tensor,
    value_cache: cute.Tensor,
    is_predictor: cutlass.Constexpr = False,
    residual: cute.Tensor = None,
    residual_output: cute.Tensor = None,
    deferred_residual: cutlass.Constexpr = False,
    has_residual: cutlass.Constexpr = False,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    # note (Hao Wang): Full-K avoids split-reduction rounding before talker Q/K normalization.
    k_splits: cutlass.Constexpr = 1 if deferred_residual else 2
    cluster_size: cutlass.Constexpr = 2 * k_splits
    local_k: cutlass.Constexpr = hidden_size // k_splits
    head, rank = block // cluster_size, block % cluster_size
    split = rank % k_splits
    chains: cutlass.Constexpr = 2 if is_predictor else 1
    weights = cutlass.Array(
        cutlass.BFloat16,
        64 * local_k,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    normalized = cutlass.Array(
        cutlass.BFloat16,
        batch_tile * local_k,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    partials = cutlass.Array(
        cutlass.Float32,
        cluster_size * chains * batch_tile * 64,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    if cutlass.const_expr(hidden_size == 2048 or is_predictor):
        # Only rank 0 receives remote partials; senders can stage in their own mailbox.
        local_partial = partials
    else:
        local_partial = cutlass.Array(
            cutlass.Float32,
            chains * batch_tile * 64,
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
    bars = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    if cutlass.const_expr(is_predictor):
        norm_partials = cutlass.Array(
            cutlass.Float32,
            4 * batch_tile,
            space=cutlass.AddressSpace.smem,
            alignment=16,
        )
        # note (Hao Wang): Norm inputs expire before remote partials arrive, so the mailbox can hold both.
        raw_inputs = partials.subview(2 * batch_tile * 64)
    else:
        pass
    if thread == 0:
        p.prefetch_tensormap(weight_map.get_ptr())
        p.mbarrier_init(bars, 1)
        if rank == 0:
            p.mbarrier_init(bars.subview(1), 1)
            p.mbarrier_arrive_expect_tx(
                bars.subview(1),
                (cluster_size - 1) * chains * batch_tile * 64 * 4
                + (2 * batch_tile * 4 if is_predictor else 0),
            )
        else:
            pass
        p.fence_mbarrier_init()
        p.mbarrier_arrive_expect_tx(bars, 128 * local_k)
        p.cp_async_bulk_tensor_shared_cluster_global(
            weights,
            weight_map.get_ptr(),
            [0, 0, split * (local_k // 64), head * 2 + rank // k_splits],
            bars,
            [],
        )
    else:
        pass

    lane, warp = thread % 32, thread // 32
    input_inverse = cute.make_rmem_tensor((batch_tile // 16,), cutlass.Float32)
    input_groups: cutlass.Constexpr = hidden_size // (512 if is_predictor else 256)
    for token in cutlass.range_constexpr(batch_tile // 16):
        row = warp + token * 16
        values = cute.make_rmem_tensor(
            (8, local_k // 256),
            cutlass.Float32 if deferred_residual else cutlass.BFloat16,
        )
        scales = cute.make_rmem_tensor((8, local_k // 256), cutlass.BFloat16)
        total = cutlass.Float32(0)
        for group in cutlass.range_constexpr(input_groups):
            column = lane * 8 + group * 256
            if cutlass.const_expr(is_predictor):
                column += split * (hidden_size // 2)
            else:
                pass
            val = cutlass.full((8,), 0, cutlass.BFloat16)
            if row < batch_size:
                val = (x.iterator.raw_ptr() + row * hidden_size + column).load(
                    count=8, alignment=16
                )
            else:
                pass
            if cutlass.const_expr(is_predictor):
                if rank < 2:
                    local_column = column % (hidden_size // 2)
                    raw_inputs.store(
                        val.bitcast(cutlass.Float32),
                        idx=(
                            (local_column // 64) * batch_tile * 64
                            + swizzle(row, local_column % 64)
                        )
                        // 2,
                        alignment=16,
                    )
                else:
                    pass
            else:
                pass
            if cutlass.const_expr(deferred_residual):
                val = val.to(cutlass.Float32)
                if cutlass.const_expr(has_residual):
                    if row < batch_size:
                        val += (
                            (residual.iterator.raw_ptr() + row * hidden_size + column)
                            .load(count=8, alignment=16)
                            .to(cutlass.Float32)
                        )
                    else:
                        pass
                else:
                    pass
                if head == 0 and rank == 0 and row < batch_size:
                    (
                        residual_output.iterator.raw_ptr() + row * hidden_size + column
                    ).store(val.to(cutlass.BFloat16), alignment=16)
                else:
                    pass
            else:
                pass
            if cutlass.const_expr(not is_predictor and not deferred_residual):
                reduction_values = val.to(cutlass.Float32)
                squares = reduction_values * reduction_values
                total += ((squares[0] + squares[1]) + (squares[2] + squares[3])) + (
                    (squares[4] + squares[5]) + (squares[6] + squares[7])
                )
            else:
                pass
            if is_predictor or group // (local_k // 256) == split:
                scale = (gamma.iterator.raw_ptr() + column).load(count=8, alignment=16)
                for i in cutlass.range_constexpr(8):
                    values[i, group % (local_k // 256)] = val[i]
                    scales[i, group % (local_k // 256)] = scale[i]
            else:
                pass
        if cutlass.const_expr(not is_predictor):
            if cutlass.const_expr(deferred_residual):
                input_values = values.load()
                total = (input_values * input_values).reduce(
                    cute.ReductionOp.ADD,
                    init_val=cutlass.Float32(0),
                    reduction_profile=0,
                )
            else:
                pass
            for shift in cutlass.range_constexpr(5):
                total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
            input_rms_inverse = cute.math.rsqrt(total / hidden_size + norm_epsilon)
        else:
            pass
        for group in cutlass.range_constexpr(input_groups):
            if is_predictor or group // (local_k // 256) == split:
                column = lane * 8 + (group % (local_k // 256)) * 256
                v = (
                    values[None, group % (local_k // 256)]
                    .load()
                    .to_vector()
                    .to(cutlass.Float32)
                )
                norm_scale = (
                    scales[None, group % (local_k // 256)]
                    .load()
                    .to_vector()
                    .to(cutlass.Float32)
                )
                if cutlass.const_expr(is_predictor):
                    norm_result = (v * norm_scale).to(cutlass.BFloat16)
                elif cutlass.const_expr(deferred_residual):
                    norm_result = ((v * input_rms_inverse) * norm_scale).to(
                        cutlass.BFloat16
                    )
                else:
                    norm_result = (
                        (v * input_rms_inverse).to(cutlass.BFloat16).to(cutlass.Float32)
                        * norm_scale
                    ).to(cutlass.BFloat16)
                normalized.store(
                    norm_result,
                    idx=(column // 64) * batch_tile * 64 + swizzle(row, column % 64),
                    alignment=16,
                )
            else:
                pass
    p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
    p.barrier_cta_sync()
    # Join the cluster before publishing DSMEM, but overlap its arrival with MMA.
    if cutlass.const_expr(not is_predictor):
        p.barrier_cluster_arrive_relaxed()
    else:
        pass

    accumulator = cute.make_rmem_tensor((batch_tile // 2, chains), cutlass.Float32)
    accumulator.fill(0)
    if thread >= 128 and thread < 256:
        while not p.mbarrier_try_wait_parity(bars, 0):
            pass
        p.wgmma_fence_aligned()
        for slab in cutlass.range_constexpr(local_k // 64):
            weight_descriptor = descriptor(weights, slab * 4096)
            input_descriptor = descriptor(normalized, slab * batch_tile * 64)
            for k in cutlass.range_constexpr(4):
                mma(
                    accumulator[None, slab // (local_k // 64 // chains)],
                    weight_descriptor + 2 * k,
                    input_descriptor + 2 * k,
                    batch_tile,
                )
        p.wgmma_commit_group_sync_aligned()
        p.wgmma_wait_group_sync_aligned(0)
    else:
        pass
    if cutlass.const_expr(is_predictor):
        if rank < 2 and (warp < 4 or warp >= 8):
            task = cutlass.select_(warp < 4, warp, warp - 4)
            if task < batch_tile // 4:
                row = (task % (batch_tile // 8)) * 8 + lane // 4
                piece = task // (batch_tile // 8)
                total = square_sum(
                    raw_inputs,
                    row,
                    lane,
                    batch_tile,
                    begin_tile=piece * 16,
                    tiles=16,
                    packed=True,
                )
                if lane // 4 == (lane % 4) * 2 or lane // 4 == (lane % 4) * 2 + 1:
                    norm_partials.store(total, idx=piece * batch_tile + row)
                else:
                    pass
            else:
                pass
        else:
            pass
    else:
        pass
    if thread >= 128 and thread < 256:
        for chain in cutlass.range_constexpr(chains):
            for i in cutlass.range_constexpr(batch_tile // 2):
                channel = fragment_row(thread - 128, i)
                row = fragment_column(thread - 128, i)
                if rank == 0:
                    partials.store(
                        accumulator[i, chain],
                        idx=(chain * batch_tile + row) * 64 + channel,
                    )
                else:
                    local_partial.store(
                        accumulator[i, chain],
                        idx=(chain * batch_tile + row) * 64 + channel,
                    )
    else:
        pass
    p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
    p.barrier_cta_sync()
    if cutlass.const_expr(is_predictor):
        p.barrier_cluster_arrive_relaxed()
    else:
        pass
    p.barrier_cluster_wait()
    if thread == 0 and rank != 0:
        p.cp_async_bulk_shared_cluster_shared_cta(
            p.mapa(partials.subview(rank * chains * batch_tile * 64), 0),
            local_partial,
            p.mapa(bars.subview(1), 0),
            chains * batch_tile * 64 * 4,
        )
    else:
        pass
    if cutlass.const_expr(is_predictor):
        if thread == 0 and rank == 1:
            p.cp_async_bulk_shared_cluster_shared_cta(
                p.mapa(norm_partials.subview(2 * batch_tile), 0),
                norm_partials,
                p.mapa(bars.subview(1), 0),
                2 * batch_tile * 4,
            )
        else:
            pass
    else:
        pass
    if rank == 0:
        while not p.mbarrier_try_wait_parity(bars.subview(1), 0):
            pass
        for token in cutlass.range_constexpr(batch_tile // 16):
            row = warp + token * 16
            if cutlass.const_expr(is_predictor):
                total = cutlass.Float32(0)
                for piece in cutlass.range_constexpr(4):
                    total += cutlass.Float32(
                        norm_partials.load(idx=piece * batch_tile + row)
                    )
                input_inverse[token] = cute.math.rsqrt(
                    total / hidden_size + norm_epsilon
                )
            else:
                pass
            position = positions[0]
            slot = cutlass.Int64(0)
            if row < batch_size:
                position = positions[row]
                slot = cutlass.Int64(cache_slots[row])
            else:
                pass
            channel = lane * 4
            half, local_channel = channel // 64, channel % 64
            product = cutlass.full((4,), 0, cutlass.Float32)
            for piece in cutlass.range_constexpr(k_splits * chains):
                product += partials.load(
                    idx=((half * k_splits * chains + piece) * batch_tile + row) * 64
                    + local_channel,
                    vector_size=4,
                    alignment=16,
                )
            if cutlass.const_expr(is_predictor):
                product *= input_inverse[token]
            else:
                pass
            projected = product.to(cutlass.BFloat16)
            if head < 24:
                vf = projected.to(cutlass.Float32)
                head_squares = vf * vf
                total = (head_squares[0] + head_squares[1]) + (
                    head_squares[2] + head_squares[3]
                )
                if cutlass.const_expr(is_predictor):
                    for shift in cutlass.range_constexpr(5):
                        delta: cutlass.Constexpr = (1, 8, 4, 2, 16)[shift]
                        total += cute.arch.shuffle_sync_bfly(total, delta)
                else:
                    for shift in cutlass.range_constexpr(5):
                        total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
                inverse = cute.math.rsqrt(total / 128 + norm_epsilon)
                head_scale = cutlass.full((4,), 0, cutlass.BFloat16)
                if head < 16:
                    head_scale = (qgamma.iterator.raw_ptr() + channel).load(
                        count=4, alignment=8
                    )
                else:
                    head_scale = (kgamma.iterator.raw_ptr() + channel).load(
                        count=4, alignment=8
                    )
                if cutlass.const_expr(is_predictor or deferred_residual):
                    unit = vf * inverse
                else:
                    unit = (vf * inverse).to(cutlass.BFloat16).to(cutlass.Float32)
                scaled = (
                    (unit * head_scale.to(cutlass.Float32))
                    .to(cutlass.BFloat16)
                    .to(cutlass.Float32)
                )
                rotated = cute.make_rmem_tensor((4,), cutlass.Float32)
                for i in cutlass.range_constexpr(4):
                    opposite = cute.arch.shuffle_sync_bfly(scaled[i], 16)
                    rotated[i] = cutlass.select_(channel < 64, -opposite, opposite)
                rope_offset = position * 128 + channel % 64
                cv = (
                    (cos.iterator.raw_ptr() + rope_offset)
                    .load(count=4, alignment=8)
                    .to(cutlass.Float32)
                )
                sv = (
                    (cos.iterator.raw_ptr() + rope_offset + 64)
                    .load(count=4, alignment=8)
                    .to(cutlass.Float32)
                )
                if cutlass.const_expr(is_predictor or deferred_residual):
                    left = scaled * cv
                    right = rotated.load().to_vector() * sv
                else:
                    cv = cv.to(cutlass.BFloat16).to(cutlass.Float32)
                    sv = sv.to(cutlass.BFloat16).to(cutlass.Float32)
                    left = (scaled * cv).to(cutlass.BFloat16).to(cutlass.Float32)
                    right = (
                        (rotated.load().to_vector() * sv)
                        .to(cutlass.BFloat16)
                        .to(cutlass.Float32)
                    )
                if cutlass.const_expr(is_predictor or deferred_residual):
                    rope_result = cute.make_rmem_tensor((4,), cutlass.Float32)
                    for i in cutlass.range_constexpr(4):
                        rope_result[i] = fma(scaled[i], cv[i], right[i])
                    result = rope_result.load().to_vector().to(cutlass.BFloat16)
                else:
                    result = (left + right).to(cutlass.BFloat16)
                if row < batch_size:
                    if head < 16:
                        (
                            q.iterator.raw_ptr() + (row * 16 + head) * 128 + channel
                        ).store(result, alignment=8)
                    else:
                        (
                            kv.iterator.raw_ptr()
                            + (slot * 8 + head - 16) * 128
                            + channel
                        ).store(result, alignment=8)
                else:
                    pass
            else:
                if row < batch_size:
                    (
                        value_cache.iterator.raw_ptr()
                        + (slot * 8 + head - 24) * 128
                        + channel
                    ).store(projected, alignment=8)
                else:
                    pass
    else:
        pass
    p.barrier_cluster_arrive_relaxed()
    p.barrier_cluster_wait()


@cute.jit
def pre_attention_host(
    x: cute.Tensor,
    gamma: cute.Tensor,
    weights: cute.Tensor,
    qgamma: cute.Tensor,
    kgamma: cute.Tensor,
    cos: cute.Tensor,
    positions: cute.Tensor,
    kv: cute.Tensor,
    q: cute.Tensor,
    stream: cuda.CUstream,
    batch_tile: cutlass.Constexpr,
    norm_epsilon: cutlass.Constexpr,
    hidden_size: cutlass.Constexpr,
    cache_slots: cute.Tensor,
    value_cache: cute.Tensor,
    is_predictor: cutlass.Constexpr = False,
    residual: cute.Tensor = None,
    residual_output: cute.Tensor = None,
    deferred_residual: cutlass.Constexpr = False,
    has_residual: cutlass.Constexpr = False,
):
    batch_size = cutlass.Int32(x.shape[0])
    k_splits: cutlass.Constexpr = 1 if deferred_residual else 2
    weight_map = tensor_map.create_tensor_map_tiled_from_view(
        weights,
        (1, hidden_size // (64 * k_splits), 64, 64),
        swizzle=tensor_map.TensorMapSwizzle.s128b,
    )
    pre_attention_kernel(
        x,
        gamma,
        weight_map,
        qgamma,
        kgamma,
        cos,
        positions,
        kv,
        q,
        batch_size,
        batch_tile,
        norm_epsilon,
        hidden_size,
        cache_slots,
        value_cache,
        is_predictor,
        residual,
        residual_output,
        deferred_residual,
        has_residual,
    ).launch(
        grid=(64 * k_splits, 1, 1),
        block=(512, 1, 1),
        cluster=(2 * k_splits, 1, 1),
        stream=stream,
    )
