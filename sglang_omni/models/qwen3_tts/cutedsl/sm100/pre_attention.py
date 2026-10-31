# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Blackwell fused QKV preparation."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.experimental import cuda as tensor_map
from cutlass.experimental import primitives as p

from sglang_omni.models.qwen3_tts.cutedsl.sm100.mma import (
    descriptor,
    fragment_column,
    fragment_row,
    load_accumulator,
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
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    head, rank = block // 4, block % 4
    split = rank % 2
    weights = cutlass.Array(
        cutlass.BFloat16,
        64 * (hidden_size // 2),
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    normalized = cutlass.Array(
        cutlass.BFloat16,
        batch_tile * (hidden_size // 2),
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    partials = cutlass.Array(
        cutlass.Float32,
        4 * batch_tile * 64,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    if cutlass.const_expr(hidden_size == 2048):
        # Only rank 0 receives remote partials; senders can stage in their own mailbox.
        local_partial = partials
    else:
        local_partial = cutlass.Array(
            cutlass.Float32,
            batch_tile * 64,
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
    bars = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    mma_done = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    tmem_address = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    if thread >= 128 and thread < 160:
        p.tcgen05_alloc(tmem_address, 32)
        p.tcgen05_relinquish_alloc_permit()
    else:
        pass
    if thread == 0:
        p.prefetch_tensormap(weight_map.get_ptr())
        p.mbarrier_init(bars, 1)
        p.mbarrier_init(mma_done, 1)
        if rank == 0:
            p.mbarrier_init(bars.subview(1), 1)
            p.mbarrier_arrive_expect_tx(bars.subview(1), 3 * batch_tile * 64 * 4)
        else:
            pass
        p.fence_mbarrier_init()
        p.mbarrier_arrive_expect_tx(bars, 64 * hidden_size)
        p.cp_async_bulk_tensor_shared_cluster_global(
            weights,
            weight_map.get_ptr(),
            [0, 0, split * (hidden_size // 128), head * 2 + rank // 2],
            bars,
            [],
        )
    else:
        pass

    lane, warp = thread % 32, thread // 32
    for token in cutlass.range_constexpr(batch_tile // 16):
        row = warp + token * 16
        values = cute.make_rmem_tensor((8, hidden_size // 256), cutlass.BFloat16)
        scales = cute.make_rmem_tensor((8, hidden_size // 256), cutlass.BFloat16)
        for group in cutlass.range_constexpr(hidden_size // 256):
            column = lane * 8 + group * 256
            val = cutlass.full((8,), 0, cutlass.BFloat16)
            if row < batch_size:
                val = (x.iterator.raw_ptr() + row * hidden_size + column).load(
                    count=8, alignment=16
                )
            else:
                pass
            scale = (gamma.iterator.raw_ptr() + column).load(count=8, alignment=16)
            for i in cutlass.range_constexpr(8):
                values[i, group], scales[i, group] = val[i], scale[i]
        total = cutlass.Float32(0)
        for group in cutlass.range_constexpr(hidden_size // 256):
            v = values[None, group].load().to_vector().to(cutlass.Float32)
            squares = v * v
            total += ((squares[0] + squares[1]) + (squares[2] + squares[3])) + (
                (squares[4] + squares[5]) + (squares[6] + squares[7])
            )
        for shift in cutlass.range_constexpr(5):
            total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
        inverse = cute.math.rsqrt(total / hidden_size + norm_epsilon)
        for group in cutlass.range_constexpr(hidden_size // 256):
            if group // (hidden_size // 512) == split:
                column = lane * 8 + (group % (hidden_size // 512)) * 256
                v = values[None, group].load().to_vector().to(cutlass.Float32)
                norm_scale = scales[None, group].load().to_vector().to(cutlass.Float32)
                norm_result = (
                    (v * inverse).to(cutlass.BFloat16).to(cutlass.Float32) * norm_scale
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
    p.barrier_cluster_arrive_relaxed()

    accumulator = cute.make_rmem_tensor((batch_tile // 2,), cutlass.Float32)
    accumulator.fill(0)
    if thread >= 128 and thread < 256:
        while not p.mbarrier_try_wait_parity(bars, 0):
            pass
        p.tcgen05_fence(p.Tcgen05Fence.AFTER_THREAD_SYNC)
        if thread == 128:
            da_base = descriptor(weights)
            db_base = descriptor(normalized)
            # Advance the encoded 16-byte address instead of rebuilding each descriptor.
            for slab in cutlass.range_constexpr(hidden_size // 128):
                weight_descriptor = da_base + slab * 512
                input_descriptor = db_base + slab * batch_tile * 8
                for k in cutlass.range_constexpr(4):
                    mma(
                        tmem_address.load(),
                        weight_descriptor + 2 * k,
                        input_descriptor + 2 * k,
                        batch_tile,
                        slab != 0 or k != 0,
                    )
            p.tcgen05_commit(mma_done)
        else:
            pass
        while not p.mbarrier_try_wait_parity(mma_done, 0):
            pass
        load_accumulator(accumulator, tmem_address.load(), thread - 128, batch_tile)
    else:
        pass
    if thread >= 128 and thread < 256:
        for i in cutlass.range_constexpr(batch_tile // 2):
            channel = fragment_row(thread - 128, i)
            row = fragment_column(thread - 128, i)
            if rank == 0:
                partials.store(accumulator[i], idx=row * 64 + channel)
            else:
                local_partial.store(accumulator[i], idx=row * 64 + channel)
    else:
        pass
    p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
    p.barrier_cta_sync()
    p.barrier_cluster_wait()
    if thread == 0 and rank != 0:
        p.cp_async_bulk_shared_cluster_shared_cta(
            p.mapa(partials.subview(rank * batch_tile * 64), 0),
            local_partial,
            p.mapa(bars.subview(1), 0),
            batch_tile * 64 * 4,
        )
    else:
        pass
    if rank == 0:
        while not p.mbarrier_try_wait_parity(bars.subview(1), 0):
            pass
        for token in cutlass.range_constexpr(batch_tile // 16):
            row = warp + token * 16
            position = positions[0]
            slot = cutlass.Int64(0)
            if row < batch_size:
                position = positions[row]
                slot = cutlass.Int64(cache_slots[row])
            else:
                pass
            channel = lane * 4
            half, local_channel = channel // 64, channel % 64
            first = partials.load(
                idx=(half * 2 * batch_tile + row) * 64 + local_channel,
                vector_size=4,
                alignment=16,
            )
            second = partials.load(
                idx=((half * 2 + 1) * batch_tile + row) * 64 + local_channel,
                vector_size=4,
                alignment=16,
            )
            projected = (first + second).to(cutlass.BFloat16)
            if head < 24:
                vf = projected.to(cutlass.Float32)
                head_squares = vf * vf
                total = (head_squares[0] + head_squares[1]) + (
                    head_squares[2] + head_squares[3]
                )
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
                    .to(cutlass.BFloat16)
                    .to(cutlass.Float32)
                )
                sv = (
                    (cos.iterator.raw_ptr() + rope_offset + 64)
                    .load(count=4, alignment=8)
                    .to(cutlass.BFloat16)
                    .to(cutlass.Float32)
                )
                left = (scaled * cv).to(cutlass.BFloat16).to(cutlass.Float32)
                right = (
                    (rotated.load().to_vector() * sv)
                    .to(cutlass.BFloat16)
                    .to(cutlass.Float32)
                )
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
    if thread >= 128 and thread < 160:
        p.tcgen05_dealloc(cutlass.inttoptr(tmem_address.load(), 6, cutlass.Int32), 32)
    else:
        pass


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
):
    batch_size = cutlass.Int32(x.shape[0])
    weight_map = tensor_map.create_tensor_map_tiled_from_view(
        weights,
        (1, hidden_size // 128, 64, 64),
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
    ).launch(grid=(128, 1, 1), block=(512, 1, 1), cluster=(4, 1, 1), stream=stream)
