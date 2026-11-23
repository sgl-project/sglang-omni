# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Blackwell fused post-attention for hidden=2048 and FFN=6144."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.experimental import cuda as tensor_map
from cutlass.experimental import primitives as p

from sglang_omni.models.qwen3_tts.cutedsl.sm100.mma import descriptor, swizzle
from sglang_omni.models.qwen3_tts.cutedsl.sm100.post_attention import (
    PAYLOAD_EMPTY,
    wait_barrier,
)


@cute.jit
def mma(tmem, weight_descriptor, input_descriptor, n: cutlass.Constexpr, accumulate):
    """Issue one M128 x N x K16 MMA into TMEM."""
    instruction = p.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        m_dim=128,
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
def load_accumulator(accumulator, tmem, n: cutlass.Constexpr):
    """Read one output channel per thread from the M128 accumulator."""
    p.tcgen05_fence(p.Tcgen05Fence.AFTER_THREAD_SYNC)
    values = p.tcgen05_ld("32x32b", cutlass.inttoptr(tmem, 6, cutlass.Float32), num=n)
    p.tcgen05_wait(p.Tcgen05Wait.LOAD)
    for i in cutlass.range_constexpr(n):
        accumulator[i] = values[i]
    p.tcgen05_fence(p.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def gate_up_index(row, column, batch_tile: cutlass.Constexpr):
    """Swizzle channels for fragment stores and contiguous paired SiLU reads."""
    return column * 128 + (row ^ (((column // 2) % 4) * 8))


@cute.jit
def scatter_partial(
    accumulator,
    mailbox,
    ready,
    phase: cutlass.Constexpr,
    thread,
    rank,
    batch_tile: cutlass.Constexpr,
):
    """Scatter four K partials; each rank owns 32 output channels."""
    if thread >= 128:
        row = thread - 128
        for column in cutlass.range_constexpr(batch_tile):
            cute.arch.store_async_dsmem(
                cute.make_ptr(
                    cutlass.Float32,
                    mailbox.data_ptr(
                        (rank * batch_tile + column) * 32 + row % 32
                    ).ir_value(),
                ),
                accumulator[column].bitcast(cutlass.Int32),
                cute.make_ptr(cutlass.Int64, ready.data_ptr(phase).ir_value()),
                peer_cta_rank=row // 32,
            )
    else:
        pass
    wait_barrier(ready.subview(phase), 0)


@cute.jit
def reduce_partial(mailbox, row, column, batch_tile: cutlass.Constexpr):
    """Preserve the baseline four-way FP32 accumulation order."""
    value = cutlass.Float32(0)
    for peer in cutlass.range_constexpr(4):
        value += mailbox.load(idx=(peer * batch_tile + column) * 32 + row % 32)
    return value


@cute.jit
def project(
    accumulator,
    tmem,
    weights,
    operands,
    inverse_rms,
    r,
    gamma,
    bar,
    free,
    weight_map,
    attention,
    activation,
    thread,
    rank,
    tile,
    slot,
    projection: cutlass.Constexpr,
    batch_size: cutlass.Int32,
    batch_tile: cutlass.Constexpr,
):
    """Double-buffer weights and operands between producer and MMA warpgroups."""
    K = 256
    stages = (
        2048 if projection == "gate_up" else (1536 if projection == "down" else 512)
    ) // K
    accumulator.fill(0)
    if thread == 0:
        for buffer in cutlass.range_constexpr(2):
            p.mbarrier_init(bar.subview(buffer), 1)
            p.mbarrier_init(free.subview(buffer), 1)
            p.mbarrier_arrive(free.subview(buffer))
    else:
        pass
    p.fence_mbarrier_init()
    p.barrier_cta_sync()
    if thread < 128:
        for stage in cutlass.range_constexpr(stages):
            buffer = stage % 2
            wait_barrier(free.subview(buffer), (stage // 2) & 1)
            channel_start = (
                stage * K
                if projection == "gate_up"
                else rank * (1536 if projection == "down" else 512) + stage * K
            )
            if thread == 0:
                p.mbarrier_expect_tx(bar.subview(buffer), 128 * K * 2)
                p.cp_async_bulk_tensor_shared_cluster_global(
                    weights.subview(buffer * 128 * K),
                    weight_map.get_ptr(),
                    [0, 0, channel_start // 64, tile],
                    bar.subview(buffer),
                    [],
                )
            else:
                pass
            if cutlass.const_expr(projection == "down"):
                payload = cute.make_rmem_tensor(
                    (4, K * batch_tile // 1024), cutlass.Int32
                )
                for j in cutlass.range_constexpr(K * batch_tile // 1024):
                    idx = (thread + j * 128) * 8
                    slab, row, k = (
                        idx // (batch_tile * 64),
                        (idx // 64) % batch_tile,
                        idx % 64,
                    )
                    channel = channel_start + slab * 64 + k
                    offset = (
                        (slot * 96 + channel // 64) * batch_tile + row
                    ) * 32 + channel % 64 // 2
                    values = p.load_ext(
                        activation.iterator.raw_ptr() + offset,
                        dtype=cutlass.Int32,
                        count=4,
                        order="volatile",
                    )
                    for word in cutlass.range_constexpr(4):
                        payload[word, j] = values[word]
                ready = cutlass.Boolean(False)
                while not ready:
                    ready = cutlass.Boolean(True)
                    for j in cutlass.range_constexpr(K * batch_tile // 1024):
                        if cutlass.any_(
                            payload[None, j].load().to_vector() == PAYLOAD_EMPTY
                        ):
                            idx = (thread + j * 128) * 8
                            slab, row, k = (
                                idx // (batch_tile * 64),
                                (idx // 64) % batch_tile,
                                idx % 64,
                            )
                            channel = channel_start + slab * 64 + k
                            offset = (
                                (slot * 96 + channel // 64) * batch_tile + row
                            ) * 32 + channel % 64 // 2
                            values = p.load_ext(
                                activation.iterator.raw_ptr() + offset,
                                dtype=cutlass.Int32,
                                count=4,
                                order="volatile",
                            )
                            for word in cutlass.range_constexpr(4):
                                payload[word, j] = values[word]
                        else:
                            pass
                        ready &= cutlass.all_(
                            payload[None, j].load().to_vector() != PAYLOAD_EMPTY
                        )
            else:
                pass
            if cutlass.const_expr(projection == "gate_up"):
                payload = cute.make_rmem_tensor(
                    (4, K * batch_tile // 1024), cutlass.Int32
                )
                scales = cute.make_rmem_tensor(
                    (4, K * batch_tile // 1024), cutlass.Int32
                )
                for j in cutlass.range_constexpr(K * batch_tile // 1024):
                    idx = (thread + j * 128) * 8
                    slab, row, k = (
                        idx // (batch_tile * 64),
                        (idx // 64) % batch_tile,
                        idx % 64,
                    )
                    channel = channel_start + slab * 64 + k
                    offset = (slot * batch_tile + row) * 1024 + channel // 2
                    values = p.load_ext(
                        r.iterator.raw_ptr() + offset,
                        dtype=cutlass.Int32,
                        count=4,
                        order="volatile",
                    )
                    scale = (
                        (gamma.iterator.raw_ptr() + channel)
                        .load(count=8, alignment=16)
                        .bitcast(cutlass.Int32)
                    )
                    for word in cutlass.range_constexpr(4):
                        payload[word, j] = values[word]
                        scales[word, j] = scale[word]
            else:
                pass
            for j in cutlass.range_constexpr(K * batch_tile // 1024):
                idx = (thread + j * 128) * 8
                slab, row, k = (
                    idx // (batch_tile * 64),
                    (idx // 64) % batch_tile,
                    idx % 64,
                )
                value = cutlass.full((8,), 0, cutlass.BFloat16)
                if cutlass.const_expr(projection == "output"):
                    if row < batch_size:
                        value = (
                            attention.iterator.raw_ptr()
                            + row * 2048
                            + channel_start
                            + slab * 64
                            + k
                        ).load(count=8, alignment=16)
                    else:
                        pass
                elif cutlass.const_expr(projection == "gate_up"):
                    words = payload[None, j].load().to_vector()
                    raw = words.bitcast(cutlass.BFloat16).to(cutlass.Float32)
                    scale = (
                        scales[None, j]
                        .load()
                        .to_vector()
                        .bitcast(cutlass.BFloat16)
                        .to(cutlass.Float32)
                    )
                    unit = (
                        (raw * inverse_rms.load(idx=row))
                        .to(cutlass.BFloat16)
                        .to(cutlass.Float32)
                    )
                    value = (unit * scale).to(cutlass.BFloat16)
                else:
                    value = (
                        payload[None, j].load().to_vector().bitcast(cutlass.BFloat16)
                    )
                operands.store(
                    value,
                    idx=buffer * batch_tile * K
                    + slab * batch_tile * 64
                    + swizzle(row, k),
                    alignment=16,
                )
            p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
            p.barrier_cta_sync(barrier_id=1, thread_count=128)
            if thread == 0:
                p.mbarrier_arrive(bar.subview(buffer))
            else:
                pass
    else:
        # note (Hao Wang): Keep phase tracking on the issuer so delayed reader warps cannot miss buffer reuse.
        if thread == 128:
            for stage in cutlass.range_constexpr(stages):
                buffer = stage % 2
                wait_barrier(bar.subview(buffer), (stage // 2) & 1)
                p.tcgen05_fence(p.Tcgen05Fence.AFTER_THREAD_SYNC)
                da_base = descriptor(weights, buffer * 128 * K)
                db_base = descriptor(operands, buffer * batch_tile * K)
                for slab in cutlass.range_constexpr(K // 64):
                    for k in cutlass.range_constexpr(4):
                        mma(
                            tmem,
                            da_base + slab * 1024 + 2 * k,
                            db_base + slab * batch_tile * 8 + 2 * k,
                            batch_tile,
                            stage != 0 or slab != 0 or k != 0,
                        )
                # Hardware releases the buffer; the next MMA need not wait for this one.
                p.tcgen05_commit(free.subview(buffer))
            wait_barrier(free.subview((stages - 1) % 2), ((stages - 1) // 2 + 1) & 1)
        else:
            pass
        p.barrier_cta_sync(barrier_id=2, thread_count=128)
        p.tcgen05_fence(p.Tcgen05Fence.AFTER_THREAD_SYNC)
        load_accumulator(accumulator, tmem, batch_tile)
    p.barrier_cta_sync()


@cute.jit
def compute_inverse_rms(
    r,
    inverse_rms,
    thread,
    slot,
    batch_tile: cutlass.Constexpr,
    norm_epsilon: cutlass.Constexpr,
):
    """Poll each residual word and retain one FP32 inverse RMS per token."""
    for token in cutlass.range(batch_tile // 8, unroll=1):
        row = thread // 32 + token * 8
        words = cute.make_rmem_tensor((4, 8), cutlass.Int32)
        for group in cutlass.range_constexpr(8):
            offset = (slot * batch_tile + row) * 1024 + (thread % 32) * 4 + group * 128
            values = p.load_ext(
                r.iterator.raw_ptr() + offset,
                dtype=cutlass.Int32,
                count=4,
                order="volatile",
            )
            for i in cutlass.range_constexpr(4):
                words[i, group] = values[i]
        ready = cutlass.Boolean(False)
        while not ready:
            ready = cutlass.Boolean(True)
            for group in cutlass.range_constexpr(8):
                if cutlass.any_(words[None, group].load().to_vector() == PAYLOAD_EMPTY):
                    offset = (
                        (slot * batch_tile + row) * 1024
                        + (thread % 32) * 4
                        + group * 128
                    )
                    values = p.load_ext(
                        r.iterator.raw_ptr() + offset,
                        dtype=cutlass.Int32,
                        count=4,
                        order="volatile",
                    )
                    for i in cutlass.range_constexpr(4):
                        words[i, group] = values[i]
                else:
                    pass
                ready &= cutlass.all_(
                    words[None, group].load().to_vector() != PAYLOAD_EMPTY
                )
        total = cutlass.Float32(0)
        for group in cutlass.range_constexpr(8):
            values = (
                words[None, group]
                .load()
                .to_vector()
                .bitcast(cutlass.BFloat16)
                .to(cutlass.Float32)
            )
            squares = values * values
            total += ((squares[0] + squares[1]) + (squares[2] + squares[3])) + (
                (squares[4] + squares[5]) + (squares[6] + squares[7])
            )
        for delta in cutlass.range_constexpr(5):
            total += cute.arch.shuffle_sync_bfly(total, 1 << delta)
        inv = cute.math.rsqrt(total / 2048 + norm_epsilon)
        if thread % 32 == 0:
            inverse_rms.store(inv, idx=row)
        else:
            pass
    p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
    p.barrier_cta_sync()


@cute.kernel
def post_attention_kernel(
    attention: cute.Tensor,
    residual: cute.Tensor,
    wo: cutlass.GridConstant[tensor_map.TensorMap],
    wgu: cutlass.GridConstant[tensor_map.TensorMap],
    wd: cutlass.GridConstant[tensor_map.TensorMap],
    gamma: cute.Tensor,
    r: cute.Tensor,
    activation: cute.Tensor,
    epochs: cute.Tensor,
    output: cute.Tensor,
    batch_size: cutlass.Int32,
    batch_tile: cutlass.Constexpr,
    norm_epsilon: cutlass.Constexpr,
):
    """Use one channel wave: 64 CTAs for output/down and all 96 for gate/up."""
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    cluster, rank = block // 4, block % 4
    weights = cutlass.Array(
        cutlass.BFloat16, 65536, space=cutlass.AddressSpace.smem, alignment=1024
    )
    inverse_rms = cutlass.Array(
        cutlass.Float32, batch_tile, space=cutlass.AddressSpace.smem, alignment=128
    )
    operands = cutlass.Array(
        cutlass.BFloat16,
        2 * batch_tile * 256,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    partial = cutlass.Array(
        cutlass.Float32,
        batch_tile * 128,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    mailbox = cutlass.Array(
        cutlass.Float32,
        batch_tile * 128,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    tmem_address = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    if thread >= 128 and thread < 160:
        p.tcgen05_alloc(tmem_address, 32)
        p.tcgen05_relinquish_alloc_permit()
    else:
        pass
    bar = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    free = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    ready = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    epoch_s = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    if thread == 0:
        epoch_s.store(epochs[block] + 1)
        epochs[block] = epochs[block] + 1
        for phase in cutlass.range_constexpr(2):
            p.mbarrier_init(ready.subview(phase), 1)
            p.mbarrier_arrive_expect_tx(ready.subview(phase), 4 * 32 * batch_tile * 4)
    else:
        pass
    p.fence_mbarrier_init()
    p.barrier_cta_sync()
    slot = epoch_s.load() & 1
    p.barrier_cluster_arrive_relaxed()
    p.barrier_cluster_wait()
    # Double-buffered payloads: reset only the next serialized invocation's slot.
    tile = block
    for j in cutlass.range_constexpr(batch_tile // 8):
        offset = ((1 - slot) * 96 + tile) * batch_tile * 32 + thread + j * 256
        p.store_ext(
            cutlass.Int32(PAYLOAD_EMPTY).ir_value(),
            activation.iterator.raw_ptr() + offset,
            order="volatile",
        )
    accumulator = cute.make_rmem_tensor((batch_tile,), cutlass.Float32)
    if cluster < 16:
        tile = cluster
        project(
            accumulator,
            tmem_address.load(),
            weights,
            operands,
            inverse_rms,
            r,
            gamma,
            bar,
            free,
            wo,
            attention,
            activation,
            thread,
            rank,
            tile,
            slot,
            "output",
            batch_size,
            batch_tile,
        )
        scatter_partial(accumulator, mailbox, ready, 0, thread, rank, batch_tile)
        for j in cutlass.range_constexpr(batch_tile // 8):
            idx = thread + j * 256
            row, column = rank * 32 + idx % 32, idx // 32
            value = cutlass.BFloat16(0)
            if column < batch_size:
                projected = cutlass.BFloat16(
                    reduce_partial(mailbox, row, column, batch_tile)
                )
                value = cutlass.BFloat16(
                    cutlass.Float32(projected)
                    + cutlass.Float32(residual[column, tile * 128 + row])
                )
            else:
                pass
            bits = cutlass.Int32(value.bitcast(cutlass.Uint16))
            if bits == 0x8000:
                bits = cutlass.Int32(0)
            else:
                pass
            neighbor = cute.arch.shuffle_sync_bfly(bits, 1)
            if row % 2 == 0:
                offset = (slot * batch_tile + column) * 1024 + tile * 64 + row // 2
                p.store_ext(
                    (bits | (neighbor << 16)).ir_value(),
                    r.iterator.raw_ptr() + offset,
                    order="volatile",
                )
                reset = ((1 - slot) * batch_tile + column) * 1024 + tile * 64 + row // 2
                p.store_ext(
                    cutlass.Int32(PAYLOAD_EMPTY).ir_value(),
                    r.iterator.raw_ptr() + reset,
                    order="volatile",
                )
            else:
                pass
        # Join the four ranks before reusing their split-K mailboxes.
        p.barrier_cluster_arrive_relaxed()
        p.barrier_cluster_wait()
    else:
        pass
    compute_inverse_rms(r, inverse_rms, thread, slot, batch_tile, norm_epsilon)
    tile = block
    project(
        accumulator,
        tmem_address.load(),
        weights,
        operands,
        inverse_rms,
        r,
        gamma,
        bar,
        free,
        wgu,
        attention,
        activation,
        thread,
        rank,
        tile,
        slot,
        "gate_up",
        batch_size,
        batch_tile,
    )
    if thread >= 128:
        for i in cutlass.range_constexpr(batch_tile):
            row, column = thread - 128, i
            partial.store(
                cutlass.Float32(cutlass.BFloat16(accumulator[i])),
                idx=gate_up_index(row, column, batch_tile),
            )
    else:
        pass
    p.barrier_cta_sync()
    for j in cutlass.range_constexpr(batch_tile // 8):
        idx = thread + j * 256
        channel, batch = 2 * (idx % 32), idx // 32
        packed = cutlass.Int32(0)
        for half in cutlass.range_constexpr(2):
            gate = partial.load(idx=gate_up_index(channel + half, batch, batch_tile))
            up = partial.load(idx=gate_up_index(channel + half + 64, batch, batch_tile))
            silu = cutlass.BFloat16(gate / (1 + cute.math.exp(-gate)))
            value = cutlass.BFloat16(cutlass.Float32(silu) * up)
            bits = cutlass.Int32(value.bitcast(cutlass.Uint16))
            if bits == 0x8000:
                bits = cutlass.Int32(0)
            else:
                pass
            packed |= bits << (16 * half)
        offset = ((slot * 96 + tile) * batch_tile + batch) * 32 + channel // 2
        p.store_ext(
            packed.ir_value(), activation.iterator.raw_ptr() + offset, order="volatile"
        )
    p.barrier_cta_sync()
    if cluster < 16:
        tile = cluster
        project(
            accumulator,
            tmem_address.load(),
            weights,
            operands,
            inverse_rms,
            r,
            gamma,
            bar,
            free,
            wd,
            attention,
            activation,
            thread,
            rank,
            tile,
            slot,
            "down",
            batch_size,
            batch_tile,
        )
        scatter_partial(accumulator, mailbox, ready, 1, thread, rank, batch_tile)
        for j in cutlass.range_constexpr(batch_tile // 8):
            idx = thread + j * 256
            row, column = rank * 32 + idx % 32, idx // 32
            value = cutlass.BFloat16(reduce_partial(mailbox, row, column, batch_tile))
            if column < batch_size:
                offset = (slot * batch_tile + column) * 1024 + tile * 64 + row // 2
                word = p.load_ext(
                    r.iterator.raw_ptr() + offset, dtype=cutlass.Int32, order="volatile"
                )
                skip = cutlass.Uint16((word >> (16 * (row % 2))) & 0xFFFF).bitcast(
                    cutlass.BFloat16
                )
                output[column, tile * 128 + row] = cutlass.BFloat16(
                    cutlass.Float32(value) + cutlass.Float32(skip)
                )
            else:
                pass
        p.barrier_cluster_arrive_relaxed()
        p.barrier_cluster_wait()
    else:
        pass

    if thread >= 128 and thread < 160:
        p.tcgen05_dealloc(cutlass.inttoptr(tmem_address.load(), 6, cutlass.Int32), 32)
    else:
        pass


@cute.jit
def post_attention_host(
    attention: cute.Tensor,
    residual: cute.Tensor,
    wo: cute.Tensor,
    wgu: cute.Tensor,
    wd: cute.Tensor,
    gamma: cute.Tensor,
    r: cute.Tensor,
    activation: cute.Tensor,
    epochs: cute.Tensor,
    output: cute.Tensor,
    stream: cuda.CUstream,
    batch_tile: cutlass.Constexpr,
    norm_epsilon: cutlass.Constexpr,
):
    """Build K-stage tensor maps and launch the cooperative wide schedule."""
    batch_size = cutlass.Int32(attention.shape[0])
    wo_map = tensor_map.create_tensor_map_tiled_from_view(
        wo, (1, 4, 128, 64), swizzle=tensor_map.TensorMapSwizzle.s128b
    )
    wgu_map = tensor_map.create_tensor_map_tiled_from_view(
        wgu, (1, 4, 128, 64), swizzle=tensor_map.TensorMapSwizzle.s128b
    )
    wd_map = tensor_map.create_tensor_map_tiled_from_view(
        wd, (1, 4, 128, 64), swizzle=tensor_map.TensorMapSwizzle.s128b
    )
    post_attention_kernel(
        attention,
        residual,
        wo_map,
        wgu_map,
        wd_map,
        gamma,
        r,
        activation,
        epochs,
        output,
        batch_size,
        batch_tile,
        norm_epsilon,
    ).launch(
        grid=(96, 1, 1),
        block=(256, 1, 1),
        cluster=(4, 1, 1),
        cooperative=True,
        stream=stream,
    )
