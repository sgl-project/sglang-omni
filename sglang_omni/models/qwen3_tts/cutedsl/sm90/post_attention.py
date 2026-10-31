# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Hopper fused output projection and MLP."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.experimental import cuda as tensor_map
from cutlass.experimental import primitives as p

from sglang_omni.models.qwen3_tts.cutedsl.sm90.mma import (
    descriptor,
    fragment_column,
    fragment_row,
    mma,
    swizzle,
)

# Two BF16 negative zeros, reserved for an unpublished aligned 32-bit word.
PAYLOAD_EMPTY = -2147450880


@cute.jit
def wait_barrier(barrier, phase):
    while not p.mbarrier_try_wait_parity(barrier, phase):
        pass


@cute.jit
def prefetch_gate_up(weights, bars, weight_map, tile, thread):
    """Prefetch a resident 128-KiB gate/up tile."""
    if thread == 0:
        p.mbarrier_init(bars, 1)
    else:
        pass
    p.fence_mbarrier_init()
    p.barrier_cta_sync()
    if thread == 0:
        p.mbarrier_expect_tx(bars, 131072)
        p.cp_async_bulk_tensor_shared_cluster_global(
            weights, weight_map.get_ptr(), [0, 0, 0, tile], bars, []
        )
        p.mbarrier_arrive(bars)
    else:
        pass


@cute.jit
def activation_words(
    activation,
    channel_start,
    thread,
    j: cutlass.Constexpr,
    slot,
    batch_tile: cutlass.Constexpr,
):
    """Load eight BF16 values as four independently published 32-bit words."""
    idx = (thread + j * 128) * 8
    slab, row, k = idx // (batch_tile * 64), (idx // 64) % batch_tile, idx % 64
    tile = channel_start // 32 + slab * 2 + k // 32
    offset = ((slot * 96 + tile) * batch_tile + row) * 16 + k % 32 // 2
    return p.load_ext(
        activation.iterator.raw_ptr() + offset,
        dtype=cutlass.Int32,
        count=4,
        order="volatile",
    )


@cute.jit
def residual_words(
    residual, thread, j: cutlass.Constexpr, slot, batch_tile: cutlass.Constexpr
):
    """Load one of four 8-value groups per lane; each warp owns a full row."""
    row, column = thread // 32 + (j // 4) * 8, (thread % 32) * 8 + (j % 4) * 256
    offset = (slot * batch_tile + row) * 512 + column // 2
    return p.load_ext(
        residual.iterator.raw_ptr() + offset,
        dtype=cutlass.Int32,
        count=4,
        order="volatile",
    )


@cute.jit
def gemm(
    accumulator,
    weights,
    operands,
    bars,
    weight_map,
    attention,
    normalized,
    activation,
    epochs,
    epoch,
    thread,
    cluster,
    rank,
    tile,
    projection: cutlass.Constexpr,
    batch_size: cutlass.Int32,
    batch_tile: cutlass.Constexpr,
):
    """Consume resident gate/up operands or stage two split-projection operands."""
    if cutlass.const_expr(projection == "gate_up"):
        if thread >= 128:
            p.wgmma_fence_aligned()
            wait_barrier(bars, 0)
            da_base = descriptor(weights)
            db_base = descriptor(normalized)
            # Descriptor addresses use 16-byte units; a weight K64 slab is 8 KiB.
            for slab in cutlass.range_constexpr(16):
                weight_descriptor = da_base + slab * 512
                input_descriptor = db_base + slab * batch_tile * 8
                for k in cutlass.range_constexpr(4):
                    mma(
                        accumulator,
                        weight_descriptor + 2 * k,
                        input_descriptor + 2 * k,
                        batch_tile,
                    )
            p.wgmma_commit_group_sync_aligned()
            p.wgmma_wait_group_sync_aligned(0)
        else:
            pass
        p.barrier_cta_sync()
    else:
        # Output weights are already in flight from the kernel prologue.
        if cutlass.const_expr(projection == "down"):
            if thread == 0:
                for stage in cutlass.range_constexpr(2):
                    p.mbarrier_init(bars.subview(stage), 1)
            else:
                pass
            p.fence_mbarrier_init()
            p.barrier_cta_sync()
        else:
            pass
        group = 6 if projection == "down" else 4
        if thread < 128:
            for stage in range(2):
                if cutlass.const_expr(projection == "output"):
                    channel_start = rank * 512 + stage * 256
                else:
                    channel_start = stage * 1536 + rank * 384
                if cutlass.const_expr(projection == "down"):  # noqa: SIM102
                    if thread == 0:
                        p.mbarrier_expect_tx(bars.subview(stage), group * 8192)
                        p.cp_async_bulk_tensor_shared_cluster_global(
                            weights.subview(stage * group * 4096),
                            weight_map.get_ptr(),
                            [0, 0, channel_start // 64, cluster],
                            bars.subview(stage),
                            [],
                        )
                    else:
                        pass
                else:
                    pass
                p.barrier_cta_sync(barrier_id=1, thread_count=128)
                if cutlass.const_expr(projection == "down"):
                    # Issue all loads before testing them. Polling each vector
                    # separately serializes L2 latency. Retain completed words.
                    payload = cute.make_rmem_tensor(
                        (4, group * batch_tile // 16), cutlass.Int32
                    )
                    for j in cutlass.range_constexpr(group * batch_tile // 16):
                        words = activation_words(
                            activation, channel_start, thread, j, epoch & 1, batch_tile
                        )
                        for word in cutlass.range_constexpr(4):
                            payload[word, j] = words[word]
                    ready = cutlass.Boolean(True)
                    for j in cutlass.range_constexpr(group * batch_tile // 16):
                        ready &= cutlass.all_(
                            payload[None, j].load().to_vector() != PAYLOAD_EMPTY
                        )
                    while not ready:
                        for j in cutlass.range_constexpr(group * batch_tile // 16):
                            if cutlass.any_(
                                payload[None, j].load().to_vector() == PAYLOAD_EMPTY
                            ):
                                words = activation_words(
                                    activation,
                                    channel_start,
                                    thread,
                                    j,
                                    epoch & 1,
                                    batch_tile,
                                )
                                for word in cutlass.range_constexpr(4):
                                    payload[word, j] = words[word]
                            else:
                                pass
                        ready = cutlass.Boolean(True)
                        for j in cutlass.range_constexpr(group * batch_tile // 16):
                            ready &= cutlass.all_(
                                payload[None, j].load().to_vector() != PAYLOAD_EMPTY
                            )
                else:
                    pass
                for j in cutlass.range_constexpr(group * batch_tile // 16):
                    idx = (thread + j * 128) * 8
                    slab, row, k = (
                        idx // (batch_tile * 64),
                        (idx // 64) % batch_tile,
                        idx % 64,
                    )
                    if cutlass.const_expr(projection == "output"):
                        value = cutlass.full((8,), 0, cutlass.BFloat16)
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
                    else:
                        value = (
                            payload[None, j]
                            .load()
                            .to_vector()
                            .bitcast(cutlass.BFloat16)
                        )
                    operands.store(
                        value,
                        idx=(stage * group + slab) * batch_tile * 64 + swizzle(row, k),
                        alignment=16,
                    )
                p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
                p.barrier_cta_sync(barrier_id=1, thread_count=128)
                if thread == 0:
                    p.mbarrier_arrive(bars.subview(stage))
                else:
                    pass
        else:
            p.wgmma_fence_aligned()
            for stage in range(2):
                wait_barrier(bars.subview(stage), 0)
                da_base = descriptor(weights, stage * group * 4096)
                db_base = descriptor(operands, stage * group * batch_tile * 64)
                for slab in cutlass.range_constexpr(group):
                    weight_descriptor = da_base + slab * 512
                    input_descriptor = db_base + slab * batch_tile * 8
                    for k in cutlass.range_constexpr(4):
                        mma(
                            accumulator,
                            weight_descriptor + 2 * k,
                            input_descriptor + 2 * k,
                            batch_tile,
                        )
                p.wgmma_commit_group_sync_aligned()
                p.wgmma_wait_group_sync_aligned(0)
                if cutlass.const_expr(projection == "output"):  # noqa: SIM102
                    if stage == 0 and rank == 3 and thread == 128:
                        # Scheduling hint only: the norm still polls every
                        # residual word. No payload ordering or release fence.
                        p.store_ext(
                            epoch.ir_value(),
                            epochs.iterator.raw_ptr() + 96 + cluster,
                            order="volatile",
                        )
                    else:
                        pass
                else:
                    pass
        p.barrier_cta_sync()


@cute.jit
def gate_up_index(row, column, batch_tile: cutlass.Constexpr):
    """Keep accumulator pairs adjacent and batch-row epilogue reads conflict-free."""
    rows_per_bank_group = 32 // batch_tile
    batch_swizzle = ((row // rows_per_bank_group) * 8) % batch_tile
    return row * batch_tile + (column ^ batch_swizzle)


@cute.jit
def scatter_partial(
    accumulator,
    partial,
    ready,
    phase: cutlass.Constexpr,
    thread,
    rank,
    batch_tile: cutlass.Constexpr,
):
    """Push partials into each owner's mailbox and await this CTA's inputs."""
    if thread >= 128:
        for i in cutlass.range_constexpr(batch_tile // 2):
            row = fragment_row(thread - 128, i)
            column = fragment_column(thread - 128, i)
            offset = (rank * batch_tile + column) * 16 + row % 16
            cute.arch.store_async_dsmem(
                cute.make_ptr(cutlass.Float32, partial.data_ptr(offset).ir_value()),
                accumulator[i].bitcast(cutlass.Int32),
                cute.make_ptr(cutlass.Int64, ready.data_ptr(phase).ir_value()),
                peer_cta_rank=row // 16,
            )
    else:
        pass
    wait_barrier(ready.subview(phase), 0)


@cute.jit
def reduce_partial(partial, row, column, batch_tile: cutlass.Constexpr):
    """Sum the four sender slots in the original split-K order."""
    value = cutlass.Float32(0)
    for peer in cutlass.range_constexpr(4):
        value += partial.load(idx=(peer * batch_tile + column) * 16 + row % 16)
    return value


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
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    cluster, rank = block // 4, block % 4
    # Gate/up needs 128 KiB of weights but no staged operands; alias that unused space.
    weights = cutlass.Array(
        cutlass.BFloat16,
        max(65536, 49152 + 12 * batch_tile * 64),
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    operands = weights.subview(49152)
    bars = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    # Weight loads need neither the epoch nor the DSMEM mailbox. Start both
    # stages before that setup; the epoch CTA barrier also joins this init.
    if block < 64 and thread == 0:
        p.prefetch_tensormap(wo.get_ptr())
        for stage in cutlass.range_constexpr(2):
            p.mbarrier_init(bars.subview(stage), 1)
        p.fence_mbarrier_init()
        for stage in cutlass.range_constexpr(2):
            p.mbarrier_expect_tx(bars.subview(stage), 32768)
            p.cp_async_bulk_tensor_shared_cluster_global(
                weights.subview(stage * 16384),
                wo.get_ptr(),
                [0, 0, rank * 8 + stage * 4, cluster],
                bars.subview(stage),
                [],
            )
    else:
        pass
    normalized = cutlass.Array(
        cutlass.BFloat16,
        batch_tile * 1024,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    partial = cutlass.Array(
        cutlass.Float32, 64 * batch_tile, space=cutlass.AddressSpace.smem, alignment=128
    )
    # A peer may start down while this CTA still reads its gate/up partials.
    # Keep the incoming mailbox separate from the gate/up epilogue.
    mailbox = cutlass.Array(
        cutlass.Float32, 64 * batch_tile, space=cutlass.AddressSpace.smem, alignment=128
    )
    epoch_s = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
    )
    if thread == 0:
        next_epoch = epochs[block] + 1
        epoch_s.store(next_epoch)
        epochs[block] = next_epoch
    else:
        pass
    p.barrier_cta_sync()
    epoch = epoch_s.load()
    ready = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    if cluster < 16:
        if thread == 0:
            for stage in cutlass.range_constexpr(2):
                p.mbarrier_init(ready.subview(stage), 1)
                p.mbarrier_arrive_expect_tx(
                    ready.subview(stage), 4 * 16 * batch_tile * 4
                )
        else:
            pass
        p.fence_mbarrier_init()
        p.barrier_cluster_arrive_relaxed()
        p.barrier_cluster_wait()
        # Reset only the inactive residual slot for the next serialized call.
        for j in cutlass.range_constexpr(batch_tile // 16):
            idx = thread + j * 256
            batch, channel = idx // 16, idx % 16
            if channel % 2 == 0:
                offset = (
                    ((1 - (epoch & 1)) * batch_tile + batch) * 512
                    + block * 8
                    + channel // 2
                )
                p.store_ext(
                    cutlass.Int32(PAYLOAD_EMPTY).ir_value(),
                    r.iterator.raw_ptr() + offset,
                    order="volatile",
                )
            else:
                pass
    else:
        pass
    # Clear the inactive slot. The next serialized invocation uses it, so no
    # producer can race its reset and no extra reset kernel is needed.
    tile = block
    for j in cutlass.range_constexpr(batch_tile // 16):
        offset = ((1 - (epoch & 1)) * 96 + tile) * batch_tile * 16 + thread + j * 256
        p.store_ext(
            cutlass.Int32(PAYLOAD_EMPTY).ir_value(),
            activation.iterator.raw_ptr() + offset,
            order="volatile",
        )
    accumulator = cute.make_rmem_tensor((batch_tile // 2,), cutlass.Float32)
    accumulator.fill(0)

    if cluster < 16:
        # Split K=2048 over the four ranks; every rank covers the same 64 outputs.
        gemm(
            accumulator,
            weights,
            operands,
            bars,
            wo,
            attention,
            normalized,
            activation,
            epochs,
            epoch,
            thread,
            cluster,
            rank,
            0,
            projection="output",
            batch_size=batch_size,
            batch_tile=batch_tile,
        )
        scatter_partial(accumulator, mailbox, ready, 0, thread, rank, batch_tile)

        # Each rank owns 16 output channels after the cluster reduction.
        for j in cutlass.range_constexpr(batch_tile // 16):
            idx = thread + j * 256
            batch, channel = idx // 16, idx % 16
            row = rank * 16 + channel
            projected = cutlass.BFloat16(
                reduce_partial(mailbox, row, batch, batch_tile)
            )
            value = cutlass.BFloat16(0)
            if batch < batch_size:
                value = cutlass.BFloat16(
                    cutlass.Float32(projected)
                    + cutlass.Float32(residual[batch, cluster * 64 + row])
                )
            else:
                pass
            bits = cutlass.Int32(value.bitcast(cutlass.Uint16))
            if bits == 0x8000:
                bits = cutlass.Int32(0)
            else:
                pass
            # Neighboring lanes own adjacent channels; publish each pair in
            # one aligned word so consumers can test that word independently.
            neighbor = cute.arch.shuffle_sync_bfly(bits, 1)
            if channel % 2 == 0:
                packed = bits | (neighbor << 16)
                offset = (
                    ((epoch & 1) * batch_tile + batch) * 512 + block * 8 + channel // 2
                )
                p.store_ext(
                    packed.ir_value(), r.iterator.raw_ptr() + offset, order="volatile"
                )
            else:
                pass
        p.barrier_cta_sync()
    else:
        pass

    # Every CTA produces one gate/up tile before the output CTAs start down.
    if cluster >= 16:
        if thread == 0:
            # Start prefetch once a producer finishes its first MMA
            # stage, rather than competing with its initial loads.
            ptr = epochs.iterator.raw_ptr() + 96 + (block - 64) % 16
            while p.load_ext(ptr, dtype=cutlass.Int32, order="volatile") != epoch:
                p.nanosleep(32)
        else:
            pass
        # prefetch_gate_up's initialization barrier also joins this wait.
        if thread == 0:
            # These 32 CTAs warm the other 64 tiles for the CTAs still
            # doing output projection. Their own tiles go to SMEM below.
            for tile_offset in cutlass.range_constexpr(2):
                p.cp_async_bulk_tensor_prefetch(
                    wgu.get_ptr(), [0, 0, 0, block - 64 + tile_offset * 32], []
                )
        else:
            pass
    else:
        pass
    prefetch_gate_up(weights, bars, wgu, block, thread)
    norm_inputs = cute.make_rmem_tensor((4, batch_tile // 2), cutlass.Int32)
    norm_scales = cute.make_rmem_tensor((8, 4), cutlass.BFloat16)
    for group in cutlass.range_constexpr(4):
        values = (gamma.iterator.raw_ptr() + (thread % 32) * 8 + group * 256).load(
            count=8, alignment=16
        )
        for k in cutlass.range_constexpr(8):
            norm_scales[k, group] = values[k]
    for j in cutlass.range_constexpr(batch_tile // 2):
        values = residual_words(r, thread, j, epoch & 1, batch_tile)
        for k in cutlass.range_constexpr(4):
            norm_inputs[k, j] = values[k]
    norm_ready = cutlass.Boolean(True)
    for j in cutlass.range_constexpr(batch_tile // 2):
        norm_ready &= cutlass.all_(
            norm_inputs[None, j].load().to_vector() != PAYLOAD_EMPTY
        )
    while not norm_ready:
        for j in cutlass.range_constexpr(batch_tile // 2):
            if cutlass.any_(norm_inputs[None, j].load().to_vector() == PAYLOAD_EMPTY):
                values = residual_words(r, thread, j, epoch & 1, batch_tile)
                for k in cutlass.range_constexpr(4):
                    norm_inputs[k, j] = values[k]
            else:
                pass
        norm_ready = cutlass.Boolean(True)
        for j in cutlass.range_constexpr(batch_tile // 2):
            norm_ready &= cutlass.all_(
                norm_inputs[None, j].load().to_vector() != PAYLOAD_EMPTY
            )
    # Retain polled inputs in registers. Each warp reduces its own rows,
    # with no shared partials or cross-warp inverse-RMS broadcast.
    norm_inverse = cute.make_rmem_tensor((batch_tile // 8,), cutlass.Float32)
    for token in cutlass.range_constexpr(batch_tile // 8):
        total = cutlass.Float32(0)
        for group in cutlass.range_constexpr(4):
            words = norm_inputs[None, token * 4 + group].load().to_vector()
            values = words.bitcast(cutlass.BFloat16).to(cutlass.Float32)
            squares = values * values
            left = (squares[0] + squares[1]) + (squares[2] + squares[3])
            right = (squares[4] + squares[5]) + (squares[6] + squares[7])
            total += left + right
        for delta in cutlass.range_constexpr(5):
            total += cute.arch.shuffle_sync_bfly(total, 1 << delta)
        norm_inverse[token] = cute.math.rsqrt(total / 1024 + norm_epsilon)
    for j in cutlass.range_constexpr(batch_tile // 2):
        row, column = thread // 32 + (j // 4) * 8, (thread % 32) * 8 + (j % 4) * 256
        values = (
            norm_inputs[None, j]
            .load()
            .to_vector()
            .bitcast(cutlass.BFloat16)
            .to(cutlass.Float32)
        )
        scale = norm_scales[None, j % 4].load().to_vector().to(cutlass.Float32)
        unit = (values * norm_inverse[j // 4]).to(cutlass.BFloat16).to(cutlass.Float32)
        offset = (column // 64) * batch_tile * 64 + swizzle(row, column % 64)
        normalized.store((unit * scale).to(cutlass.BFloat16), idx=offset, alignment=16)
    # The only norm CTA rendezvous: all operand rows must reach SMEM
    # before the consumer warpgroup issues WGMMA.
    p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
    p.barrier_cta_sync()
    accumulator.fill(0)
    gemm(
        accumulator,
        weights,
        operands,
        bars,
        wgu,
        attention,
        normalized,
        activation,
        epochs,
        epoch,
        thread,
        cluster,
        rank,
        tile,
        projection="gate_up",
        batch_size=batch_size,
        batch_tile=batch_tile,
    )
    if thread >= 128:
        for j in cutlass.range_constexpr(batch_tile // 2):
            row = fragment_row(thread - 128, j)
            column = fragment_column(thread - 128, j)
            partial.store(
                cutlass.Float32(cutlass.BFloat16(accumulator[j])),
                idx=gate_up_index(row, column, batch_tile),
            )
    else:
        pass
    p.barrier_cta_sync()
    for j in cutlass.range_constexpr(batch_tile // 16):
        idx = thread + j * 256
        channel, batch = 2 * (idx // batch_tile), idx % batch_tile
        # Publish a whole aligned word, not two independent 16-bit
        # stores. Canonicalize -0 so no live word equals the sentinel.
        packed = cutlass.Int32(0)
        for half in cutlass.range_constexpr(2):
            gate = partial.load(idx=gate_up_index(channel + half, batch, batch_tile))
            up = partial.load(idx=gate_up_index(channel + half + 32, batch, batch_tile))
            silu = cutlass.BFloat16(gate / (1 + cute.math.exp(-gate)))
            value = cutlass.BFloat16(cutlass.Float32(silu) * up)
            bits = cutlass.Int32(value.bitcast(cutlass.Uint16))
            if bits == 0x8000:
                bits = cutlass.Int32(0)
            else:
                pass
            packed |= bits << (16 * half)
        offset = (((epoch & 1) * 96 + tile) * batch_tile + batch) * 16 + channel // 2
        p.store_ext(
            packed.ir_value(), activation.iterator.raw_ptr() + offset, order="volatile"
        )
    p.barrier_cta_sync()

    if cluster < 16:
        accumulator.fill(0)

        # Down waits only for its own activation tiles from the other CTAs.
        gemm(
            accumulator,
            weights,
            operands,
            bars,
            wd,
            attention,
            normalized,
            activation,
            epochs,
            epoch,
            thread,
            cluster,
            rank,
            0,
            projection="down",
            batch_size=batch_size,
            batch_tile=batch_tile,
        )
        # note (Hao Wang): Down needs every residual publication, each following its phase-0 mailbox read, so reuse needs no extra barrier.
        scatter_partial(accumulator, mailbox, ready, 1, thread, rank, batch_tile)
        for j in cutlass.range_constexpr(batch_tile // 16):
            idx = thread + j * 256
            row, column = rank * 16 + idx % 16, idx // 16
            value = cutlass.BFloat16(reduce_partial(mailbox, row, column, batch_tile))
            if column < batch_size:
                offset = (
                    ((epoch & 1) * batch_tile + column) * 512 + cluster * 32 + row // 2
                )
                word = p.load_ext(
                    r.iterator.raw_ptr() + offset, dtype=cutlass.Int32, order="volatile"
                )
                skip = cutlass.Uint16((word >> (16 * (row % 2))) & 0xFFFF).bitcast(
                    cutlass.BFloat16
                )
                output[column, cluster * 64 + row] = cutlass.BFloat16(
                    cutlass.Float32(value) + cutlass.Float32(skip)
                )
            else:
                pass
        # Every mailbox is consumed; only the cluster lifetime rendezvous remains.
        p.barrier_cluster_arrive_relaxed()
        p.barrier_cluster_wait()
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
    batch_size = cutlass.Int32(attention.shape[0])
    wo_map = tensor_map.create_tensor_map_tiled_from_view(
        wo, (1, 4, 64, 64), swizzle=tensor_map.TensorMapSwizzle.s128b
    )
    wgu_map = tensor_map.create_tensor_map_tiled_from_view(
        wgu, (1, 16, 64, 64), swizzle=tensor_map.TensorMapSwizzle.s128b
    )
    wd_map = tensor_map.create_tensor_map_tiled_from_view(
        wd, (1, 6, 64, 64), swizzle=tensor_map.TensorMapSwizzle.s128b
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
