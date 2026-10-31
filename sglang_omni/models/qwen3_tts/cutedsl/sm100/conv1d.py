# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Persistent BF16 causal Conv1D with tiled TMA and double-buffered accumulators."""

import cutlass
from cuda.bindings import driver
from cutlass import cute
from cutlass.experimental import cuda
from cutlass.experimental import primitives as p


@cute.kernel
def conv1d_kernel(
    operation: cutlass.Constexpr,
    input_map: cutlass.GridConstant[cuda.TensorMap],
    weight_map: cutlass.GridConstant[cuda.TensorMap],
    output_map: cutlass.GridConstant[cuda.TensorMap],
    bias: cute.Tensor,
    residual: cute.Tensor,
    alpha: cute.Tensor,
    inverse_beta: cute.Tensor,
    sine_table: cute.Tensor,
    batch_size: cutlass.Int32,
    sequence_length: cutlass.Int32,
    input_channels: cutlass.Constexpr,
    output_channels: cutlass.Constexpr,
    kernel_size: cutlass.Constexpr,
    dilation: cutlass.Constexpr,
    channel_tile: cutlass.Constexpr,
    stages: cutlass.Constexpr,
    reduction_tile: cutlass.Constexpr,
    store_channels: cutlass.Constexpr,
    full_rows: cutlass.Constexpr,
) -> None:
    """Warps 0-3 store results, warp 4 issues MMA, warp 5 loads operands."""
    thread_index, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(thread_index // 32)
    tile_index, _, _ = cute.arch.block_idx()
    grid_blocks, _, _ = cute.arch.grid_dim()
    sequence_tiles = (sequence_length + 127) // 128
    channel_tiles: cutlass.Constexpr = (
        output_channels + channel_tile - 1
    ) // channel_tile
    output_tiles = batch_size * sequence_tiles * channel_tiles
    channel_steps: cutlass.Constexpr = (
        input_channels + reduction_tile - 1
    ) // reduction_tile
    reduction_steps: cutlass.Constexpr = kernel_size * channel_steps
    input_stage_bytes: cutlass.Constexpr = 128 * reduction_tile * 2
    weight_stage_bytes: cutlass.Constexpr = channel_tile * reduction_tile * 2
    tmem_columns: cutlass.Constexpr = 1 << (2 * channel_tile - 1).bit_length()
    load_channels: cutlass.Constexpr = 64 if store_channels == 64 else 32
    full = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem)
    empty = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem)
    acc_full = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem)
    acc_empty = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem)
    tmem_address = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    a = cutlass.Array(
        cutlass.Int8,
        input_stage_bytes * stages,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    b = cutlass.Array(
        cutlass.Int8,
        weight_stage_bytes * stages,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    c = cutlass.Array(
        cutlass.BFloat16,
        2 * 128 * store_channels,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    if thread_index == 0:
        p.prefetch_tensormap(input_map.get_ptr())
        p.prefetch_tensormap(weight_map.get_ptr())
        p.prefetch_tensormap(output_map.get_ptr())
        for i in cutlass.range_constexpr(stages):
            p.mbarrier_init(full.subview(i), 1)
            p.mbarrier_init(empty.subview(i), 1)
        for i in cutlass.range_constexpr(2):
            p.mbarrier_init(acc_full.subview(i), 1)
            p.mbarrier_init(acc_empty.subview(i), 4)
        p.fence_mbarrier_init()
    else:
        pass
    if warp == 0:
        p.tcgen05_alloc(tmem_address, tmem_columns)
        p.tcgen05_relinquish_alloc_permit()
    else:
        pass
    p.barrier_cta_sync(0, thread_count=192)
    stage = cutlass.Int32(0)
    phase = cutlass.Int32(0)
    acc_stage = cutlass.Int32(0)
    acc_phase = cutlass.Int32(0)
    if warp == 5:
        if p.elect_sync():
            while tile_index < output_tiles:
                batch_index = tile_index // (sequence_tiles * channel_tiles)
                row = tile_index // channel_tiles % sequence_tiles * 128
                col = tile_index % channel_tiles * channel_tile
                for k in cutlass.range(reduction_steps, unroll=1):
                    while not p.mbarrier_try_wait_parity(
                        empty.subview(stage), phase ^ 1
                    ):
                        pass
                    channel = k % channel_steps * reduction_tile
                    tap = k // channel_steps
                    p.mbarrier_arrive_expect_tx(
                        full.subview(stage), input_stage_bytes + weight_stage_bytes
                    )
                    p.cp_async_bulk_tensor_shared_cluster_global(
                        a.subview(stage * input_stage_bytes),
                        input_map.get_ptr(),
                        (
                            channel,
                            row + (tap - kernel_size + 1) * dilation,
                            batch_index,
                        ),
                        full.subview(stage),
                        [],
                    )
                    p.cp_async_bulk_tensor_shared_cluster_global(
                        b.subview(stage * weight_stage_bytes),
                        weight_map.get_ptr(),
                        (channel, tap, col),
                        full.subview(stage),
                        [],
                    )
                    stage += 1
                    if stage == stages:
                        stage = 0
                        phase ^= 1
                    else:
                        pass
                tile_index += grid_blocks
        else:
            pass
    elif warp == 4:
        p.tcgen05_fence(p.Tcgen05Fence.AFTER_THREAD_SYNC)
        desc_a = p.Tcgen05SmemDesc.build(
            a,
            leading_byte_offset=16,
            stride_byte_offset=reduction_tile * 16,
            layout=2 if reduction_tile == 64 else 4,
        )
        desc_b = p.Tcgen05SmemDesc.build(
            b,
            leading_byte_offset=16,
            stride_byte_offset=reduction_tile * 16,
            layout=2 if reduction_tile == 64 else 4,
        )
        instruction = p.Tcgen05InstrDesc.build(
            a_dtype=cutlass.BFloat16,
            b_dtype=cutlass.BFloat16,
            c_dtype=cutlass.Float32,
            m_dim=128,
            n_dim=channel_tile,
        )
        while tile_index < output_tiles:
            while not p.mbarrier_try_wait_parity(
                acc_empty.subview(acc_stage), acc_phase ^ 1
            ):
                pass
            p.tcgen05_fence(p.Tcgen05Fence.AFTER_THREAD_SYNC)
            destination = cutlass.Array(
                p.make_tmem_ptr(tmem_address.load(), cutlass.Float32).data_ptr()
                + acc_stage * channel_tile,
                dtype=cutlass.Int32,
                addrspace=6,
            )
            for k in cutlass.range(reduction_steps, unroll=1):
                while not p.mbarrier_try_wait_parity(full.subview(stage), phase):
                    pass
                p.tcgen05_fence(p.Tcgen05Fence.AFTER_THREAD_SYNC)
                for subk in cutlass.range_constexpr(reduction_tile // 16):
                    if p.elect_sync():
                        p.tcgen05_mma(
                            p.Tcgen05MMAKind.F16,
                            p.CTAGroup.CTA_1,
                            destination,
                            desc_a + stage * (input_stage_bytes // 16) + subk * 2,
                            desc_b + stage * (weight_stage_bytes // 16) + subk * 2,
                            instruction,
                            (k > 0) | (subk > 0),
                        )
                    else:
                        pass
                if p.elect_sync():
                    p.tcgen05_commit(empty.subview(stage))
                else:
                    pass
                stage += 1
                if stage == stages:
                    stage = 0
                    phase ^= 1
                else:
                    pass
            if p.elect_sync():
                p.tcgen05_commit(acc_full.subview(acc_stage))
            else:
                pass
            acc_stage ^= 1
            if acc_stage == 0:
                acc_phase ^= 1
            else:
                pass
            tile_index += grid_blocks
    else:
        address = tmem_address.load()
        store_stage = cutlass.Int32(0)
        issuer = p.elect_sync()
        while tile_index < output_tiles:
            batch_index = tile_index // (sequence_tiles * channel_tiles)
            row = tile_index // channel_tiles % sequence_tiles * 128 + thread_index
            col = tile_index % channel_tiles * channel_tile
            while not p.mbarrier_try_wait_parity(
                acc_full.subview(acc_stage), acc_phase
            ):
                pass
            p.tcgen05_fence(p.Tcgen05Fence.AFTER_THREAD_SYNC)
            for chunk in cutlass.range_constexpr(channel_tile // store_channels):
                for fragment in cutlass.range_constexpr(
                    store_channels // load_channels
                ):
                    column = chunk * store_channels + fragment * load_channels
                    pointer = p.make_tmem_ptr_from_warp_row_col(
                        address,
                        warp,
                        (address & 65535) + acc_stage * channel_tile + column,
                        cutlass.Float32,
                    )
                    values = p.tcgen05_ld("32x32b", pointer, num=load_channels)
                    p.tcgen05_wait(p.Tcgen05Wait.LOAD)
                    for v in cutlass.range_constexpr(load_channels // 8):
                        channel = col + column + v * 8
                        rounded = cutlass.full((8,), 0, cutlass.BFloat16)
                        if (
                            full_rows or row < sequence_length
                        ) and channel < output_channels:
                            result = values[v * 8 : v * 8 + 8]
                            offset = (
                                cutlass.Int64(batch_index) * sequence_length + row
                            ) * output_channels + channel
                            if cutlass.const_expr(bias is not None):
                                result = result.to(cutlass.BFloat16).to(
                                    cutlass.Float32
                                ) + (
                                    (bias.iterator.raw_ptr() + channel)
                                    .load(count=8, alignment=16)
                                    .to(cutlass.Float32)
                                )
                            else:
                                pass
                            rounded = result.to(cutlass.BFloat16)
                            if cutlass.const_expr(residual is not None):
                                rounded += (residual.iterator.raw_ptr() + offset).load(
                                    count=8, alignment=16
                                )
                            else:
                                pass
                            if cutlass.const_expr(alpha is not None):
                                scale = (alpha.iterator.raw_ptr() + channel).load(
                                    count=8, alignment=16
                                )
                                inverse = (
                                    inverse_beta.iterator.raw_ptr() + channel
                                ).load(count=8, alignment=16)
                                scaled = rounded * scale
                                indices = scaled.bitcast(cutlass.Uint16).to(
                                    cutlass.Int32
                                )
                                sine = sine_table.iterator.raw_ptr().masked_gather(
                                    indices, cutlass.full((8,), True, cutlass.Boolean)
                                )
                                square = sine * sine
                                rounded += square * inverse
                            else:
                                pass
                        else:
                            pass
                        c.subview(
                            store_stage * 128 * store_channels
                            + thread_index * store_channels
                            + fragment * load_channels
                            + v * 8
                        ).data_ptr().store_swizzled(
                            rounded,
                            alignment=16,
                            swizzle=cutlass.Swizzle(
                                3 if store_channels == 64 else 2, 4, 3
                            ),
                        )
                cute.arch.fence_view_async_shared()
                p.barrier_cta_sync(2, thread_count=128)
                if warp == 0 and issuer:
                    if cutlass.const_expr(store_channels == 96):
                        coordinates = (
                            0,
                            (col + chunk * store_channels) // 32,
                            tile_index // channel_tiles % sequence_tiles * 128,
                            batch_index,
                        )
                    else:
                        coordinates = (
                            col + chunk * store_channels,
                            tile_index // channel_tiles % sequence_tiles * 128,
                            batch_index,
                        )
                    p.cp_async_bulk_tensor_global_shared_cta(
                        output_map.get_ptr(),
                        c.subview(store_stage * 128 * store_channels),
                        coordinates,
                    )
                    p.cp_async_bulk_commit_group()
                    p.cp_async_bulk_wait_group(1, read=True)
                else:
                    pass
                p.barrier_cta_sync(2, thread_count=128)
                store_stage ^= 1
            p.tcgen05_fence(p.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if p.elect_sync():
                p.mbarrier_arrive(acc_empty.subview(acc_stage))
            else:
                pass
            acc_stage ^= 1
            if acc_stage == 0:
                acc_phase ^= 1
            else:
                pass
            tile_index += grid_blocks
        if warp == 0 and issuer:
            p.cp_async_bulk_wait_group(0, read=False)
        else:
            pass
    # note (Hao Wang): All producer/consumer work must finish before freeing TMEM.
    p.barrier_cta_sync(0, thread_count=192)
    if warp == 0:
        p.tcgen05_dealloc(
            p.make_tmem_ptr(tmem_address.load(), cutlass.Float32), tmem_columns
        )
    else:
        pass


class Conv1dLaunch:
    """Bind tiled TMA maps for M128 convolution tiles."""

    def __init__(
        self,
        weight_shape: tuple[int, int, int],
        dilation: int,
        channel_tile: int,
        stages: int,
        reduction_tile: int,
        store_channels: int,
        full_rows: bool,
    ) -> None:
        self.weight_shape = weight_shape
        self.full_rows = full_rows
        self.dilation = dilation
        self.channel_tile = channel_tile
        self.stages = stages
        self.reduction_tile, self.store_channels = reduction_tile, store_channels

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        bias: cute.Tensor,
        residual: cute.Tensor,
        alpha: cute.Tensor,
        inverse_beta: cute.Tensor,
        sine_table: cute.Tensor,
        stream: driver.CUstream,
        batch_size: cutlass.Int32,
        sequence_length: cutlass.Int32,
        thread_blocks: cutlass.Int32,
    ) -> None:
        input_channels = self.weight_shape[2]
        reduction_tile, store_channels = self.reduction_tile, self.store_channels
        output_channels, kernel_size, _ = self.weight_shape
        a = cuda.create_tensor_map_tiled(
            global_address=x.iterator.toint(),
            dtype=cutlass.BFloat16,
            global_dims=(input_channels, sequence_length, batch_size),
            global_strides=(input_channels // 8, sequence_length * input_channels // 8),
            box_dims=(reduction_tile, 128, 1),
            swizzle=(
                cuda.TensorMapSwizzle.s128b
                if reduction_tile == 64
                else cuda.TensorMapSwizzle.s64b
            ),
        )
        b = cuda.create_tensor_map_tiled(
            global_address=weight.iterator.toint(),
            dtype=cutlass.BFloat16,
            global_dims=(input_channels, kernel_size, output_channels),
            global_strides=(input_channels // 8, kernel_size * input_channels // 8),
            box_dims=(reduction_tile, 1, self.channel_tile),
            swizzle=(
                cuda.TensorMapSwizzle.s128b
                if reduction_tile == 64
                else cuda.TensorMapSwizzle.s64b
            ),
        )
        # note (Hao Wang): Group three 32-channel blocks to store N96 with a 64-byte swizzle.
        c = cuda.create_tensor_map_tiled(
            global_address=output.iterator.toint(),
            dtype=cutlass.BFloat16,
            global_dims=(
                (32, output_channels // 32, sequence_length, batch_size)
                if store_channels == 96
                else (output_channels, sequence_length, batch_size)
            ),
            global_strides=(
                (4, output_channels // 8, sequence_length * output_channels // 8)
                if store_channels == 96
                else (output_channels // 8, sequence_length * output_channels // 8)
            ),
            box_dims=(
                (32, 3, 128, 1) if store_channels == 96 else (store_channels, 128, 1)
            ),
            swizzle=(
                cuda.TensorMapSwizzle.s128b
                if store_channels == 64
                else cuda.TensorMapSwizzle.s64b
            ),
        )
        conv1d_kernel(
            f"c{input_channels}_c{output_channels}_k{kernel_size}_d{self.dilation}_"
            f"{'residual' if residual is not None else 'bias'}_{'snake' if alpha is not None else 'raw'}",
            a,
            b,
            c,
            bias,
            residual,
            alpha,
            inverse_beta,
            sine_table,
            batch_size,
            sequence_length,
            input_channels,
            output_channels,
            kernel_size,
            self.dilation,
            self.channel_tile,
            self.stages,
            reduction_tile,
            store_channels,
            self.full_rows,
        ).launch(grid=(thread_blocks, 1, 1), block=(192, 1, 1), stream=stream)
