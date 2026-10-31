# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Hopper causal Conv1D with two alternating WGMMA consumer warpgroups."""

import cutlass
from cuda.bindings import driver
from cutlass import cute
from cutlass.experimental import cuda
from cutlass.experimental import primitives as p

from sglang_omni.models.qwen3_tts.cutedsl.sm90.mma import (
    fragment_column,
    fragment_row,
    mma,
)


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
    """Warps 0-3/4-7 alternate output tiles; warp 8 stages operands for both."""
    thread_index, _, _ = cute.arch.thread_idx()
    block_index, _, _ = cute.arch.block_idx()
    grid_blocks, _, _ = cute.arch.grid_dim()
    warpgroup_index = cute.arch.make_warp_uniform(thread_index // 128)
    warpgroup_thread = thread_index % 128
    sequence_tiles = (sequence_length + 63) // 64
    channel_tiles: cutlass.Constexpr = (
        output_channels + channel_tile - 1
    ) // channel_tile
    output_tiles = batch_size * sequence_tiles * channel_tiles
    channel_steps: cutlass.Constexpr = (
        input_channels + reduction_tile - 1
    ) // reduction_tile
    reduction_steps: cutlass.Constexpr = kernel_size * channel_steps
    input_stage_bytes: cutlass.Constexpr = 64 * reduction_tile * 2
    weight_stage_bytes: cutlass.Constexpr = channel_tile * reduction_tile * 2
    full = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem)
    empty = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem)
    mma_done = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem)
    input_shared = cutlass.Array(
        cutlass.Int8,
        input_stage_bytes * stages,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    weight_shared = cutlass.Array(
        cutlass.Int8,
        weight_stage_bytes * stages,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    output = cutlass.Array(
        cutlass.BFloat16,
        2 * 2 * 64 * store_channels,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    if thread_index == 0:
        p.prefetch_tensormap(input_map.get_ptr())
        p.prefetch_tensormap(weight_map.get_ptr())
        p.prefetch_tensormap(output_map.get_ptr())
        for stage in cutlass.range_constexpr(stages):
            p.mbarrier_init(full.subview(stage), 1)
            p.mbarrier_init(empty.subview(stage), 1)
        for index in cutlass.range_constexpr(2):
            p.mbarrier_init(mma_done.subview(index), 1)
        p.fence_mbarrier_init()
    else:
        pass
    p.barrier_cta_sync(0, thread_count=288)
    if warpgroup_index == 2:
        if p.elect_sync():
            cursor = cutlass.Int32(0)
            tile_index = block_index
            while tile_index < output_tiles:
                batch_index = tile_index // (sequence_tiles * channel_tiles)
                row = tile_index // channel_tiles % sequence_tiles * 64
                column = tile_index % channel_tiles * channel_tile
                for k in cutlass.range(reduction_steps, unroll=1):
                    stage = cursor % stages
                    while not p.mbarrier_try_wait_parity(
                        empty.subview(stage), (cursor // stages % 2) ^ 1
                    ):
                        pass
                    p.mbarrier_arrive_expect_tx(
                        full.subview(stage), input_stage_bytes + weight_stage_bytes
                    )
                    p.cp_async_bulk_tensor_shared_cluster_global(
                        input_shared.subview(stage * input_stage_bytes),
                        input_map.get_ptr(),
                        (
                            k % channel_steps * reduction_tile,
                            row + (k // channel_steps - kernel_size + 1) * dilation,
                            batch_index,
                        ),
                        full.subview(stage),
                        [],
                    )
                    p.cp_async_bulk_tensor_shared_cluster_global(
                        weight_shared.subview(stage * weight_stage_bytes),
                        weight_map.get_ptr(),
                        (
                            k % channel_steps * reduction_tile,
                            k // channel_steps,
                            column,
                        ),
                        full.subview(stage),
                        [],
                    )
                    cursor += 1
                tile_index += grid_blocks
        else:
            pass
    else:
        cursor = warpgroup_index * reduction_steps
        phase = warpgroup_index ^ 1
        store_stage = cutlass.Int32(0)
        tile_index = block_index + warpgroup_index * grid_blocks
        # note (Hao Wang): K-major swizzle atoms contain eight rows of reduction_tile BF16 elements.
        descriptor_bits: cutlass.Constexpr = (
            (1 << 16)
            | (reduction_tile << 32)
            | ((1 if reduction_tile == 64 else 2) << 62)
        )
        input_descriptor = (
            cutlass.Int64(input_shared.data_ptr().toint()) >> 4
        ) | descriptor_bits
        weight_descriptor = (
            cutlass.Int64(weight_shared.data_ptr().toint()) >> 4
        ) | descriptor_bits
        accumulator = cute.make_rmem_tensor((channel_tile // 2,), cutlass.Float32)
        while tile_index < output_tiles:
            # note (Hao Wang): Ordered MMA consumption prevents ring-generation aliasing while the other warpgroup stores its output.
            while not p.mbarrier_try_wait_parity(
                mma_done.subview(1 - warpgroup_index), phase
            ):
                pass
            accumulator.fill(0)
            p.wgmma_fence_aligned()
            for k in cutlass.range(reduction_steps, unroll=1):
                stage = cursor % stages
                while not p.mbarrier_try_wait_parity(
                    full.subview(stage), cursor // stages % 2
                ):
                    pass
                for subk in cutlass.range_constexpr(reduction_tile // 16):
                    mma(
                        accumulator,
                        input_descriptor + stage * (input_stage_bytes // 16) + subk * 2,
                        weight_descriptor
                        + stage * (weight_stage_bytes // 16)
                        + subk * 2,
                        channel_tile,
                    )
                p.wgmma_commit_group_sync_aligned()
                p.wgmma_wait_group_sync_aligned(1)
                if k > 0 and warpgroup_thread == 0:
                    p.mbarrier_arrive(empty.subview((cursor - 1) % stages))
                else:
                    pass
                cursor += 1
            p.wgmma_wait_group_sync_aligned(0)
            if warpgroup_thread == 0:
                p.mbarrier_arrive(empty.subview((cursor - 1) % stages))
                p.mbarrier_arrive(mma_done.subview(warpgroup_index))
            else:
                pass
            batch_index = tile_index // (sequence_tiles * channel_tiles)
            row_base = tile_index // channel_tiles % sequence_tiles * 64
            col_base = tile_index % channel_tiles * channel_tile
            for sub in cutlass.range_constexpr(channel_tile // store_channels):
                for pair in cutlass.range_constexpr(store_channels // 4):
                    index = sub * (store_channels // 2) + pair * 2
                    row_local = fragment_row(warpgroup_thread, index)
                    column = fragment_column(warpgroup_thread, index)
                    row, column = row_base + row_local, col_base + column
                    rounded = cutlass.full((2,), 0, cutlass.BFloat16)
                    if (
                        full_rows or row < sequence_length
                    ) and column < output_channels:
                        values = cute.make_rmem_tensor((2,), cutlass.Float32)
                        values[0], values[1] = (
                            accumulator[index],
                            accumulator[index + 1],
                        )
                        rounded = values.load().to_vector().to(cutlass.BFloat16)
                        if cutlass.const_expr(bias is not None):
                            rounded += (bias.iterator.raw_ptr() + column).load(
                                count=2, alignment=4
                            )
                        else:
                            pass
                        if cutlass.const_expr(residual is not None):
                            offset = (
                                cutlass.Int64(batch_index) * sequence_length + row
                            ) * output_channels + column
                            rounded += (residual.iterator.raw_ptr() + offset).load(
                                count=2, alignment=4
                            )
                        else:
                            pass
                        if cutlass.const_expr(alpha is not None):
                            scale = (alpha.iterator.raw_ptr() + column).load(
                                count=2, alignment=4
                            )
                            inverse = (inverse_beta.iterator.raw_ptr() + column).load(
                                count=2, alignment=4
                            )
                            scaled = rounded * scale
                            sine = sine_table.iterator.raw_ptr().masked_gather(
                                scaled.bitcast(cutlass.Uint16).to(cutlass.Int32),
                                cutlass.full((2,), True, cutlass.Boolean),
                            )
                            square = sine * sine
                            rounded += square * inverse
                        else:
                            pass
                    else:
                        pass
                    output.subview(
                        warpgroup_index * 128 * store_channels
                        + store_stage * 64 * store_channels
                        + row_local * store_channels
                        + column % store_channels
                    ).data_ptr().store_swizzled(
                        rounded,
                        alignment=4,
                        swizzle=cutlass.Swizzle(3 if store_channels == 64 else 2, 4, 3),
                    )
                cute.arch.fence_view_async_shared()
                p.barrier_cta_sync(warpgroup_index + 1, thread_count=128)
                if warpgroup_thread == 0:
                    if cutlass.const_expr(store_channels == 96):
                        coordinates = (
                            0,
                            (col_base + sub * store_channels) // 32,
                            row_base,
                            batch_index,
                        )
                    else:
                        coordinates = (
                            col_base + sub * store_channels,
                            row_base,
                            batch_index,
                        )
                    p.cp_async_bulk_tensor_global_shared_cta(
                        output_map.get_ptr(),
                        output.subview(
                            warpgroup_index * 128 * store_channels
                            + store_stage * 64 * store_channels
                        ),
                        coordinates,
                    )
                    p.cp_async_bulk_commit_group()
                    p.cp_async_bulk_wait_group(1, read=True)
                else:
                    pass
                p.barrier_cta_sync(warpgroup_index + 1, thread_count=128)
                store_stage ^= 1
            cursor += reduction_steps
            phase ^= 1
            tile_index += 2 * grid_blocks
        if warpgroup_thread == 0:
            p.cp_async_bulk_wait_group(0, read=False)
        else:
            pass
    p.barrier_cta_sync(0, thread_count=288)


class Conv1dLaunch:
    """Bind tiled TMA maps for M64 convolution tiles."""

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
        self.dilation, self.channel_tile, self.stages = dilation, channel_tile, stages
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
        length: cutlass.Int32,
        thread_blocks: cutlass.Int32,
    ) -> None:
        reduction_tile, store_channels = self.reduction_tile, self.store_channels
        output_channels, kernel, input_channels = self.weight_shape
        input_shared = cuda.create_tensor_map_tiled(
            global_address=x.iterator.toint(),
            dtype=cutlass.BFloat16,
            global_dims=(input_channels, length, batch_size),
            global_strides=(input_channels // 8, length * input_channels // 8),
            box_dims=(reduction_tile, 64, 1),
            swizzle=(
                cuda.TensorMapSwizzle.s128b
                if reduction_tile == 64
                else cuda.TensorMapSwizzle.s64b
            ),
        )
        weight_shared = cuda.create_tensor_map_tiled(
            global_address=weight.iterator.toint(),
            dtype=cutlass.BFloat16,
            global_dims=(input_channels, kernel, output_channels),
            global_strides=(input_channels // 8, kernel * input_channels // 8),
            box_dims=(reduction_tile, 1, self.channel_tile),
            swizzle=(
                cuda.TensorMapSwizzle.s128b
                if reduction_tile == 64
                else cuda.TensorMapSwizzle.s64b
            ),
        )
        # note (Hao Wang): Group three 32-channel blocks to store N96 with a 64-byte swizzle.
        output_map = cuda.create_tensor_map_tiled(
            global_address=output.iterator.toint(),
            dtype=cutlass.BFloat16,
            global_dims=(
                (32, output_channels // 32, length, batch_size)
                if store_channels == 96
                else (output_channels, length, batch_size)
            ),
            global_strides=(
                (4, output_channels // 8, length * output_channels // 8)
                if store_channels == 96
                else (output_channels // 8, length * output_channels // 8)
            ),
            box_dims=(
                (32, 3, 64, 1) if store_channels == 96 else (store_channels, 64, 1)
            ),
            swizzle=(
                cuda.TensorMapSwizzle.s128b
                if store_channels == 64
                else cuda.TensorMapSwizzle.s64b
            ),
        )
        conv1d_kernel(
            f"c{input_channels}_c{output_channels}_k{kernel}_d{self.dilation}_"
            f"{'residual' if residual is not None else 'bias'}_{'snake' if alpha is not None else 'raw'}",
            input_shared,
            weight_shared,
            output_map,
            bias,
            residual,
            alpha,
            inverse_beta,
            sine_table,
            batch_size,
            length,
            input_channels,
            output_channels,
            kernel,
            self.dilation,
            self.channel_tile,
            self.stages,
            reduction_tile,
            store_channels,
            self.full_rows,
        ).launch(grid=(thread_blocks, 1, 1), block=(288, 1, 1), stream=stream)
