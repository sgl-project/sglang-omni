# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Gather and scatter BF16 codec-state rows with one vectorized copy launch."""

import math
from functools import cache

import cutlass
import torch
from cuda.bindings import driver
from cutlass import cute
from cutlass.base_dsl.jit_executor import JitCompiledFunction
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSKVRings,
)


@cute.kernel
def state_copy_kernel(
    storage: tuple[cute.Tensor, ...],
    rows: tuple[cute.Tensor, ...],
    indices: cute.Tensor,
    widths: cutlass.Constexpr,
    scatter: cutlass.Constexpr,
) -> None:
    tile, row, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    slot = indices[row]
    owns_slot = True
    if cutlass.const_expr(scatter):
        # note (Hao Wang): Padded graph rows share a scratch slot; only one row writes it.
        for first_previous_row in cutlass.range(0, row, 32):
            previous_row = first_previous_row + thread % 32
            is_duplicate = False
            if previous_row < row:
                is_duplicate = indices[previous_row] == slot
            else:
                pass
            owns_slot = owns_slot & ~cute.arch.vote_any_sync(is_duplicate)
    else:
        pass
    first_tile = 0
    for buffer_index in cutlass.range_constexpr(len(widths)):
        width = widths[buffer_index]
        tiles = (width + 1023) // 1024
        if owns_slot and tile >= first_tile and tile < first_tile + tiles:
            offset = (tile - first_tile) * 1024 + thread * 8
            if offset < width:
                storage_pointer = (
                    storage[buffer_index].iterator.raw_ptr() + slot * width + offset
                )
                row_pointer = (
                    rows[buffer_index].iterator.raw_ptr()
                    + cutlass.Int64(row) * width
                    + offset
                )
                if cutlass.const_expr(scatter):
                    storage_pointer.store(
                        row_pointer.load(count=8, alignment=16), alignment=16
                    )
                else:
                    row_pointer.store(
                        storage_pointer.load(count=8, alignment=16), alignment=16
                    )
            else:
                pass
        else:
            pass
        first_tile += tiles


@cute.jit
def state_copy_host(
    storage: tuple,
    rows: tuple,
    indices: cute.Tensor,
    stream: driver.CUstream,
    widths: cutlass.Constexpr,
    scatter: cutlass.Constexpr,
) -> None:
    # note (Hao Wang): TVM FFI infers tuple elements; variadic tuple annotations are not supported.
    tiles = 0
    for buffer_index in cutlass.range_constexpr(len(widths)):
        tiles += (widths[buffer_index] + 1023) // 1024
    state_copy_kernel(storage, rows, indices, widths, scatter).launch(
        grid=(tiles, indices.shape[0], 1), block=(128, 1, 1), stream=stream
    )


@cache
def compile_kernel(
    widths: tuple[int, ...], scatter: bool, architecture: str
) -> JitCompiledFunction:
    slots, batch = cutlass.sym_int64(), cutlass.sym_int64()
    storage = tuple(
        make_fake_compact_tensor(
            cutlass.BFloat16, (slots, width), stride_order=(1, 0), assumed_align=16
        )
        for width in widths
    )
    rows = tuple(
        make_fake_compact_tensor(
            cutlass.BFloat16, (batch, width), stride_order=(1, 0), assumed_align=16
        )
        for width in widths
    )
    indices = make_fake_compact_tensor(cutlass.Int64, (batch,), stride_order=(0,))
    return cute.compile(
        state_copy_host,
        storage,
        rows,
        indices,
        make_fake_stream(),
        widths,
        scatter,
        options=f"--enable-tvm-ffi --gpu-arch={architecture}",
    )


def copy_state_rows(
    storage: tuple[torch.Tensor, ...],
    rows: tuple[torch.Tensor, ...],
    indices: torch.Tensor,
    *,
    scatter: bool,
) -> None:
    """Copy BF16 rows; scatter uses the first occurrence of each destination index."""
    widths = tuple(math.prod(buffer.shape[1:]) for buffer in storage)
    assert len(storage) == len(rows) > 0
    assert (
        indices.dtype == torch.int64 and indices.ndim == 1 and indices.is_contiguous()
    )
    for source, target, width in zip(storage, rows, widths):
        assert width > 0 and width % 8 == 0
        assert source.dtype == target.dtype == torch.bfloat16
        assert source.device == target.device == indices.device
        assert source.is_contiguous() and target.is_contiguous()
        assert target.shape == (indices.numel(), *source.shape[1:])
    capability = torch.cuda.get_device_capability(indices.device)
    architecture = f"sm_{capability[0]}{capability[1]}a"
    compile_kernel(widths, scatter, architecture)(
        tuple(buffer.view(buffer.shape[0], -1) for buffer in storage),
        tuple(buffer.view(buffer.shape[0], -1) for buffer in rows),
        indices,
        driver.CUstream(torch.cuda.current_stream(indices.device).cuda_stream),
    )


class CuTeDSLCodecStateArena(Qwen3TTSCodecStateArena):
    """Keep codec storage and slot ownership unchanged while batching the device copies."""

    def gather_by_index(self, index: torch.Tensor) -> Qwen3TTSIncrementalCodecState:
        storage = self.storage
        state = Qwen3TTSIncrementalCodecState(
            transformer_context_length=storage.transformer_context_length,
            frame_positions=storage.frame_positions.index_select(0, index),
        )
        if self.decoder.channels_last_weights is not None:
            state.slot_index = index
            state.pad_slot = self.scratch_slot
            state.conv_histories = dict(storage.conv_histories)
            state.transconv_overlaps = dict(storage.transconv_overlaps)
        else:
            pass
        if self.key_rings:
            state.kv_rings = Qwen3TTSKVRings(
                keys=self.key_rings,
                values=self.value_rings,
                ring_frames=self.ring_frames,
                slot_rows=index * self.ring_frames,
                trash_slot_row=self.scratch_slot * self.ring_frames,
            )
        else:
            pass
        source_buffers: list[torch.Tensor] = []
        target_buffers: list[torch.Tensor] = []
        groups = [
            (storage.transformer_keys, state.transformer_keys),
            (storage.transformer_values, state.transformer_values),
        ]
        if state.slot_index is None:
            groups.extend(
                (
                    (storage.conv_histories, state.conv_histories),
                    (storage.transconv_overlaps, state.transconv_overlaps),
                )
            )
        else:
            pass
        for source_group, target_group in groups:
            for name, buffer in source_group.items():
                target = buffer.new_empty(index.numel(), *buffer.shape[1:])
                target_group[name] = target
                if buffer.numel():
                    source_buffers.append(buffer)
                    target_buffers.append(target)
                else:
                    pass
        if source_buffers:
            copy_state_rows(
                tuple(source_buffers), tuple(target_buffers), index, scatter=False
            )
        else:
            pass
        return state

    def scatter_by_index(
        self, index: torch.Tensor, state: Qwen3TTSIncrementalCodecState
    ) -> None:
        storage = self.storage
        assert state.frame_positions is not None
        self.copy_rows(storage.frame_positions, index, state.frame_positions)
        target_buffers: list[torch.Tensor] = []
        source_buffers: list[torch.Tensor] = []
        groups = [
            (storage.transformer_keys, state.transformer_keys),
            (storage.transformer_values, state.transformer_values),
        ]
        if state.slot_index is None:
            groups.extend(
                (
                    (storage.conv_histories, state.conv_histories),
                    (storage.transconv_overlaps, state.transconv_overlaps),
                )
            )
        else:
            pass
        for target_group, source_group in groups:
            for name, buffer in target_group.items():
                if buffer.numel():
                    target_buffers.append(buffer)
                    source_buffers.append(source_group[name].contiguous())
                else:
                    pass
        if target_buffers:
            copy_state_rows(
                tuple(target_buffers), tuple(source_buffers), index, scatter=True
            )
        else:
            pass
