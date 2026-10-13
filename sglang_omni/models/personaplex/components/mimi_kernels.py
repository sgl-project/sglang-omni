# SPDX-License-Identifier: Apache-2.0
"""CuTe RoPE and ring-cache writes for singleton Mimi streaming inference."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import tvm_ffi
from cuda.bindings.driver import CUstream
from cutlass.cute.math import RoundingMode
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from sglang_omni.models.personaplex.architecture import MIMI

HEAD_DIM = MIMI.dim // MIMI.num_heads
ROTARY_PAIR_COUNT = MIMI.frame_ratio * MIMI.dim // 2
BLOCK_THREADS = 256
PROJECTION_SHAPE = (1, MIMI.frame_ratio, 3 * MIMI.dim)
PHASE_SHAPE = (MIMI.frame_ratio, HEAD_DIM // 2)
CACHE_SHAPE = (1, MIMI.num_heads, MIMI.context, HEAD_DIM)
QUERY_SHAPE = (1, MIMI.num_heads, MIMI.frame_ratio, HEAD_DIM)
MASK_SHAPE = (MIMI.frame_ratio, MIMI.context)


@cute.kernel
def mimi_rope_cache_kernel(
    qkv: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    q: cute.Tensor,
    mask: cute.Tensor,
    offset: cutlass.Int64,
    end_offset: cutlass.Int64,
) -> None:
    block_index, _, _ = cute.arch.block_idx()
    lane_index, _, _ = cute.arch.thread_idx()
    element_index = block_index * BLOCK_THREADS + lane_index
    rotary_pair_index = element_index % ROTARY_PAIR_COUNT
    head_index = rotary_pair_index // (MIMI.frame_ratio * HEAD_DIM // 2)
    position_index = rotary_pair_index // (HEAD_DIM // 2) % MIMI.frame_ratio
    channel_index = rotary_pair_index % (HEAD_DIM // 2) * 2
    projection_index = (
        element_index // ROTARY_PAIR_COUNT * MIMI.dim
        + head_index * HEAD_DIM
        + channel_index
    )
    real = qkv[0, position_index, projection_index]
    imaginary = qkv[0, position_index, projection_index + 1]
    cosine_value = cos[position_index, channel_index // 2]
    sine_value = sin[position_index, channel_index // 2]
    # note (Codex): Eager RoPE rounds each FP32 product before adding or subtracting.
    rounding = RoundingMode.NEAREST_EVEN
    real_cosine = cute.math.mul(real, cosine_value, rounding=rounding)
    imaginary_sine = cute.math.mul(imaginary, sine_value, rounding=rounding)
    real_sine = cute.math.mul(real, sine_value, rounding=rounding)
    imaginary_cosine = cute.math.mul(imaginary, cosine_value, rounding=rounding)
    rotated_real = cute.math.add(real_cosine, -imaginary_sine, rounding=rounding)
    rotated_imaginary = cute.math.add(real_sine, imaginary_cosine, rounding=rounding)
    if element_index < ROTARY_PAIR_COUNT:
        q[0, head_index, position_index, channel_index] = rotated_real
        q[0, head_index, position_index, channel_index + 1] = rotated_imaginary
    else:
        slot_index = (end_offset + position_index) % MIMI.context
        k[0, head_index, slot_index, channel_index] = rotated_real
        k[0, head_index, slot_index, channel_index + 1] = rotated_imaginary

    position_index = element_index // MIMI.dim
    head_index = element_index % MIMI.dim // HEAD_DIM
    channel_index = element_index % HEAD_DIM
    slot_index = (end_offset + position_index) % MIMI.context
    v[0, head_index, slot_index, channel_index] = qkv[
        0, position_index, 2 * MIMI.dim + head_index * HEAD_DIM + channel_index
    ]

    if element_index < MIMI.frame_ratio * MIMI.context:
        position_index = element_index // MIMI.context
        slot_index = cutlass.Int64(element_index % MIMI.context)
        updated_end_offset = end_offset + MIMI.frame_ratio
        slot_delta = slot_index - updated_end_offset % MIMI.context
        key_position = updated_end_offset + slot_delta
        if slot_delta > 0:
            key_position -= MIMI.context
        else:
            pass
        position_delta = offset + position_index - key_position
        mask[position_index, slot_index] = (
            (slot_index < updated_end_offset)
            & (key_position >= 0)
            & (position_delta >= 0)
            & (position_delta < MIMI.context)
        )
    else:
        pass


@cute.jit
def launch_mimi_rope_cache(
    qkv: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    q: cute.Tensor,
    mask: cute.Tensor,
    offset: cutlass.Int64,
    end_offset: cutlass.Int64,
    stream: CUstream,
) -> None:
    mimi_rope_cache_kernel(qkv, cos, sin, k, v, q, mask, offset, end_offset).launch(
        grid=(MIMI.frame_ratio * MIMI.dim // BLOCK_THREADS, 1, 1),
        block=(BLOCK_THREADS, 1, 1),
        stream=stream,
    )


@cache
def compile_mimi_rope_cache(device_index: int) -> tvm_ffi.Function:
    with torch.cuda.device(device_index):
        tensors = [
            make_fake_compact_tensor(
                cutlass.Float32, shape, stride_order=tuple(reversed(range(len(shape))))
            )
            for shape in (
                PROJECTION_SHAPE,
                PHASE_SHAPE,
                PHASE_SHAPE,
                CACHE_SHAPE,
                CACHE_SHAPE,
                QUERY_SHAPE,
            )
        ]
        return cute.compile(
            launch_mimi_rope_cache,
            *tensors,
            make_fake_compact_tensor(cutlass.Boolean, MASK_SHAPE, stride_order=(1, 0)),
            cutlass.Int64(0),
            cutlass.Int64(0),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def fused_mimi_rope_cache(
    projected: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    offset: int,
    end_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q/K and write K/V using the pre-write end_offset."""
    assert projected.is_cuda and not torch.is_grad_enabled()
    assert offset >= 0 and end_offset >= 0
    rotated_queries = projected.new_empty(QUERY_SHAPE)
    attention_mask = projected.new_empty(MASK_SHAPE, dtype=torch.bool)
    apply_rope = compile_mimi_rope_cache(projected.get_device())
    apply_rope(
        projected,
        cosine,
        sine,
        key_cache,
        value_cache,
        rotated_queries,
        attention_mask,
        offset,
        end_offset,
    )
    return rotated_queries, attention_mask
