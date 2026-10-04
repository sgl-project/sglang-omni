# SPDX-License-Identifier: Apache-2.0
"""CuTe RoPE and ring-cache writes for singleton Mimi streaming inference."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import tvm_ffi
from cuda.bindings.driver import CUstream
from cutlass.cute.math import RoundingMode
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from sglang_omni.models.personaplex.architecture import MIMI

HEAD_DIM = MIMI.dim // MIMI.num_heads
ROTARY_PAIR_COUNT = MIMI.frame_ratio * MIMI.dim // 2
BLOCK_THREADS = 256
CUTE_COMPILE_OPTIONS = "--enable-tvm-ffi"


@cute.kernel
def mimi_rope_cache_kernel(
    projected: cute.Tensor,
    cosine: cute.Tensor,
    sine: cute.Tensor,
    key_cache: cute.Tensor,
    value_cache: cute.Tensor,
    rotated_queries: cute.Tensor,
    attention_mask: cute.Tensor,
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
    real = projected[0, position_index, projection_index]
    imaginary = projected[0, position_index, projection_index + 1]
    cosine_value = cosine[position_index, channel_index // 2]
    sine_value = sine[position_index, channel_index // 2]
    # note (Codex): Eager RoPE rounds each FP32 product before adding or subtracting.
    real_cosine = cute.math.mul(real, cosine_value, rounding=RoundingMode.NEAREST_EVEN)
    imaginary_sine = cute.math.mul(
        imaginary, sine_value, rounding=RoundingMode.NEAREST_EVEN
    )
    real_sine = cute.math.mul(real, sine_value, rounding=RoundingMode.NEAREST_EVEN)
    imaginary_cosine = cute.math.mul(
        imaginary, cosine_value, rounding=RoundingMode.NEAREST_EVEN
    )
    rotated_real = cute.math.add(
        real_cosine, -imaginary_sine, rounding=RoundingMode.NEAREST_EVEN
    )
    rotated_imaginary = cute.math.add(
        real_sine, imaginary_cosine, rounding=RoundingMode.NEAREST_EVEN
    )
    if element_index < ROTARY_PAIR_COUNT:
        rotated_queries[0, head_index, position_index, channel_index] = rotated_real
        rotated_queries[0, head_index, position_index, channel_index + 1] = (
            rotated_imaginary
        )
    else:
        slot_index = (end_offset + position_index) % MIMI.context
        key_cache[0, head_index, slot_index, channel_index] = rotated_real
        key_cache[0, head_index, slot_index, channel_index + 1] = rotated_imaginary

    position_index = element_index // MIMI.dim
    head_index = element_index % MIMI.dim // HEAD_DIM
    channel_index = element_index % HEAD_DIM
    slot_index = (end_offset + position_index) % MIMI.context
    value_cache[0, head_index, slot_index, channel_index] = projected[
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
        if slot_index >= updated_end_offset:
            key_position = cutlass.Int64(-1)
        else:
            pass
        position_delta = offset + position_index - key_position
        attention_mask[position_index, slot_index] = (
            (key_position >= 0)
            & (position_delta >= 0)
            & (position_delta < MIMI.context)
        )
    else:
        pass


@cute.jit
def launch_mimi_rope_cache(
    projected: cute.Tensor,
    cosine: cute.Tensor,
    sine: cute.Tensor,
    key_cache: cute.Tensor,
    value_cache: cute.Tensor,
    rotated_queries: cute.Tensor,
    attention_mask: cute.Tensor,
    offset: cutlass.Int64,
    end_offset: cutlass.Int64,
    stream: CUstream,
) -> None:
    mimi_rope_cache_kernel(
        projected,
        cosine,
        sine,
        key_cache,
        value_cache,
        rotated_queries,
        attention_mask,
        offset,
        end_offset,
    ).launch(
        grid=(MIMI.frame_ratio * MIMI.dim // BLOCK_THREADS, 1, 1),
        block=(BLOCK_THREADS, 1, 1),
        stream=stream,
    )


@cache
def compile_mimi_rope_cache(device_index: int) -> tvm_ffi.Function:
    with torch.cuda.device(device_index):
        trigonometric_values = make_fake_tensor(
            cutlass.Float32, (MIMI.frame_ratio, HEAD_DIM // 2), (HEAD_DIM // 2, 1)
        )
        ring_cache = make_fake_tensor(
            cutlass.Float32,
            (1, MIMI.num_heads, MIMI.context, HEAD_DIM),
            (MIMI.dim * MIMI.context, MIMI.context * HEAD_DIM, HEAD_DIM, 1),
        )
        return cute.compile(
            launch_mimi_rope_cache,
            make_fake_tensor(
                cutlass.Float32,
                (1, MIMI.frame_ratio, 3 * MIMI.dim),
                (MIMI.frame_ratio * 3 * MIMI.dim, 3 * MIMI.dim, 1),
            ),
            trigonometric_values,
            trigonometric_values,
            ring_cache,
            ring_cache,
            make_fake_tensor(
                cutlass.Float32,
                (1, MIMI.num_heads, MIMI.frame_ratio, HEAD_DIM),
                (MIMI.frame_ratio * MIMI.dim, MIMI.frame_ratio * HEAD_DIM, HEAD_DIM, 1),
            ),
            make_fake_tensor(
                cutlass.Boolean,
                (MIMI.frame_ratio, MIMI.context),
                (MIMI.context, 1),
            ),
            cutlass.Int64(0),
            cutlass.Int64(0),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options=CUTE_COMPILE_OPTIONS,
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
    """Rotate Q/K and write K/V using the pre-write end_offset.

    offset is the absolute query position; Python cache state is not advanced.
    """
    assert not torch.is_grad_enabled()
    assert projected.is_cuda and projected.shape == (1, MIMI.frame_ratio, 3 * MIMI.dim)
    assert cosine.shape == sine.shape == (MIMI.frame_ratio, HEAD_DIM // 2)
    assert (
        key_cache.shape
        == value_cache.shape
        == (
            1,
            MIMI.num_heads,
            MIMI.context,
            HEAD_DIM,
        )
    )
    assert offset >= 0 and end_offset >= 0
    for tensor in (projected, cosine, sine, key_cache, value_cache):
        assert tensor.dtype == torch.float32 and tensor.is_contiguous()
        assert tensor.device == projected.device
    rotated_queries = projected.new_empty(
        (1, MIMI.num_heads, MIMI.frame_ratio, HEAD_DIM)
    )
    attention_mask = projected.new_empty(
        (MIMI.frame_ratio, MIMI.context), dtype=torch.bool
    )
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
