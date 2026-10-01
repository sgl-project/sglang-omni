# SPDX-License-Identifier: Apache-2.0
"""FP32 noncausal stride-two upsampling for the AuK VAE."""

import torch
import triton
import triton.language as tl

UPSAMPLE_BLOCK_SAMPLES = 256
UPSAMPLE_NUM_WARPS = 4


@triton.jit
def upsample_f32_kernel(
    input_samples: tl.tensor,
    filter_weights: tl.tensor,
    output_samples: tl.tensor,
    input_length: int,
    channel_count: int,
    batch_stride: tl.constexpr,
    channel_stride: tl.constexpr,
    sample_stride: tl.constexpr,
    block_samples: tl.constexpr,
) -> None:
    upsample_ratio: tl.constexpr = 2
    filter_samples: tl.constexpr = 12
    input_padding: tl.constexpr = filter_samples // upsample_ratio - 1
    output_crop: tl.constexpr = (
        input_padding * upsample_ratio + (filter_samples - upsample_ratio) // 2
    )
    sample_index = tl.program_id(0) * block_samples + tl.arange(0, block_samples)
    channel_index = tl.program_id(1)
    batch_index = tl.program_id(2)
    output_length = input_length * upsample_ratio
    valid_samples = sample_index < output_length
    output_offset = (batch_index * channel_count + channel_index) * output_length
    filter_parity = (sample_index + output_crop) % upsample_ratio
    accumulated_samples = tl.full((block_samples,), 0, tl.float32)
    # note (BBuf): Ascending FP32 FMAs preserve the transpose convolution's rounding.
    for tap_index in tl.static_range(filter_samples // upsample_ratio):
        filter_index = filter_parity + tap_index * upsample_ratio
        input_index = (
            sample_index + output_crop - filter_index
        ) // upsample_ratio - input_padding
        input_index = tl.minimum(tl.maximum(input_index, 0), input_length - 1)
        samples = tl.load(
            input_samples
            + batch_index * batch_stride
            + channel_index * channel_stride
            + input_index * sample_stride,
            valid_samples,
            0,
        )
        weight = tl.load(filter_weights + filter_index)
        accumulated_samples = tl.fma(samples, weight, accumulated_samples)
    tl.store(
        output_samples + output_offset + sample_index,
        accumulated_samples * upsample_ratio,
        valid_samples,
    )


def upsample_f32(
    input_samples: torch.Tensor, filter_weights: torch.Tensor
) -> torch.Tensor:
    """Apply a twelve-tap filter with replicate padding and stride two."""
    batch_count, channel_count, input_length = input_samples.shape
    output_samples = torch.empty(
        (batch_count, channel_count, input_length * 2),
        device=input_samples.device,
        dtype=input_samples.dtype,
    )
    upsample_f32_kernel[
        (
            triton.cdiv(input_length * 2, UPSAMPLE_BLOCK_SAMPLES),
            channel_count,
            batch_count,
        )
    ](
        input_samples,
        filter_weights,
        output_samples,
        input_length,
        channel_count,
        *input_samples.stride(),
        block_samples=UPSAMPLE_BLOCK_SAMPLES,
        num_warps=UPSAMPLE_NUM_WARPS,
        enable_fp_fusion=True,
    )
    return output_samples
