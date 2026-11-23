# SPDX-License-Identifier: Apache-2.0
"""Triton linear upsampling for Ming AudioVAE."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

HAS_TRITON = triton is not None
UPSAMPLE_BLOCK_ELEMENTS = 512
MINIMUM_BF16_COMPUTE_CAPABILITY = (8, 0)

if HAS_TRITON:

    @triton.jit
    def upsample_linear_bth_kernel(
        timeline,
        upsampled_timeline,
        batch_size: tl.constexpr,
        input_time_steps: tl.constexpr,
        hidden_size: tl.constexpr,
        scale_factor: tl.constexpr,
        timeline_batch_stride_elements: tl.constexpr,
        timeline_time_stride_elements: tl.constexpr,
        timeline_hidden_stride_elements: tl.constexpr,
        output_element_count: tl.constexpr,
        block_elements: tl.constexpr,
    ) -> None:
        output_element_offsets = tl.program_id(0) * block_elements + tl.arange(
            0, block_elements
        )
        valid_output_mask = output_element_offsets < output_element_count
        output_time_steps = input_time_steps * scale_factor
        hidden_index = output_element_offsets % hidden_size
        output_time_index = (output_element_offsets // hidden_size) % output_time_steps
        batch_index = output_element_offsets // (hidden_size * output_time_steps)

        # note(chenye): Match align_corners=False by sampling at half pixel coordinates.
        input_position_time_steps = tl.maximum(
            (output_time_index.to(tl.float32) + 0.5) / scale_factor - 0.5,
            0.0,
        )
        left_time_index = tl.floor(input_position_time_steps).to(tl.int32)
        right_time_index = tl.minimum(left_time_index + 1, input_time_steps - 1)
        interpolation_weight = input_position_time_steps - left_time_index.to(
            tl.float32
        )
        left_element_offset = (
            batch_index * timeline_batch_stride_elements
            + left_time_index * timeline_time_stride_elements
            + hidden_index * timeline_hidden_stride_elements
        )
        right_element_offset = (
            batch_index * timeline_batch_stride_elements
            + right_time_index * timeline_time_stride_elements
            + hidden_index * timeline_hidden_stride_elements
        )
        left_hidden_value = tl.load(
            timeline + left_element_offset, mask=valid_output_mask, other=0.0
        ).to(tl.float32)
        right_hidden_value = tl.load(
            timeline + right_element_offset, mask=valid_output_mask, other=0.0
        ).to(tl.float32)
        interpolated_hidden_value = left_hidden_value * (1.0 - interpolation_weight) + (
            right_hidden_value * interpolation_weight
        )
        tl.store(
            upsampled_timeline + output_element_offsets,
            interpolated_hidden_value,
            mask=valid_output_mask,
        )

else:
    upsample_linear_bth_kernel = None


def can_upsample_linear_bth(timeline: torch.Tensor) -> bool:
    """Return whether Triton supports this non-empty CUDA BF16 BTH tensor."""

    return (
        HAS_TRITON
        and timeline.device.type == "cuda"
        and timeline.dtype == torch.bfloat16
        and timeline.ndim == 3
        and all(size > 0 for size in timeline.shape)
        and all(stride >= 0 for stride in timeline.stride())
        and torch.cuda.get_device_capability(timeline.device)
        >= MINIMUM_BF16_COMPUTE_CAPABILITY
    )


def upsample_linear_bth(timeline: torch.Tensor, *, scale_factor: int) -> torch.Tensor:
    """Upsample an eligible CUDA BF16 BTH tensor into contiguous BTH output."""

    batch_size, input_time_steps, hidden_size = timeline.shape
    output_time_steps = input_time_steps * scale_factor
    upsampled_timeline = torch.empty(
        (batch_size, output_time_steps, hidden_size),
        dtype=timeline.dtype,
        device=timeline.device,
    )
    output_element_count = upsampled_timeline.numel()
    upsample_linear_bth_kernel[
        (triton.cdiv(output_element_count, UPSAMPLE_BLOCK_ELEMENTS),)
    ](
        timeline,
        upsampled_timeline,
        batch_size,
        input_time_steps,
        hidden_size,
        scale_factor,
        timeline.stride(0),
        timeline.stride(1),
        timeline.stride(2),
        output_element_count,
        UPSAMPLE_BLOCK_ELEMENTS,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return upsampled_timeline
