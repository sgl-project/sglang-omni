# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of the Zonos3 reference encoder's convolution stack: one kernel where the encoder runs fused, the eager ops elsewhere.

Each kernel rounds exactly where the eager chain rounds, so the outputs match the eager ops bit for bit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

SNAKE_EPSILON = 1e-9
# note (zhaochenyang20): a snake program covers this many (sample, channel) elements, 32 samples of 64 channels or 16 of 128; 8192 spilled registers around the accurate sine and ran twice as long on one H100.
SNAKE_TILE_ELEMENTS = 2048
# note (zhaochenyang20): a full 300-frame lane's patch-down took 40.8 us with 32-frame blocks against 42.9 us for 16, 41.0 us for 64 and 135 us for 8 (CUDA graph replay, one H100).
PATCH_DOWN_BLOCK_FRAMES = 32


@triton.jit
def biased_snake_kernel(
    convolved_pointer,
    bias_pointer,
    skip_pointer,
    alpha_pointer,
    states_pointer,
    activated_pointer,
    sample_count,
    left_padding,
    convolved_batch_stride,
    convolved_channel_stride,
    convolved_sample_stride,
    CHANNELS: tl.constexpr,
    HAS_SKIP: tl.constexpr,
    SHOULD_KEEP_STATES: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_SAMPLES: tl.constexpr,
):
    """One batch row's block of output samples over all channels: states = skip + (convolved + bias), each sum rounded to bfloat16, then x + sin(alpha x)^2 / (alpha + epsilon) of the states in float32, left zero padded, in bfloat16.

    The skip, states and activation are (B, T, C) memory; the convolution output may have any strides.
    """
    batch_index = tl.program_id(0).to(tl.int64)
    output_positions = (
        tl.program_id(1) * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)[:, None]
    )
    channels = tl.arange(0, CHANNELS)[None, :]
    sample_positions = output_positions - left_padding
    is_sample = (sample_positions >= 0) & (sample_positions < sample_count)
    samples = (
        libdevice.add_rn(
            tl.load(
                convolved_pointer
                + batch_index * convolved_batch_stride
                + channels * convolved_channel_stride
                + sample_positions * convolved_sample_stride,
                mask=is_sample,
                other=0.0,
            ).to(tl.float32),
            tl.load(bias_pointer + channels).to(tl.float32),
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    state_offsets = (
        batch_index * sample_count + sample_positions
    ) * CHANNELS + channels
    if HAS_SKIP:
        samples = (
            libdevice.add_rn(
                tl.load(skip_pointer + state_offsets, mask=is_sample, other=0.0).to(
                    tl.float32
                ),
                samples,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
    else:
        pass
    if SHOULD_KEEP_STATES:
        tl.store(
            states_pointer + state_offsets, samples.to(tl.bfloat16), mask=is_sample
        )
    else:
        pass
    alpha = tl.load(alpha_pointer + channels)
    inverse_alpha = libdevice.div_rn(1.0, libdevice.add_rn(alpha, EPSILON))
    sine = libdevice.sin(libdevice.mul_rn(alpha, samples))
    activated = libdevice.add_rn(
        samples, libdevice.mul_rn(inverse_alpha, libdevice.mul_rn(sine, sine))
    )
    tl.store(
        activated_pointer
        + (batch_index * (left_padding + sample_count) + output_positions) * CHANNELS
        + channels,
        tl.where(is_sample, activated, 0.0).to(tl.bfloat16),
        mask=output_positions < left_padding + sample_count,
    )


def biased_snake(
    convolved: torch.Tensor,
    bias: torch.Tensor,
    alpha: torch.Tensor,
    left_padding: int,
    skip_states: torch.Tensor | None,
    should_keep_states: bool,
    uses_fused_encoder: bool,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """The states skip_states + (convolved + bias) of a (B, C, T) bias-free convolution and a (C,) bias, and their snake with a (1, C, 1) alpha, zero padded on the left by left_padding samples; the states come back only when kept.

    Each sum rounds to the convolution's dtype as the eager bias and skip adds do. When the encoder runs fused the states and the activation are bfloat16 (B, C, T) views of (B, T, C) memory, which the next channels-last convolution reads without a transpose; otherwise the snake keeps the eager float32 result.
    """
    if uses_fused_encoder and convolved.dtype == torch.bfloat16:
        # note (zhaochenyang20): the convolution stack ran a bias add after every convolution, a skip add per residual unit and cuDNN layout transposes around every convolution; folding them into the snake and holding (B, T, C) memory cut a 4.8 s reference's stack from 2.0 ms to 1.0 ms of graph replay on one H100, bit for bit.
        batch_size, channel_count, sample_count = convolved.shape
        assert skip_states is None or skip_states.stride(1) == 1, skip_states.stride()
        if should_keep_states:
            states = torch.empty(
                (batch_size, sample_count, channel_count),
                dtype=torch.bfloat16,
                device=convolved.device,
            ).transpose(1, 2)
        else:
            states = None
        activated = torch.empty(
            (batch_size, left_padding + sample_count, channel_count),
            dtype=torch.bfloat16,
            device=convolved.device,
        ).transpose(1, 2)
        block_samples = SNAKE_TILE_ELEMENTS // channel_count
        biased_snake_kernel[
            (batch_size, triton.cdiv(left_padding + sample_count, block_samples))
        ](
            convolved,
            bias,
            convolved if skip_states is None else skip_states,
            alpha.reshape(-1),
            activated if states is None else states,
            activated,
            sample_count,
            left_padding,
            *convolved.stride(),
            CHANNELS=channel_count,
            HAS_SKIP=skip_states is not None,
            SHOULD_KEEP_STATES=should_keep_states,
            EPSILON=SNAKE_EPSILON,
            BLOCK_SAMPLES=block_samples,
            enable_fp_fusion=False,
        )
        return states, activated
    else:
        states = convolved + bias[:, None]
        if skip_states is None:
            pass
        else:
            states = skip_states + states
        periodic = torch.sin(alpha * states).pow(2)
        activated = F.pad(
            states + (alpha + SNAKE_EPSILON).reciprocal() * periodic,
            (left_padding, 0),
        )
        return (states if should_keep_states else None), activated


@triton.jit
def biased_patch_down_kernel(
    convolved_pointer,
    bias_pointer,
    patched_pointer,
    patched_frame_count,
    CHANNELS: tl.constexpr,
    PATCH_SIZE: tl.constexpr,
    BLOCK_FRAMES: tl.constexpr,
):
    """One (batch, patched frame block, phase): convolved + bias of (B, T, C) memory, rounded to bfloat16, stored at channel c * PATCH_SIZE + phase of the (B, C * p, T / p) output."""
    batch_index = tl.program_id(0).to(tl.int64)
    patched_frames = (
        tl.program_id(1) * BLOCK_FRAMES + tl.arange(0, BLOCK_FRAMES)[:, None]
    )
    phase = tl.program_id(2)
    channels = tl.arange(0, CHANNELS)[None, :]
    is_frame = patched_frames < patched_frame_count
    states = libdevice.add_rn(
        tl.load(
            convolved_pointer
            + (
                (batch_index * patched_frame_count + patched_frames) * PATCH_SIZE
                + phase
            )
            * CHANNELS
            + channels,
            mask=is_frame,
            other=0.0,
        ).to(tl.float32),
        tl.load(bias_pointer + channels).to(tl.float32),
    )
    tl.store(
        patched_pointer
        + ((batch_index * CHANNELS + channels) * PATCH_SIZE + phase)
        * patched_frame_count
        + patched_frames,
        states.to(tl.bfloat16),
        mask=is_frame,
    )


def biased_patch_down(
    convolved: torch.Tensor,
    bias: torch.Tensor,
    patch_size: int,
    uses_fused_encoder: bool,
) -> torch.Tensor:
    """patch_down of a (B, C, T) bias-free convolution plus its (C,) bias, the sum rounded as the eager add; when the encoder runs fused the convolution is (B, T, C) memory."""
    if uses_fused_encoder and convolved.dtype == torch.bfloat16:
        batch_size, channel_count, sample_count = convolved.shape
        assert convolved.stride() == (
            sample_count * channel_count,
            1,
            channel_count,
        ), convolved.stride()
        assert sample_count % patch_size == 0, (sample_count, patch_size)
        patched = torch.empty(
            (batch_size, channel_count * patch_size, sample_count // patch_size),
            dtype=torch.bfloat16,
            device=convolved.device,
        )
        biased_patch_down_kernel[
            (
                batch_size,
                triton.cdiv(sample_count // patch_size, PATCH_DOWN_BLOCK_FRAMES),
                patch_size,
            )
        ](
            convolved,
            bias,
            patched,
            sample_count // patch_size,
            CHANNELS=channel_count,
            PATCH_SIZE=patch_size,
            BLOCK_FRAMES=PATCH_DOWN_BLOCK_FRAMES,
        )
        return patched
    else:
        batch_size, channel_count, sample_count = convolved.shape
        states = (convolved + bias[:, None]).reshape(
            batch_size, channel_count, sample_count // patch_size, patch_size
        )
        return states.permute(0, 1, 3, 2).reshape(
            batch_size, channel_count * patch_size, sample_count // patch_size
        )
