# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

PAD_BLOCK_POSITIONS = 32
PAD_BLOCK_CHANNELS = 64
SNAKE_BLOCK_POSITIONS = 16
SNAKE_BLOCK_CHANNELS = 128


@triton.jit(
    do_not_specialize=[
        "frames",
        "source_batch_stride",
        "source_channel_stride",
        "padded_frames",
    ]
)
def replicate_pad_channels_last_kernel(
    source_ptr,
    padded_ptr,
    channels,
    frames,
    source_batch_stride,
    source_channel_stride,
    source_frame_stride,
    padded_frames,
    pad,
    BLOCK_POSITIONS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    batch = tl.program_id(2).to(tl.int64)
    positions = tl.program_id(0) * BLOCK_POSITIONS + tl.arange(0, BLOCK_POSITIONS)
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    mask = (channel < channels)[:, None] & (positions < padded_frames)[None, :]
    frame = tl.minimum(tl.maximum(positions - pad, 0), frames - 1)
    values = tl.load(
        source_ptr
        + batch * source_batch_stride
        + channel[:, None] * source_channel_stride
        + frame[None, :] * source_frame_stride,
        mask=mask,
        other=0.0,
    )
    tl.store(
        padded_ptr
        + (batch * padded_frames + positions[None, :]) * channels
        + channel[:, None],
        values,
        mask=mask,
    )


@triton.jit(
    do_not_specialize=[
        "upsampled_batch_stride",
        "upsampled_channel_stride",
        "frames",
        "activated_frames",
    ]
)
def snake_beta_channels_last_kernel(
    upsampled_ptr,
    activated_ptr,
    alpha_scale_ptr,
    inverse_beta_ptr,
    channels,
    upsampled_batch_stride,
    upsampled_channel_stride,
    upsampled_frame_stride,
    crop_start,
    frames,
    activated_frames,
    pad_left,
    ratio,
    BLOCK_POSITIONS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    batch = tl.program_id(2).to(tl.int64)
    positions = tl.program_id(0) * BLOCK_POSITIONS + tl.arange(0, BLOCK_POSITIONS)
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    is_channel = channel < channels
    mask = (positions < activated_frames)[:, None] & is_channel[None, :]
    alpha_scale = tl.load(alpha_scale_ptr + channel, mask=is_channel, other=0.0)
    inverse_beta = tl.load(inverse_beta_ptr + channel, mask=is_channel, other=0.0)
    frame = tl.minimum(tl.maximum(positions - pad_left, 0), frames - 1) + crop_start
    upsampled = (
        tl.load(
            upsampled_ptr
            + batch * upsampled_batch_stride
            + frame[:, None] * upsampled_frame_stride
            + channel[None, :] * upsampled_channel_stride,
            mask=mask,
            other=0.0,
        )
        * ratio
    )
    sine = tl.sin(upsampled * alpha_scale[None, :])
    tl.store(
        activated_ptr
        + (batch * activated_frames + positions[:, None]) * channels
        + channel[None, :],
        upsampled + inverse_beta[None, :] * (sine * sine),
        mask=mask,
    )


def anti_aliased_snake_beta(
    hidden_states: torch.Tensor,
    alpha_scale: torch.Tensor,
    inverse_beta: torch.Tensor,
    *,
    upsample_filter: torch.Tensor,
    upsample_ratio: int,
    upsample_pad: int,
    upsample_crop_start: int,
    upsample_crop_end: int,
    lowpass_filter: torch.Tensor,
    lowpass_stride: int,
    lowpass_pad_left: int,
    lowpass_pad_right: int,
) -> torch.Tensor:
    batch_size, channels, frames = hidden_states.shape
    padded_frames = frames + 2 * upsample_pad
    padded = torch.empty(
        batch_size,
        padded_frames,
        channels,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    replicate_pad_channels_last_kernel[
        (
            triton.cdiv(padded_frames, PAD_BLOCK_POSITIONS),
            triton.cdiv(channels, PAD_BLOCK_CHANNELS),
            batch_size,
        )
    ](
        hidden_states,
        padded,
        channels,
        frames,
        *hidden_states.stride(),
        padded_frames,
        upsample_pad,
        BLOCK_POSITIONS=PAD_BLOCK_POSITIONS,
        BLOCK_CHANNELS=PAD_BLOCK_CHANNELS,
    )
    upsampled = F.conv_transpose1d(
        padded.transpose(1, 2),
        upsample_filter.expand(channels, -1, -1),
        stride=upsample_ratio,
        groups=channels,
    )
    upsampled_frames = upsampled.shape[-1] - upsample_crop_start - upsample_crop_end
    activated_frames = upsampled_frames + lowpass_pad_left + lowpass_pad_right
    activated = torch.empty(
        batch_size,
        activated_frames,
        channels,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    snake_beta_channels_last_kernel[
        (
            triton.cdiv(activated_frames, SNAKE_BLOCK_POSITIONS),
            triton.cdiv(channels, SNAKE_BLOCK_CHANNELS),
            batch_size,
        )
    ](
        upsampled,
        activated,
        alpha_scale,
        inverse_beta,
        channels,
        upsampled.stride(0),
        upsampled.stride(1),
        upsampled.stride(2),
        upsample_crop_start,
        upsampled_frames,
        activated_frames,
        lowpass_pad_left,
        float(upsample_ratio),
        BLOCK_POSITIONS=SNAKE_BLOCK_POSITIONS,
        BLOCK_CHANNELS=SNAKE_BLOCK_CHANNELS,
        enable_fp_fusion=False,
    )
    return F.conv1d(
        activated.transpose(1, 2),
        lowpass_filter.expand(channels, -1, -1),
        stride=lowpass_stride,
        groups=channels,
    ).contiguous()
