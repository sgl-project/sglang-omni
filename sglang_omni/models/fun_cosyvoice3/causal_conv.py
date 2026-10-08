# SPDX-License-Identifier: Apache-2.0
"""The DiT's positional conv, a grouped Conv1d followed by Mish, in one Triton kernel over
frames in channels last order."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

FRAMES_PER_PROGRAM = 64


@triton.jit
def group_conv_mish_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    out_frames,
    in_frames,
    CHANNELS: tl.constexpr,
    GROUP_CHANNELS: tl.constexpr,
    TAPS: tl.constexpr,
    BLOCK_FRAMES: tl.constexpr,
):
    frame_block = tl.program_id(0)
    group = tl.program_id(1)
    batch = tl.program_id(2)
    frames = frame_block * BLOCK_FRAMES + tl.arange(0, BLOCK_FRAMES)
    channels = tl.arange(0, GROUP_CHANNELS)
    is_frame = frames < out_frames
    x_group = x_ptr + batch * in_frames * CHANNELS + group * GROUP_CHANNELS
    weight_group = weight_ptr + group * TAPS * GROUP_CHANNELS * GROUP_CHANNELS
    accumulator = tl.zeros((BLOCK_FRAMES, GROUP_CHANNELS), dtype=tl.float32)
    for tap in range(TAPS):
        x = tl.load(
            x_group + (frames[:, None] + tap) * CHANNELS + channels[None, :],
            mask=is_frame[:, None],
            other=0.0,
        )
        weight = tl.load(
            weight_group
            + (tap * GROUP_CHANNELS + channels[:, None]) * GROUP_CHANNELS
            + channels[None, :]
        )
        accumulator = tl.dot(x, weight, accumulator)
    accumulator += tl.load(bias_ptr + group * GROUP_CHANNELS + channels)[None, :].to(
        tl.float32
    )
    mish = accumulator * libdevice.tanh(libdevice.log1p(tl.exp(accumulator)))
    tl.store(
        out_ptr
        + (batch * out_frames + frames[:, None]) * CHANNELS
        + group * GROUP_CHANNELS
        + channels[None, :],
        mish.to(out_ptr.dtype.element_ty),
        mask=is_frame[:, None],
    )


def pack_group_conv_weight(conv: torch.nn.Conv1d) -> torch.Tensor:
    """(groups, taps, in channels per group, out channels per group), contiguous."""
    out_channels, group_channels, taps = conv.weight.shape
    groups = conv.groups
    assert out_channels == groups * group_channels, "the kernel takes square groups"
    return (
        conv.weight.detach()
        .view(groups, group_channels, group_channels, taps)
        .permute(0, 3, 2, 1)
        .contiguous()
    )


@torch.library.custom_op(
    "sglang_omni_fun_cosyvoice3::group_conv_mish",
    mutates_args=(),
    device_types="cuda",
)
def group_conv_mish(
    x: torch.Tensor, packed_weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """x: (batch, frames, channels), each output frame reading the taps - 1 frames
    before it, as an unpadded Conv1d does. Returns (batch, frames - taps + 1, channels).
    """
    groups, taps, group_channels, _ = packed_weight.shape
    batch, in_frames, channels = x.shape
    assert channels == groups * group_channels and x.is_contiguous()
    out_frames = in_frames - taps + 1
    out = x.new_empty(batch, out_frames, channels)
    grid = (triton.cdiv(out_frames, FRAMES_PER_PROGRAM), groups, batch)
    group_conv_mish_kernel[grid](
        x,
        packed_weight,
        bias,
        out,
        out_frames,
        in_frames,
        CHANNELS=channels,
        GROUP_CHANNELS=group_channels,
        TAPS=taps,
        BLOCK_FRAMES=FRAMES_PER_PROGRAM,
    )
    return out


@group_conv_mish.register_fake
def fake_group_conv_mish(
    x: torch.Tensor, packed_weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    batch, in_frames, channels = x.shape
    return x.new_empty(batch, in_frames - packed_weight.shape[1] + 1, channels)
