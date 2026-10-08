# SPDX-License-Identifier: Apache-2.0
"""The DiT's positional conv, a grouped Conv1d followed by Mish, in one Triton kernel over
frames in channels last order."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on runtime image
    HAS_TRITON = False

FRAMES_PER_PROGRAM = 64
# note (ratish): tl.dot reduces at least 16 channels per group, in powers of two. Over Triton's
# three pipeline stages the input and weight tiles take 48 KiB of shared memory at 64 channels
# and 144 KiB at 128, past the 99 KiB a block gets on some GPUs.
GROUP_CONV_KERNEL_CHANNELS = (16, 32, 64)

if HAS_TRITON:

    # note (ratish): the frame counts change with every step, one binary serves them all.
    @triton.jit(do_not_specialize=["out_frames", "in_frames"])
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
        accumulator += tl.load(bias_ptr + group * GROUP_CHANNELS + channels)[
            None, :
        ].to(tl.float32)
        mish = accumulator * libdevice.tanh(libdevice.log1p(tl.exp(accumulator)))
        tl.store(
            out_ptr
            + (batch * out_frames + frames[:, None]) * CHANNELS
            + group * GROUP_CHANNELS
            + channels[None, :],
            mish.to(out_ptr.dtype.element_ty),
            mask=is_frame[:, None],
        )

else:
    pass


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
    assert x.dtype == packed_weight.dtype, "tl.dot takes one input dtype"
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


class FusedConvPositionEmbedding(torch.nn.Module):
    """CosyVoice's CausalConvPositionEmbedding with each conv and its Mish in one launch,
    the module's own convs for frames outside the packed weights' dtype, as in a float
    copy. The DiT calls it without a mask, x: (batch, frames, channels)."""

    def __init__(self, original: torch.nn.Module) -> None:
        super().__init__()
        self.kernel_size = original.kernel_size
        self.conv1 = original.conv1
        self.conv2 = original.conv2
        self.packed_weights = (
            pack_group_conv_weight(self.conv1[0]),
            pack_group_conv_weight(self.conv2[0]),
        )
        self.train(original.training)

    def conv(self, x: torch.Tensor, conv_index: int) -> torch.Tensor:
        """Conv conv_index and its Mish over x: (batch, frames, channels), unpadded."""
        conv = (self.conv1, self.conv2)[conv_index]
        packed_weight = self.packed_weights[conv_index]
        if x.dtype == packed_weight.dtype:
            return torch.ops.sglang_omni_fun_cosyvoice3.group_conv_mish(
                x, packed_weight, conv[0].bias
            )
        else:
            return conv(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = (0, 0, self.kernel_size - 1, 0)
        x = self.conv(F.pad(x, padding), 0)
        return self.conv(F.pad(x, padding), 1)
