# SPDX-License-Identifier: Apache-2.0
"""Conv1d and ConvTranspose1d of (B, L, C) activations as channels-last cuDNN calls."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang_omni.platforms import current_platform


def is_channels_last_conv_device(device: torch.device) -> bool:
    """Channels-last convs run only on NVIDIA CUDA, the platform they are measured on."""
    return current_platform.is_cuda() and device.type == "cuda"


def channels_last_strides(weight: torch.Tensor) -> torch.Tensor:
    """A (C_out, C_in, k) weight with channels-last strides, the layout the
    channels-last convs hand to cuDNN without reformatting it."""
    return weight.transpose(1, 2).contiguous().transpose(1, 2)


def channels_last_weight(
    conv: torch.nn.Conv1d | torch.nn.ConvTranspose1d,
) -> torch.Tensor:
    """The conv's weight with channels-last strides; the module's weight is left as
    is."""
    return channels_last_strides(conv.weight.data)


def channels_last_folded_weight(weight: torch.Tensor, stride: int) -> torch.Tensor:
    """The (C_out, C_in, k) weight of a stride-s conv as the (C_out, s * C_in, k / s)
    weight of the stride-one conv over the input's s phases laid along its channels,
    with channels-last strides."""
    out_channels, in_channels, kernel_size = weight.shape
    assert kernel_size % stride == 0, "the taps fold into whole phases"
    folded = (
        weight.view(out_channels, in_channels, kernel_size // stride, stride)
        .permute(0, 3, 1, 2)
        .reshape(out_channels, stride * in_channels, kernel_size // stride)
    )
    return channels_last_strides(folded)


def channels_last_conv1d(
    hidden_states: torch.Tensor,
    conv: torch.nn.Conv1d,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    length: int,
) -> torch.Tensor:
    """The conv of a (B, L, C) activation that carries its left context: a stride-one
    conv returns every output row, a dilated one the first length rows of an input
    divisible by the dilation, and a strided one, given its folded weight and an input
    divisible by the stride, the first length rows over the input's phases."""
    dilation = conv.dilation[0]
    stride = conv.stride[0]
    assert stride == 1 or (
        dilation == 1 and conv.groups == 1
    ), "a strided conv folds undilated and ungrouped"
    if stride > 1:
        batch_size, padded_length, channels = hidden_states.shape
        assert padded_length % stride == 0, "the phases fold whole"
        phases = hidden_states.view(
            batch_size, padded_length // stride, stride * channels
        )
        output = F.conv2d(
            phases.transpose(1, 2).unsqueeze(2),
            weight.unsqueeze(2),
            bias,
            groups=conv.groups,
        )
        return output.squeeze(2).transpose(1, 2)[:, :length].contiguous()
    elif dilation == 1:
        output = F.conv2d(
            hidden_states.transpose(1, 2).unsqueeze(2),
            weight.unsqueeze(2),
            bias,
            groups=conv.groups,
        )
        return output.squeeze(2).transpose(1, 2)
    else:
        # note (ratish): cuDNN has no fast channels-last engine for some dilated
        # shapes; viewed as (B, C, L / d, d) the conv is undilated along L / d, one
        # phase per column, without copying the phases apart.
        batch_size, _, channels = hidden_states.shape
        output = F.conv2d(
            hidden_states.view(batch_size, -1, dilation, channels).permute(0, 3, 1, 2),
            weight.unsqueeze(3),
            bias,
            groups=conv.groups,
        )
        return (
            output.permute(0, 2, 3, 1)
            .reshape(batch_size, -1, output.shape[1])[:, :length]
            .contiguous()
        )


def channels_last_conv_transpose1d(
    hidden_states: torch.Tensor,
    conv: torch.nn.ConvTranspose1d,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """The transposed conv of a (B, L, C) activation, returned as (B, L_out, C_out)."""
    return (
        F.conv_transpose2d(
            hidden_states.transpose(1, 2).unsqueeze(2),
            weight.unsqueeze(2),
            bias,
            stride=(1, conv.stride[0]),
            padding=(0, conv.padding[0]),
            output_padding=(0, conv.output_padding[0]),
            groups=conv.groups,
            dilation=(1, conv.dilation[0]),
        )
        .squeeze(2)
        .transpose(1, 2)
    )
