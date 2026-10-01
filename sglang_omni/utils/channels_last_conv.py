# SPDX-License-Identifier: Apache-2.0
"""Conv1d and ConvTranspose1d of (B, L, C) activations as channels-last cuDNN calls."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang_omni.platforms import current_platform


def is_channels_last_conv_device(device: torch.device) -> bool:
    """Channels-last convs run only on NVIDIA CUDA, the platform they are measured on."""
    return current_platform.is_cuda() and device.type == "cuda"


def channels_last_weight(
    conv: torch.nn.Conv1d | torch.nn.ConvTranspose1d,
) -> torch.Tensor:
    """A copy of the conv's weight with channels-last strides, the layout the
    channels-last convs hand to cuDNN without reformatting it."""
    return conv.weight.data.transpose(1, 2).contiguous().transpose(1, 2)


def channels_last_conv1d(
    hidden_states: torch.Tensor,
    conv: torch.nn.Conv1d,
    weight: torch.Tensor,
    length: int,
) -> torch.Tensor:
    """The conv of a (B, L, C) activation that carries its left context and, for a
    dilated conv, a length divisible by the dilation; returns (B, length, C_out)."""
    dilation = conv.dilation[0]
    if dilation == 1:
        output = F.conv2d(
            hidden_states.transpose(1, 2).unsqueeze(2),
            weight.unsqueeze(2),
            conv.bias,
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
            conv.bias,
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
