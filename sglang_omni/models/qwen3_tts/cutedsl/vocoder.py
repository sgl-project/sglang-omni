# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Waveform-tail convolution fusion preserving the incremental decoder's module layout."""

import torch
from torch import nn
from torch.nn import functional as F

from sglang_omni.models.qwen3_tts.cutedsl.conv1d import conv1d
from sglang_omni.models.qwen3_tts.cutedsl.runtime import supports_auxiliary
from sglang_omni.models.qwen3_tts.cutedsl.snake_beta import CuTeDSLSnakeBeta
from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_conv_transpose1d,
    channels_last_weight,
)


class CuTeDSLCausalConv1d(nn.Module):
    """Keep BCT views over BTC storage, with immutable packed weights for inference."""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.conv = original.conv
        self.padding = original.padding
        self.has_custom_kernel = (
            self.conv.in_channels % 32 == 0 and self.conv.out_channels % 32 == 0
        )
        self.register_buffer(
            "packed_weight", channels_last_weight(self.conv), persistent=False
        )
        if self.has_custom_kernel:
            pass
        else:
            self.fallback = torch.compile(
                self.channels_last_forward,
                fullgraph=True,
                dynamic=True,
                options={"emulate_precision_casts": True},
            )

    def channels_last_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        length = hidden_states.shape[1]
        padded = F.pad(
            hidden_states,
            (0, 0, self.padding, -(length + self.padding) % self.conv.dilation[0]),
        )
        return channels_last_conv1d(
            padded, self.conv, self.packed_weight, self.conv.bias, length
        )

    def forward_btc(
        self,
        hidden_states: torch.Tensor,
        activation: CuTeDSLSnakeBeta | None = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.has_custom_kernel:
            return conv1d(
                hidden_states,
                self.packed_weight.transpose(1, 2),
                self.conv.bias,
                self.conv.dilation[0],
                activation,
                residual,
            )
        else:
            return self.fallback(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.forward_btc(hidden_states.transpose(1, 2).contiguous()).transpose(
            1, 2
        )


class ChannelsLastUpsample(nn.Module):
    """Retain channels-last storage through cuDNN transposed convolution and causal cropping."""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.conv = original.conv
        self.left_pad = original.left_pad
        self.right_pad = original.right_pad
        self.register_buffer(
            "packed_weight", channels_last_weight(self.conv), persistent=False
        )
        self.operation = torch.compile(
            self.channels_last_forward,
            fullgraph=True,
            dynamic=True,
            options={"emulate_precision_casts": True},
        )

    def channels_last_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = channels_last_conv_transpose1d(
            hidden_states, self.conv, self.packed_weight, self.conv.bias
        )
        return output[:, : output.shape[1] - self.right_pad].contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.operation(hidden_states.transpose(1, 2).contiguous()).transpose(
            1, 2
        )


class CuTeDSLResidualUnit(nn.Module):
    """Fuse K7 convolution with SnakeBeta, and K1 convolution with the residual addition."""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.act1 = original.act1
        self.act2 = original.act2
        self.conv1 = CuTeDSLCausalConv1d(original.conv1)
        self.conv2 = CuTeDSLCausalConv1d(original.conv2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states.transpose(1, 2).contiguous()
        activated = self.act1(hidden_states).transpose(1, 2).contiguous()
        output = self.conv1.forward_btc(activated, activation=self.act2)
        return self.conv2.forward_btc(output, residual=residual).transpose(1, 2)


def fuse_cutedsl_convolutions(decoder: nn.Module) -> int:
    """Replace supported BF16 waveform blocks after SnakeBeta preparation."""
    if not supports_auxiliary(next(decoder.parameters())) or isinstance(
        decoder.decoder[0], CuTeDSLCausalConv1d
    ):
        return 0
    else:
        pass
    layers = decoder.decoder
    convolutions = [layers[0], layers[-1]]
    residuals = [unit for block in layers[1:-2] for unit in block.block[2:]]
    convolutions.extend(conv for unit in residuals for conv in (unit.conv1, unit.conv2))
    if any(
        conv.conv.stride != (1,) or conv.conv.groups != 1 for conv in convolutions
    ) or any(
        not isinstance(unit.act1, CuTeDSLSnakeBeta)
        or not isinstance(unit.act2, CuTeDSLSnakeBeta)
        or unit.conv1.conv.in_channels % 32
        or unit.conv1.conv.out_channels % 32
        or unit.conv2.conv.in_channels % 32
        or unit.conv2.conv.out_channels % 32
        for unit in residuals
    ):
        return 0
    else:
        layers[0] = CuTeDSLCausalConv1d(layers[0])
        layers[-1] = CuTeDSLCausalConv1d(layers[-1])
        for block in layers[1:-2]:
            block.block[1] = ChannelsLastUpsample(block.block[1])
            for index in range(2, len(block.block)):
                block.block[index] = CuTeDSLResidualUnit(block.block[index])
        return sum(
            module.has_custom_kernel
            for module in decoder.modules()
            if isinstance(module, CuTeDSLCausalConv1d)
        )
