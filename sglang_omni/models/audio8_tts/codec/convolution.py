# SPDX-License-Identifier: Apache-2.0
"""Waveform encoder and decoder layers for the Audio8 codec."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from sglang_omni.models.audio8_tts.codec.attention import (
    CodecTransformerConfig,
    CodecWindowTransformer,
)

CODEC_CHANNEL_COUNT = 1024


class CausalConv1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()
        convolution = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.conv: nn.Conv1d = (
            weight_norm(convolution) if use_weight_norm else convolution
        )
        self.stride: int = stride
        self.left_padding: int = (kernel_size - 1) * dilation + 1 - stride

    def forward(self, waveform: Tensor) -> Tensor:
        right_padding = -waveform.shape[-1] % self.stride
        return self.conv(
            F.pad(waveform, (self.left_padding, right_padding))
        ).contiguous()


class CausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()
        convolution = nn.ConvTranspose1d(
            input_channels, output_channels, kernel_size, stride=stride
        )
        self.conv: nn.ConvTranspose1d = (
            weight_norm(convolution) if use_weight_norm else convolution
        )
        self.right_crop: int = kernel_size - stride

    def forward(self, features: Tensor) -> Tensor:
        waveform = self.conv(features)
        return waveform[..., : waveform.shape[-1] - self.right_crop].contiguous()


class Snake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha: nn.Parameter = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, waveform: Tensor) -> Tensor:
        shape = waveform.shape
        flattened = waveform.reshape(shape[0], shape[1], -1)
        activated = flattened + (self.alpha + 1e-9).reciprocal() * torch.sin(
            self.alpha * flattened
        ).pow(2)
        return activated.reshape(shape)


class CodecResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block: nn.Sequential = nn.Sequential(
            Snake1d(channels),
            CausalConv1d(
                channels,
                channels,
                kernel_size=7,
                dilation=dilation,
                use_weight_norm=True,
            ),
            Snake1d(channels),
            CausalConv1d(channels, channels, kernel_size=1, use_weight_norm=True),
        )

    def forward(self, waveform: Tensor) -> Tensor:
        residual = self.block(waveform)
        return waveform[..., : residual.shape[-1]] + residual


class CodecEncoderBlock(nn.Module):
    def __init__(
        self, output_channels: int, stride: int, transformer_layers: int
    ) -> None:
        super().__init__()
        input_channels = output_channels // 2
        if transformer_layers:
            transformer = CodecWindowTransformer(
                CodecTransformerConfig(
                    layer_count=transformer_layers,
                    attention_head_count=output_channels // 64,
                    hidden_size=output_channels,
                    intermediate_size=output_channels * 3,
                ),
                window_size=512,
            )
        else:
            transformer = nn.Identity()
        self.block: nn.Sequential = nn.Sequential(
            CodecResidualUnit(input_channels, 1),
            CodecResidualUnit(input_channels, 3),
            CodecResidualUnit(input_channels, 9),
            Snake1d(input_channels),
            CausalConv1d(
                input_channels,
                output_channels,
                kernel_size=2 * stride,
                stride=stride,
                use_weight_norm=True,
            ),
            transformer,
        )

    def forward(self, waveform: Tensor) -> Tensor:
        return self.block(waveform)


class CodecEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = 64
        layers: list[nn.Module] = [
            CausalConv1d(1, channels, kernel_size=7, use_weight_norm=True)
        ]
        for stride, transformer_layers in zip((2, 4, 8, 8), (0, 0, 0, 4), strict=True):
            channels *= 2
            layers.append(CodecEncoderBlock(channels, stride, transformer_layers))
        layers.extend(
            (
                Snake1d(channels),
                CausalConv1d(
                    channels, CODEC_CHANNEL_COUNT, kernel_size=3, use_weight_norm=True
                ),
            )
        )
        self.block: nn.Sequential = nn.Sequential(*layers)

    def forward(self, waveform: Tensor) -> Tensor:
        return self.block(waveform)


class CodecDecoderBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__()
        self.block: nn.Sequential = nn.Sequential(
            Snake1d(input_channels),
            CausalConvTranspose1d(
                input_channels,
                output_channels,
                kernel_size=2 * stride,
                stride=stride,
                use_weight_norm=True,
            ),
            CodecResidualUnit(output_channels, 1),
            CodecResidualUnit(output_channels, 3),
            CodecResidualUnit(output_channels, 9),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.block(features)


class CodecDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = 1536
        layers: list[nn.Module] = [
            CausalConv1d(
                CODEC_CHANNEL_COUNT, channels, kernel_size=7, use_weight_norm=True
            )
        ]
        for stride in (8, 8, 4, 2):
            layers.append(CodecDecoderBlock(channels, channels // 2, stride))
            channels //= 2
        layers.extend(
            (
                Snake1d(channels),
                CausalConv1d(channels, 1, kernel_size=7, use_weight_norm=True),
                nn.Tanh(),
            )
        )
        self.model: nn.Sequential = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.model(features)
