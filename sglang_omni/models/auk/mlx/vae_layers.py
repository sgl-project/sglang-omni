# SPDX-License-Identifier: MIT AND Apache-2.0
# Copyright (C) 2026 Tencent. All rights reserved.
# Copyright (c) 2022 NVIDIA CORPORATION.
# Derived from Tencent-Hunyuan/AuK; see LICENSE for the MIT permission notice.
# Alias-free activation code adapts junjun3518/alias-free-torch (Apache-2.0).
"""Native MLX convolution and alias-free activation layers for the AuK VAE."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def kaiser_sinc_filter1d(
    cutoff: float, half_width: float, kernel_size: int
) -> mx.array:
    """Create the normalized low-pass filter in MLX convolution layout."""
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * 4 * half_width + 7.95
    if attenuation > 50:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21)
    else:
        beta = 0.0
    time = np.arange(kernel_size, dtype=np.float32) - (kernel_size - 1) / 2
    kernel = (
        2
        * cutoff
        * np.kaiser(kernel_size, beta).astype(np.float32)
        * np.sinc(2 * cutoff * time)
    )
    kernel = kernel / kernel.sum() if cutoff != 0 else kernel
    return mx.array(kernel.reshape(1, kernel_size, 1))


class Conv1d(nn.Conv1d):
    """Channel-last convolution with symmetric or causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        causal: bool = False,
        bias: bool = True,
    ) -> None:
        padding = 0 if causal else dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.left_padding = dilation * (kernel_size - 1) if causal else 0

    def __call__(self, x: mx.array) -> mx.array:
        if self.left_padding:
            x = mx.pad(x, [(0, 0), (self.left_padding, 0), (0, 0)])
        else:
            pass
        return super().__call__(x)


class ConvTranspose1d(nn.ConvTranspose1d):
    """Channel-last transposed convolution with the causal tail removed."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        causal: bool,
    ) -> None:
        if causal and kernel_size != 2 * stride:
            raise ValueError("Causal VAE upsampling requires kernel_size == 2 * stride")
        else:
            pass
        padding = 0 if causal else (kernel_size - stride) // 2
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.causal = causal

    def __call__(self, x: mx.array) -> mx.array:
        x = super().__call__(x)
        return x[:, : -self.stride, :] if self.causal else x


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff: float,
        half_width: float,
        stride: int,
        kernel_size: int,
        causal: bool,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.pad_left = kernel_size - 1 if causal else (kernel_size - 1) // 2
        self.pad_right = 0 if causal else kernel_size // 2
        self.filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        channels = x.shape[-1]
        x = mx.pad(x, [(0, 0), (self.pad_left, self.pad_right), (0, 0)], mode="edge")
        return mx.conv1d(
            x,
            mx.broadcast_to(self.filter, (channels, self.filter.shape[1], 1)),
            stride=self.stride,
            groups=channels,
        )


class UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12) -> None:
        super().__init__()
        self.ratio = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.filter = kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        channels = x.shape[-1]
        x = mx.pad(x, [(0, 0), (self.pad, self.pad), (0, 0)], mode="edge")
        x = self.ratio * mx.conv_transpose1d(
            x,
            mx.broadcast_to(self.filter, (channels, self.filter.shape[1], 1)),
            stride=self.ratio,
            groups=channels,
        )
        return x[:, self.pad_left : -self.pad_right, :]


class DownSample1d(nn.Module):
    def __init__(
        self, ratio: int = 2, kernel_size: int = 12, causal: bool = False
    ) -> None:
        super().__init__()
        self.lowpass = LowPassFilter1d(
            0.5 / ratio, 0.6 / ratio, ratio, kernel_size, causal
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class SnakeBeta(nn.Module):
    def __init__(self, channels: int, alpha_logscale: bool) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = mx.zeros(channels) if alpha_logscale else mx.ones(channels)
        self.beta = mx.zeros(channels) if alpha_logscale else mx.ones(channels)

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha) if self.alpha_logscale else self.alpha
        beta = mx.exp(self.beta) if self.alpha_logscale else self.beta
        return x + mx.square(mx.sin(x * alpha)) / (beta + 1e-9)


class Activation1d(nn.Module):
    def __init__(self, channels: int, causal: bool, snake_logscale: bool) -> None:
        super().__init__()
        self.act = SnakeBeta(channels, snake_logscale)
        self.upsample = UpSample1d()
        self.downsample = DownSample1d(causal=causal)

    def __call__(self, x: mx.array) -> mx.array:
        return self.downsample(self.act(self.upsample(x)))


class Conv1dS(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.layer = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.layer(x)


class ResStack(nn.Module):
    def __init__(self, channels: int, stacks: int = 6, dilation_base: int = 2) -> None:
        super().__init__()
        self.layers = [
            [
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv1d(
                    channels,
                    channels,
                    3,
                    dilation=dilation_base**i,
                    padding=dilation_base**i,
                ),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv1d(channels, channels, 3, padding=1),
            ]
            for i in range(stacks)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layers in self.layers:
            residual = x
            for layer in layers:
                residual = layer(residual)
            x = x + residual
        return x


class Encoder(nn.Module):
    def __init__(
        self, latent_dim: int, channels: list[int], downsample_rates: list[int]
    ) -> None:
        super().__init__()
        self.generator = [Conv1dS(1, channels[0], 3), nn.LeakyReLU(0.2)]
        for in_channels, out_channels, rate in zip(
            channels[:-1], channels[1:], downsample_rates
        ):
            self.generator.extend(
                [
                    Conv1dS(in_channels, out_channels, 2 * rate, stride=rate),
                    ResStack(out_channels),
                    nn.LeakyReLU(0.2),
                ]
            )
        self.generator.append(Conv1dS(channels[-1], 2 * latent_dim, 3))

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.generator:
            x = layer(x)
        return x


class AMPBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: list[int],
        causal: bool,
        act_causal: bool,
        snake_logscale: bool,
    ) -> None:
        super().__init__()
        self.convs1 = [
            Conv1d(channels, channels, kernel_size, dilation=d, causal=causal)
            for d in dilations
        ]
        self.convs2 = [
            Conv1d(channels, channels, kernel_size, causal=causal) for _ in dilations
        ]
        self.activations = [
            Activation1d(channels, act_causal, snake_logscale)
            for _ in range(2 * len(dilations))
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for i, (conv1, conv2) in enumerate(zip(self.convs1, self.convs2)):
            residual = conv1(self.activations[2 * i](x))
            x = x + conv2(self.activations[2 * i + 1](residual))
        return x
