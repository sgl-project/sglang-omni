# SPDX-License-Identifier: Apache-2.0 AND MIT
# Adapted from HiFi-GAN, ParallelWaveGAN, BigVGAN, and Edward Dixon's Snake.
# Original notices are distributed in THIRD_PARTY_NOTICES.md and licenses/.
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Hift layers for MiniCPM-o."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils.parametrizations import weight_norm


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def masked(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Zero masked positions, or return the input unchanged when no mask is needed."""
    if mask is None:
        return x
    else:
        return x * mask


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
    else:
        pass


class Snake(nn.Module):

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ) -> None:
        super(Snake, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-09

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        else:
            pass
        x = x + 1.0 / (alpha + self.no_div_by_zero) * torch.pow(torch.sin(x * alpha), 2)
        return x


class ResBlock(torch.nn.Module):

    def __init__(
        self,
        channels: int = 512,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for dilation in dilations:
            self.convs1.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        padding=get_padding(kernel_size, dilation),
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
            )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        self.activations1 = nn.ModuleList(
            [Snake(channels, alpha_logscale=False) for _ in range(len(self.convs1))]
        )
        self.activations2 = nn.ModuleList(
            [Snake(channels, alpha_logscale=False) for _ in range(len(self.convs2))]
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = masked(self.convs1[idx](xt), mask)
            xt = self.activations2[idx](xt)
            xt = masked(self.convs2[idx](xt), mask)
            x = xt + x
        return x


class SineGen2(torch.nn.Module):

    def __init__(
        self,
        samp_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ) -> None:
        super(SineGen2, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale

    def f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def f02sine(self, f0_values: torch.Tensor) -> torch.Tensor:
        rad_values = f0_values / self.sampling_rate % 1
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        rad_values = torch.nn.functional.interpolate(
            rad_values.transpose(1, 2),
            scale_factor=1 / self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
        phase = torch.nn.functional.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        return torch.sin(phase)

    def forward(
        self, f0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fn = torch.multiply(
            f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device)
        )
        sine_waves = self.f02sine(fn) * self.sine_amp
        uv = self.f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return (sine_waves, uv, noise)


class SourceModuleHnNSF2(torch.nn.Module):

    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0,
    ) -> None:
        super(SourceModuleHnNSF2, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen2(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshod,
        )
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return (sine_merge, noise, uv)
