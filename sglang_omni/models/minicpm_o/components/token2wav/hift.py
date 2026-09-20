# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Kai Hu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Hift for MiniCPM-o."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm

from sglang_omni.models.minicpm_o.components.token2wav.hift_layers import (
    ResBlock,
    SourceModuleHnNSF2,
    init_weights,
)


class ConvRNNF0Predictor(nn.Module):

    def __init__(
        self, num_class: int = 1, in_channels: int = 80, cond_channels: int = 512
    ) -> None:
        super().__init__()
        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(
                nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            nn.ELU(),
        )
        self.classifier = nn.Linear(
            in_features=cond_channels, out_features=self.num_class
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.condnet(x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class HiFTGenerator(nn.Module):

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 24000,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: tuple[int, ...] = (8, 5, 3),
        upsample_kernel_sizes: tuple[int, ...] = (16, 11, 7),
        istft_params: dict[str, int] | None = None,
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5),) * 3,
        source_resblock_kernel_sizes: tuple[int, ...] = (7, 7, 11),
        source_resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5),) * 3,
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        f0_predictor: torch.nn.Module | None = None,
    ) -> None:
        super(HiFTGenerator, self).__init__()
        if sampling_rate != 24000:
            raise ValueError("MiniCPM-o HiFT requires a 24000 Hz sample rate")
        if istft_params is None:
            istft_params = {"n_fft": 16, "hop_len": 4}
        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF2(
            sampling_rate=sampling_rate,
            upsample_scale=np.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold,
        )
        self.f0_upsamp = torch.nn.Upsample(
            scale_factor=np.prod(upsample_rates) * istft_params["hop_len"]
        )
        self.conv_pre = weight_norm(Conv1d(in_channels, base_channels, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        base_channels // 2**i,
                        base_channels // 2 ** (i + 1),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = (1,) + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(
            zip(
                downsample_cum_rates[::-1],
                source_resblock_kernel_sizes,
                source_resblock_dilation_sizes,
            )
        ):
            if u == 1:
                self.source_downs.append(
                    Conv1d(
                        istft_params["n_fft"] + 2, base_channels // 2 ** (i + 1), 1, 1
                    )
                )
            else:
                self.source_downs.append(
                    Conv1d(
                        istft_params["n_fft"] + 2,
                        base_channels // 2 ** (i + 1),
                        u * 2,
                        u,
                        padding=u // 2,
                    )
                )
            self.source_resblocks.append(ResBlock(base_channels // 2 ** (i + 1), k, d))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // 2 ** (i + 1)
            for _, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock(ch, k, d))
        self.conv_post = weight_norm(
            Conv1d(ch, istft_params["n_fft"] + 2, 7, 1, padding=3)
        )
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft_window = torch.from_numpy(
            get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32)
        )
        self.f0_predictor = (
            ConvRNNF0Predictor() if f0_predictor is None else f0_predictor
        )

    def stft(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec = torch.stft(
            x,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(x.device),
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        return (spec[..., 0], spec[..., 1])

    def istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        magnitude = torch.clip(magnitude, max=100.0)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        inverse_transform = torch.istft(
            torch.complex(real, img),
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(magnitude.device),
        )
        return inverse_transform

    def decode(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s_stft_real, s_stft_imag = self.stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, : self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1 :, :])
        x = self.istft(magnitude, phase)
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    @torch.inference_mode()
    def forward(self, speech_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f0 = self.f0_predictor(speech_feat)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        generated_speech = self.decode(x=speech_feat, s=s)
        return (generated_speech, s)
