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

from collections.abc import Sequence

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
    masked,
)

ISTFT_ENVELOPE_FLOOR = 1e-11


def length_mask(
    frame_lengths: torch.Tensor | None, scale: int, extra: int, like: torch.Tensor
) -> torch.Tensor | None:
    """Valid positions of like, whose rows hold frame_lengths * scale + extra."""
    if frame_lengths is None:
        mask = None
    else:
        positions = torch.arange(like.shape[-1], device=like.device)
        valid = positions < frame_lengths[:, None] * scale + extra
        mask = valid.unsqueeze(1).to(like.dtype)
    return mask


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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = masked(x, mask)
        for layer in self.condnet:
            x = masked(layer(x), mask)
        x = x.transpose(1, 2)
        f0 = torch.abs(self.classifier(x).squeeze(-1))
        # note (liuqihao): zero f0 past a row keeps its last-frame phase interpolation unchanged.
        return f0 if mask is None else f0 * mask[:, 0]


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
        else:
            pass
        if istft_params is None:
            istft_params = {"n_fft": 16, "hop_len": 4}
        else:
            pass
        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.upsample_rates = upsample_rates
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

    def stft(
        self, x: torch.Tensor, sample_lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_fft = self.istft_params["n_fft"]
        hop_len = self.istft_params["hop_len"]
        window = self.stft_window.to(x.device)
        if sample_lengths is None:
            spec = torch.stft(
                x, n_fft, hop_len, n_fft, window=window, return_complex=True
            )
        else:
            # note (liuqihao): each row reflects its own tail, as centered framing does alone.
            half = n_fft // 2
            positions = torch.arange(x.shape[-1] + half, device=x.device)
            last = sample_lengths[:, None] - 1
            source = torch.where(positions <= last, positions, 2 * last - positions)
            x = torch.gather(x, 1, source.clamp(min=0))
            x = F.pad(x[:, None], (half, 0), mode="reflect")[:, 0]
            spec = torch.stft(
                x,
                n_fft,
                hop_len,
                n_fft,
                window=window,
                center=False,
                return_complex=True,
            )
        spec = torch.view_as_real(spec)
        return (spec[..., 0], spec[..., 1])

    def istft(
        self,
        magnitude: torch.Tensor,
        phase: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        magnitude = torch.clip(magnitude, max=100.0)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        spectrum = torch.complex(real, img)
        n_fft = self.istft_params["n_fft"]
        hop_len = self.istft_params["hop_len"]
        window = self.stft_window.to(magnitude.device)
        if frame_mask is None:
            inverse_transform = torch.istft(
                spectrum, n_fft, hop_len, n_fft, window=window
            )
        else:
            # note (liuqihao): per-row window envelopes keep padded frames out of the
            # normalization of each row's last samples.
            frames = torch.fft.irfft(spectrum, n=n_fft, dim=1)
            frames = frames * window[:, None] * frame_mask
            width = (frames.shape[-1] - 1) * hop_len + n_fft
            signal = F.fold(
                frames, (1, width), (1, n_fft), stride=(1, hop_len)
            ).flatten(1)
            envelope = F.fold(
                window.square()[:, None] * frame_mask,
                (1, width),
                (1, n_fft),
                stride=(1, hop_len),
            ).flatten(1)
            normalized = signal / envelope.clamp(min=ISTFT_ENVELOPE_FLOOR)
            inverse_transform = normalized[:, n_fft // 2 : width - n_fft // 2]
        return inverse_transform

    def decode(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        frame_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        stft_scale = int(np.prod(self.upsample_rates))
        sample_lengths = (
            None
            if frame_lengths is None
            else frame_lengths * stft_scale * self.istft_params["hop_len"]
        )
        s_stft_real, s_stft_imag = self.stft(s.squeeze(1), sample_lengths)
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
        s_stft = masked(s_stft, length_mask(frame_lengths, stft_scale, 1, s_stft))
        mask = length_mask(frame_lengths, 1, 0, x)
        x = masked(self.conv_pre(masked(x, mask)), mask)
        scale = 1
        for i in range(self.num_upsamples):
            is_last = i == self.num_upsamples - 1
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            scale *= self.upsample_rates[i]
            if is_last:
                x = self.reflection_pad(x)
            else:
                pass
            # note (liuqihao): the reflection pad widens the last stage by one frame.
            mask = length_mask(frame_lengths, scale, int(is_last), x)
            x = masked(x, mask)
            si = masked(self.source_downs[i](s_stft), mask)
            si = self.source_resblocks[i](si, mask)
            x = x + si
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, mask)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, mask)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, : self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1 :, :])
        x = self.istft(magnitude, phase, mask)
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    @torch.inference_mode()
    def forward(
        self, speech_feat: torch.Tensor, mel_lengths: Sequence[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vocode mel rows; mel_lengths marks each row's valid frames when padded."""
        width = speech_feat.shape[-1]
        if mel_lengths is None or all(length == width for length in mel_lengths):
            frame_lengths = None
        else:
            frame_lengths = torch.tensor(mel_lengths, device=speech_feat.device)
        f0 = self.f0_predictor(
            speech_feat, length_mask(frame_lengths, 1, 0, speech_feat)
        )
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        generated_speech = self.decode(x=speech_feat, s=s, frame_lengths=frame_lengths)
        return (generated_speech, s)
