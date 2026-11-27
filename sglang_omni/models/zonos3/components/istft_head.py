# SPDX-License-Identifier: Apache-2.0
"""The iSTFT head of the Zonos3 codec decoder, which turns trunk states at 240 Hz into internal-rate audio."""

from __future__ import annotations

import torch
from torch import nn

from sglang_omni.models.zonos3.components.codec_decoder_kernels import convnext_stack
from sglang_omni.models.zonos3.components.codec_layers import ConvNeXtBlock

ISTFT_HEAD_DEPTH = 8
ISTFT_HEAD_WIDTH = 768
ISTFT_FFT_SIZES = (736, 2208)
ISTFT_HOP_SAMPLES = 184
ISTFT_MIXER_KERNEL_SIZE = 127
MAX_SPECTRAL_MAGNITUDE = 1e2
ISTFT_HEAD_LAYER_NORM_EPSILON = 1e-6


class IstftHead(nn.Module):
    """ConvNeXt trunk, two-resolution log-magnitude and phase iSTFT, FIR branch mixer."""

    def __init__(self, convnext_hidden_channels: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ConvNeXtBlock(ISTFT_HEAD_WIDTH, convnext_hidden_channels)
            for _ in range(ISTFT_HEAD_DEPTH)
        )
        self.norm = nn.LayerNorm(ISTFT_HEAD_WIDTH, eps=ISTFT_HEAD_LAYER_NORM_EPSILON)
        self.outs = nn.ModuleList(
            nn.Linear(ISTFT_HEAD_WIDTH, fft_size + 2) for fft_size in ISTFT_FFT_SIZES
        )
        self.mixer = nn.Conv1d(
            len(ISTFT_FFT_SIZES),
            1,
            ISTFT_MIXER_KERNEL_SIZE,
            padding=ISTFT_MIXER_KERNEL_SIZE // 2,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map (B, 768, T) states at 240 Hz to a (B, T * 184) waveform at the internal rate."""
        frame_states = convnext_stack(
            self.blocks, hidden_states.to(self.blocks[0].pwconv1.weight.dtype)
        )
        return self.synthesize(
            self.spectra(frame_states), frame_states.shape[1] * ISTFT_HOP_SAMPLES
        )

    def spectra(self, frame_states: torch.Tensor) -> list[torch.Tensor]:
        """Complex (B, bins, T) spectra per resolution of the trunk blocks' frame-major (B, T, 768) output."""
        normalized_states = self.norm(frame_states.float())
        spectra: list[torch.Tensor] = []
        for projection in self.outs:
            log_magnitude, phase = (
                projection(normalized_states).transpose(1, 2).chunk(2, dim=1)
            )
            magnitude = torch.exp(log_magnitude).clamp(max=MAX_SPECTRAL_MAGNITUDE)
            spectra.append(torch.polar(magnitude, phase))
        return spectra

    def synthesize(
        self, spectra: list[torch.Tensor], sample_count: int
    ) -> torch.Tensor:
        """Inverse STFT of each resolution's (B, bins, T) spectrum, mixed to a (B, sample_count) waveform."""
        branch_waveforms: list[torch.Tensor] = []
        for fft_size, spectrum in zip(ISTFT_FFT_SIZES, spectra, strict=True):
            window = torch.hann_window(fft_size, periodic=True, device=spectrum.device)
            branch_waveforms.append(
                torch.istft(
                    spectrum,
                    fft_size,
                    ISTFT_HOP_SAMPLES,
                    fft_size,
                    window,
                    center=True,
                    length=sample_count,
                )
            )
        return self.mixer(torch.stack(branch_waveforms, dim=1))[:, 0]
