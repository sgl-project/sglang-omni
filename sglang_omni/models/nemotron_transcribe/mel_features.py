# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatible log-mel features and per-channel normalization."""

from __future__ import annotations

import math

import torch
from torch import nn

STD_FLOOR = 1e-5
LOG_GUARD = 2.0**-24


def normalize_per_feature(
    features: torch.Tensor, frame_counts: torch.Tensor
) -> torch.Tensor:
    """Per-bin mean/variance over valid frames. features is (batch, bins, time)."""
    batch, _, frames = features.shape
    if frames == 0:
        return features
    else:
        pass
    positions = (
        torch.arange(frames, device=features.device).unsqueeze(0).expand(batch, frames)
    )
    valid = positions < frame_counts.unsqueeze(1)
    denominator = valid.sum(dim=1)
    reference = features[:, :, 0].masked_fill((denominator == 0).unsqueeze(1), 0.0)
    centered = torch.where(valid.unsqueeze(1), features - reference.unsqueeze(2), 0.0)
    mean = reference + centered.sum(dim=2) / denominator.clamp_min(1).unsqueeze(1)
    squared = torch.where(
        valid.unsqueeze(1), features - mean.unsqueeze(2), 0.0
    ).square()
    std = torch.sqrt(squared.sum(dim=2) / (denominator.unsqueeze(1) - 1.0))
    std = std.masked_fill(std.isnan(), 0.0) + STD_FLOOR
    normalized = (features - mean.unsqueeze(2)) / std.unsqueeze(2)
    return normalized.masked_fill(~valid.unsqueeze(1), 0.0)


def normalize_features(
    features: torch.Tensor, frame_counts: torch.Tensor, mode: str | None
) -> torch.Tensor:
    """Apply a frontend normalize mode. NA and None leave features unchanged."""
    if mode is None or mode in ("", "NA"):
        return features
    elif mode == "per_feature":
        return normalize_per_feature(features, frame_counts)
    else:
        raise ValueError(f"unsupported feature normalize mode {mode!r}")


def slaney_mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> torch.Tensor:
    """Slaney-normalized mel matrix shaped (1, n_mels, n_fft // 2 + 1)."""

    def hertz_to_mel(frequencies: torch.Tensor) -> torch.Tensor:
        spacing = 200.0 / 3.0
        log_hertz = 1000.0
        log_mel = log_hertz / spacing
        log_step = math.log(6.4) / 27.0
        linear = frequencies / spacing
        logarithmic = log_mel + torch.log(frequencies / log_hertz) / log_step
        return torch.where(frequencies >= log_hertz, logarithmic, linear)

    def mel_to_hertz(mels: torch.Tensor) -> torch.Tensor:
        spacing = 200.0 / 3.0
        log_hertz = 1000.0
        log_mel = log_hertz / spacing
        log_step = math.log(6.4) / 27.0
        linear = spacing * mels
        logarithmic = log_hertz * torch.exp(log_step * (mels - log_mel))
        return torch.where(mels >= log_mel, logarithmic, linear)

    fft_frequencies = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    mel_points = torch.linspace(
        hertz_to_mel(torch.tensor(fmin)),
        hertz_to_mel(torch.tensor(fmax)),
        n_mels + 2,
    )
    hertz_points = mel_to_hertz(mel_points)
    differences = hertz_points[1:] - hertz_points[:-1]
    ramps = hertz_points.unsqueeze(1) - fft_frequencies.unsqueeze(0)
    weights = torch.zeros(n_mels, n_fft // 2 + 1)
    for index in range(n_mels):
        lower = -ramps[index] / differences[index]
        upper = ramps[index + 2] / differences[index + 1]
        weights[index] = torch.maximum(
            torch.zeros_like(lower), torch.minimum(lower, upper)
        )
    enorm = 2.0 / (hertz_points[2 : n_mels + 2] - hertz_points[:n_mels])
    return (weights * enorm.unsqueeze(1)).unsqueeze(0)


class LogMelFrontend(nn.Module):
    """Waveform to log-mel. Buffers are named fb and window to match the checkpoint."""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_size: float = 0.025,
        window_stride: float = 0.01,
        n_fft: int = 512,
        features: int = 128,
        preemph: float = 0.97,
        mag_power: float = 2.0,
        log_zero_guard: float = LOG_GUARD,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = int(window_size * sample_rate)
        self.hop_length = int(window_stride * sample_rate)
        self.n_fft = n_fft
        self.preemph = preemph
        self.mag_power = mag_power
        self.log_zero_guard = log_zero_guard
        window = torch.hann_window(self.win_length, periodic=False)
        self.register_buffer("window", window)
        banks = slaney_mel_filterbank(
            sample_rate, n_fft, features, 0.0, sample_rate / 2
        )
        self.register_buffer("fb", banks)

    def frame_counts(self, sample_counts: torch.Tensor) -> torch.Tensor:
        """Valid mel frames for centered STFT, excluding the extra trailing bin."""
        return torch.div(sample_counts, self.hop_length, rounding_mode="floor")

    def forward(
        self, waveforms: torch.Tensor, sample_counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return log-mel (batch, bins, time) and valid frame counts."""
        waveforms = waveforms.to(torch.float32)
        sample_counts = sample_counts.to(device=waveforms.device)
        frame_counts = self.frame_counts(sample_counts)
        if self.preemph:
            positions = torch.arange(
                waveforms.shape[1], device=waveforms.device
            ).unsqueeze(0)
            valid = positions < sample_counts.unsqueeze(1)
            emphasized = torch.cat(
                (waveforms[:, :1], waveforms[:, 1:] - self.preemph * waveforms[:, :-1]),
                dim=1,
            )
            waveforms = emphasized.masked_fill(~valid, 0.0)
        else:
            pass
        with torch.amp.autocast(waveforms.device.type, enabled=False):
            spectrum = torch.stft(
                waveforms,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=True,
                window=self.window.to(dtype=torch.float32, device=waveforms.device),
                return_complex=True,
                pad_mode="constant",
            )
        power = torch.view_as_real(spectrum).pow(2).sum(-1).sqrt()
        if self.mag_power != 1.0:
            power = power.pow(self.mag_power)
        else:
            pass
        with torch.amp.autocast(waveforms.device.type, enabled=False):
            mel = torch.matmul(
                self.fb.to(dtype=power.dtype, device=power.device), power
            )
        mel = torch.log(mel + self.log_zero_guard)
        time = mel.shape[-1]
        valid = torch.arange(time, device=mel.device).unsqueeze(
            0
        ) < frame_counts.unsqueeze(1)
        mel = mel.masked_fill(~valid.unsqueeze(1), 0.0)
        return mel.to(self.fb.dtype), frame_counts.to(torch.int64)
