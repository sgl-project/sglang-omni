from __future__ import annotations

import math

import torch
from torch import nn

from . import functional


class Resample(nn.Module):
    def __init__(self, orig_freq: int, new_freq: int, *_, **__) -> None:
        super().__init__()
        self.orig_freq = int(orig_freq)
        self.new_freq = int(new_freq)

    def forward(self, waveform):
        return functional.resample(waveform, self.orig_freq, self.new_freq)


class Spectrogram(nn.Module):
    def __init__(
        self,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int | None = None,
        pad: int = 0,
        window_fn=torch.hann_window,
        power: float | None = 2.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        **window_kwargs,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.win_length = int(win_length or n_fft)
        self.hop_length = int(hop_length or self.win_length // 2)
        self.pad = int(pad)
        self.power = power
        self.normalized = bool(normalized)
        self.center = bool(center)
        self.pad_mode = pad_mode
        self.onesided = bool(onesided)
        window = window_fn(self.win_length, **window_kwargs)
        self.register_buffer("window", window)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            waveform = torch.nn.functional.pad(waveform, (self.pad, self.pad))
        original_shape = waveform.shape
        flat = waveform.reshape(-1, original_shape[-1])
        spec = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=waveform.device, dtype=torch.float32),
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )
        spec = spec.reshape(*original_shape[:-1], spec.shape[-2], spec.shape[-1])
        if self.power is None:
            return spec
        return spec.abs().pow(float(self.power))


def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float | None,
) -> torch.Tensor:
    f_max = float(sample_rate // 2 if f_max is None else f_max)
    all_freqs = torch.linspace(0, sample_rate // 2, n_fft // 2 + 1)
    m_min = _hz_to_mel(torch.tensor(float(f_min)))
    m_max = _hz_to_mel(torch.tensor(f_max))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts)
    lower = f_pts[:-2].unsqueeze(0)
    center = f_pts[1:-1].unsqueeze(0)
    upper = f_pts[2:].unsqueeze(0)
    freqs = all_freqs.unsqueeze(1)
    down = (freqs - lower) / torch.clamp(center - lower, min=1e-10)
    up = (upper - freqs) / torch.clamp(upper - center, min=1e-10)
    fb = torch.clamp(torch.minimum(down, up), min=0.0)
    enorm = 2.0 / torch.clamp(f_pts[2 : n_mels + 2] - f_pts[:n_mels], min=1e-10)
    return fb * enorm.unsqueeze(0)


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        n_mels: int = 128,
        power: float = 2.0,
        center: bool = True,
        norm: str | None = None,
        mel_scale: str = "htk",
        **kwargs,
    ) -> None:
        del norm, mel_scale
        super().__init__()
        self.spectrogram = Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=power,
            center=center,
            **kwargs,
        )
        mel_fb = _build_mel_filterbank(
            sample_rate=int(sample_rate),
            n_fft=int(n_fft),
            n_mels=int(n_mels),
            f_min=float(f_min),
            f_max=f_max,
        )
        self.register_buffer("mel_fb", mel_fb)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.spectrogram(waveform)
        mel_fb = self.mel_fb.to(device=spec.device, dtype=spec.dtype)
        return torch.matmul(spec.transpose(-2, -1), mel_fb).transpose(-2, -1)


__all__ = ["MelSpectrogram", "Resample", "Spectrogram"]
