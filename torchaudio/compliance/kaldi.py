from __future__ import annotations

import math

import torch


def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, device=device, dtype=dtype)
    mel_min = _hz_to_mel(freqs.new_tensor(20.0))
    mel_max = _hz_to_mel(freqs.new_tensor(sample_rate / 2))
    mels = torch.linspace(mel_min, mel_max, n_mels + 2, device=device, dtype=dtype)
    hz = _mel_to_hz(mels)

    filters = []
    for index in range(n_mels):
        left, center, right = hz[index], hz[index + 1], hz[index + 2]
        up = (freqs - left) / (center - left).clamp_min(1e-6)
        down = (right - freqs) / (right - center).clamp_min(1e-6)
        filters.append(torch.maximum(torch.minimum(up, down), freqs.new_zeros(())))
    return torch.stack(filters, dim=0)


def fbank(
    waveform: torch.Tensor,
    *,
    num_mel_bins: int = 80,
    sample_frequency: int = 16000,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    dither: float = 0.0,
    **__,
) -> torch.Tensor:
    if waveform.ndim == 2:
        waveform = waveform[0]
    if waveform.ndim != 1:
        raise ValueError(f"fbank expects 1D or 2D waveform, got {tuple(waveform.shape)}")

    waveform = waveform.to(dtype=torch.float32)
    if dither:
        waveform = waveform + torch.randn_like(waveform) * float(dither)
    win_length = max(int(round(sample_frequency * frame_length / 1000.0)), 1)
    hop_length = max(int(round(sample_frequency * frame_shift / 1000.0)), 1)
    n_fft = 1 << max(1, math.ceil(math.log2(win_length)))
    window = torch.hann_window(win_length, device=waveform.device, dtype=waveform.dtype)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        return_complex=True,
    )
    power = spec.abs().pow(2).transpose(0, 1)
    filters = _mel_filterbank(
        sample_rate=int(sample_frequency),
        n_fft=n_fft,
        n_mels=int(num_mel_bins),
        device=power.device,
        dtype=power.dtype,
    )
    mel = power @ filters.transpose(0, 1)
    return torch.log(mel.clamp_min(1e-10))


__all__ = ["fbank"]
