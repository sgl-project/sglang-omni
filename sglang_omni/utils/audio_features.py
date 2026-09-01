# SPDX-License-Identifier: Apache-2.0
"""Kaldi-compliant fbank with the constant tables cached across requests."""

from __future__ import annotations

import functools
import math

import torch


@functools.lru_cache(maxsize=32)
def _mel_banks(
    num_mel_bins: int,
    padded_window_size: int,
    sample_frequency: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mel filterbank padded to the rfft bin count.

    Note (Jiaxin Deng): torchaudio rebuilds this table inside every
    ``kaldi.fbank`` call; under load it is about a fifth of an ASR process.
    """
    import torchaudio.compliance.kaldi as kaldi

    banks, _ = kaldi.get_mel_banks(
        num_mel_bins,
        padded_window_size,
        sample_frequency,
        20.0,
        0.0,
        100.0,
        -500.0,
        1.0,
    )
    banks = banks.to(device=device, dtype=dtype)
    return torch.nn.functional.pad(banks, (0, 1), mode="constant", value=0)


def cached_fbank(
    waveform: torch.Tensor,
    *,
    num_mel_bins: int = 23,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    window_type: str = "povey",
    sample_frequency: float = 16000.0,
) -> torch.Tensor:
    """``kaldi.fbank`` with a cached mel table, for callers that keep the
    remaining options at their Kaldi defaults (no dither, no energy column,
    snip_edges, log fbank over the power spectrum).

    Note (Jiaxin Deng): the defaults every caller shares are inlined so the
    table can be keyed by the few that vary; bit-identity is asserted in tests.
    """
    try:
        import torchaudio.compliance.kaldi as kaldi
    except ImportError:
        return _fallback_fbank(
            waveform,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            window_type=window_type,
            sample_frequency=sample_frequency,
        )

    if not hasattr(kaldi, "_get_waveform_and_window_properties"):
        return kaldi.fbank(
            waveform,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            window_type=window_type,
            sample_frequency=sample_frequency,
            dither=0.0,
            use_energy=False,
            snip_edges=True,
            preemphasis_coefficient=0.97,
            use_power=True,
        )

    device, dtype = waveform.device, waveform.dtype
    (
        wav,
        window_shift,
        window_size,
        padded_window_size,
    ) = kaldi._get_waveform_and_window_properties(
        waveform, 0, sample_frequency, frame_shift, frame_length, True, 0.97
    )
    strided_input, _ = kaldi._get_window(
        wav,
        padded_window_size,
        window_size,
        window_shift,
        window_type,
        0.42,
        True,
        True,
        0.0,
        0.0,
        True,
        0.97,
    )
    spectrum = torch.fft.rfft(strided_input).abs().pow(2.0)
    banks = _mel_banks(
        num_mel_bins, padded_window_size, sample_frequency, device, dtype
    )
    return torch.max(
        torch.mm(spectrum, banks.T), kaldi._get_epsilon(device, dtype)
    ).log()


def _fallback_fbank(
    waveform: torch.Tensor,
    *,
    num_mel_bins: int,
    frame_length: float,
    frame_shift: float,
    window_type: str,
    sample_frequency: float,
) -> torch.Tensor:
    if window_type != "povey":
        raise ImportError("torchaudio is required for non-povey fbank windows")
    if waveform.ndim == 2:
        if waveform.shape[0] != 1:
            raise ValueError("fallback fbank expects mono waveform")
        waveform = waveform[0]
    waveform = waveform.float().contiguous()
    frame_size = int(sample_frequency * frame_length / 1000.0)
    frame_step = int(sample_frequency * frame_shift / 1000.0)
    if waveform.numel() < frame_size:
        waveform = torch.nn.functional.pad(waveform, (0, frame_size - waveform.numel()))
    waveform = torch.cat([waveform[:1], waveform[1:] - 0.97 * waveform[:-1]])
    frames = waveform.unfold(0, frame_size, frame_step)
    window = torch.hann_window(frame_size, periodic=False, device=waveform.device)
    frames = frames * window.pow(0.85)
    spectrum = torch.fft.rfft(frames).abs().pow(2.0)
    banks = _fallback_mel_banks(
        num_mel_bins,
        spectrum.shape[-1],
        sample_frequency,
        waveform.device,
        spectrum.dtype,
    )
    return torch.clamp(spectrum @ banks.T, min=torch.finfo(spectrum.dtype).eps).log()


@functools.lru_cache(maxsize=32)
def _fallback_mel_banks(
    num_mel_bins: int,
    num_fft_bins: int,
    sample_frequency: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    low_freq = 20.0
    high_freq = sample_frequency / 2.0

    def hz_to_mel(freq: float) -> float:
        return 1127.0 * math.log1p(freq / 700.0)

    def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (torch.exp(mel / 1127.0) - 1.0)

    mel_points = torch.linspace(
        hz_to_mel(low_freq),
        hz_to_mel(high_freq),
        num_mel_bins + 2,
        device=device,
        dtype=dtype,
    )
    hz_points = mel_to_hz(mel_points)
    fft_freqs = torch.linspace(
        0.0,
        high_freq,
        num_fft_bins,
        device=device,
        dtype=dtype,
    )
    lower = hz_points[:-2].unsqueeze(1)
    center = hz_points[1:-1].unsqueeze(1)
    upper = hz_points[2:].unsqueeze(1)
    left = (fft_freqs - lower) / torch.clamp(center - lower, min=1e-6)
    right = (upper - fft_freqs) / torch.clamp(upper - center, min=1e-6)
    return torch.clamp(torch.minimum(left, right), min=0.0)
