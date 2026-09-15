# SPDX-License-Identifier: Apache-2.0
"""GPU log-mel for the Qwen3-ASR encoder stream.

Cache-miss request building used to run the checkpoint feature extractor
on the host (STFT + mel) and then H2D the fbank. That CPU FFT is the
unique-input limiter at concurrency 8 to 32. This module follows the
extractor's torch fbank helper (Nyquist drop, Slaney, log10, max-8,
(x+4)/4) on the encoder stream from a pinned waveform. CPU matches
the extractor; CUDA uses cuFFT/cuBLAS and can move a small fraction of
bins by one ULP after the tower cast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AudioFrontend:
    n_fft: int
    hop_length: int
    n_mels: int
    dither: float
    # note (guozhihao-224): layout is (n_freq, n_mels); log_mel does
    # filters.T @ mag.
    mel_filters: torch.Tensor
    hann_window: torch.Tensor | None = None

    def materialize(self, device: torch.device, dtype: torch.dtype) -> None:
        """Keep filters and the Hann window on the waveform device."""
        if self.mel_filters.device != device or self.mel_filters.dtype != dtype:
            self.mel_filters = self.mel_filters.to(device=device, dtype=dtype)
        if (
            self.hann_window is None
            or self.hann_window.device != device
            or self.hann_window.dtype != dtype
        ):
            self.hann_window = torch.hann_window(self.n_fft, device=device, dtype=dtype)


def bind_audio_frontend(model: Any, extractor: Any) -> AudioFrontend:
    """Attach hop length, mel filters, and Hann window from the checkpoint extractor.

    Fails at bind time so a missing extractor cannot reach the first
    unique-input request.
    """
    if extractor is None:
        raise ValueError("Qwen3-ASR GPU mel requires a feature extractor")
    try:
        hop_length = extractor.hop_length
        filters = extractor.mel_filters
    except AttributeError as exc:
        raise ValueError(
            "Qwen3-ASR feature extractor is missing hop_length or mel_filters"
        ) from exc
    if hop_length <= 0:
        raise ValueError(
            f"Qwen3-ASR feature extractor has an invalid hop length: {hop_length}"
        )
    if filters is None:
        raise ValueError("Qwen3-ASR feature extractor has no mel_filters")
    mel_filters = torch.as_tensor(filters, dtype=torch.float32)
    frontend = AudioFrontend(
        n_fft=extractor.n_fft,
        hop_length=hop_length,
        n_mels=extractor.feature_size,
        dither=extractor.dither,
        mel_filters=mel_filters,
        hann_window=torch.hann_window(
            extractor.n_fft, device=mel_filters.device, dtype=torch.float32
        ),
    )
    model._audio_frontend = frontend
    return frontend


def log_mel_spectrogram(
    waveform: torch.Tensor,
    frontend: AudioFrontend,
) -> torch.Tensor:
    """Return log-mel of shape n_mels by n_frames on the waveform device.

    Follows the checkpoint feature extractor's fbank for a 1-D float32
    waveform: drop the Nyquist STFT bin, Slaney mel, log10, max-8, then (x+4)/4.
    CPU is bit-identical to the extractor; CUDA STFT/GEMM can differ by ULP.
    """
    wave = waveform.reshape(-1).to(dtype=torch.float32)
    frontend.materialize(wave.device, wave.dtype)
    if frontend.dither != 0.0:
        wave = wave + frontend.dither * torch.randn(
            wave.shape, dtype=wave.dtype, device=wave.device
        )
    stft = torch.stft(
        wave,
        frontend.n_fft,
        frontend.hop_length,
        window=frontend.hann_window,
        return_complex=True,
    )
    # note (guozhihao-224): drop the Nyquist bin so frames equal samples/hop;
    # the request builder estimates tokens from hop length and must not pad
    # to a 30s window (transformers issue 26241). contiguous() because the
    # slice is strided; rocm gemm on that view is the transformers regression.
    magnitudes = (stft[..., :-1].abs() ** 2).contiguous()
    mel_spec = frontend.mel_filters.T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0
