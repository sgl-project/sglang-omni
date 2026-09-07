# SPDX-License-Identifier: MIT
# Derived from mlx-audio (Copyright 2025 Prince Canuma and contributors).
"""MLX signal-processing primitives for the Qwen3-TTS speaker encoder.

Only what Qwen3-TTS needs: a Hann-window STFT, a Slaney-normalised mel
filterbank, and the log-mel front end that feeds the ECAPA-TDNN speaker
encoder. Kept separate from the models so it can be tested against a
torchaudio/librosa reference on its own.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=None)
def hann_window(size: int, periodic: bool = False) -> mx.array:
    """Hann window; ``periodic=False`` matches torch's default ``hann_window``."""
    denom = size if periodic else size - 1
    return mx.array(
        [0.5 * (1 - math.cos(2 * math.pi * n / denom)) for n in range(size)]
    )


def reflect_pad_1d(x: mx.array, padding: int) -> mx.array:
    """Reflect-pad a 1-D signal without repeating the boundary sample."""
    if padding <= 0:
        return x
    prefix = x[1 : padding + 1][::-1]
    suffix = x[-(padding + 1) : -1][::-1]
    return mx.concatenate([prefix, x, suffix])


def stft(
    x: mx.array,
    n_fft: int = 1024,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
) -> mx.array:
    """Short-time Fourier transform of a 1-D signal.

    Returns ``[frames, n_fft // 2 + 1]`` complex coefficients.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    window = hann_window(win_length)
    if window.shape[0] < n_fft:
        window = mx.concatenate([window, mx.zeros((n_fft - window.shape[0],))], axis=0)

    if center:
        x = reflect_pad_1d(x, n_fft // 2)

    num_frames = 1 + (x.shape[0] - n_fft) // hop_length
    if num_frames <= 0:
        raise ValueError(
            f"Signal of length {x.shape[0]} is too short for n_fft={n_fft} "
            f"with hop_length={hop_length} and center={center}"
        )

    frames = mx.as_strided(x, shape=(num_frames, n_fft), strides=(hop_length, 1))
    return mx.fft.rfft(frames * window)


def _hz_to_mel_slaney(freq: float) -> float:
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    if freq >= min_log_hz:
        min_log_mel = min_log_hz / f_sp
        return min_log_mel + math.log(freq / min_log_hz) / (math.log(6.4) / 27.0)
    return freq / f_sp


def _mel_to_hz_slaney(mels: mx.array) -> mx.array:
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return mx.where(
        mels >= min_log_mel,
        min_log_hz * mx.exp(logstep * (mels - min_log_mel)),
        f_sp * mels,
    )


@lru_cache(maxsize=None)
def mel_filters(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> mx.array:
    """Slaney-scale, Slaney-normalised triangular mel filterbank.

    Returns ``[n_mels, n_fft // 2 + 1]``. Built in float64 on the CPU stream
    and cast down, because the float32 path drifts enough from a torchaudio
    reference to shift log-mel values in the last couple of digits.
    """
    if f_max is None:
        f_max = sample_rate / 2

    def build(dtype) -> mx.array:
        n_freqs = n_fft // 2 + 1
        all_freqs = mx.linspace(0, sample_rate // 2, n_freqs, dtype=dtype)
        m_pts = mx.linspace(
            _hz_to_mel_slaney(f_min),
            _hz_to_mel_slaney(f_max),
            n_mels + 2,
            dtype=dtype,
        )
        f_pts = _mel_to_hz_slaney(m_pts)

        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = mx.expand_dims(f_pts, 0) - mx.expand_dims(all_freqs, 1)
        down = (-slopes[:, :-2]) / f_diff[:-1]
        up = slopes[:, 2:] / f_diff[1:]
        filterbank = mx.maximum(mx.zeros_like(down), mx.minimum(down, up))

        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        filterbank = filterbank * mx.expand_dims(enorm, 0)
        return filterbank.moveaxis(0, 1)

    with mx.stream(mx.cpu):
        return build(mx.float64).astype(mx.float32)


def mel_spectrogram(
    audio: mx.array,
    *,
    n_fft: int = 1024,
    num_mels: int = 128,
    sample_rate: int = 24000,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: float = 0.0,
    fmax: float = 12000.0,
) -> mx.array:
    """Log-mel spectrogram, ``[batch, frames, num_mels]``.

    The reference pads by ``(n_fft - hop_size) // 2`` on each side and then
    runs an uncentred STFT, which is not the same as ``center=True``; keep both
    steps or the frame grid shifts.
    """
    if audio.ndim == 1:
        audio = audio[None, :]

    basis = mel_filters(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=num_mels,
        f_min=fmin,
        f_max=fmax,
    )
    padding = (n_fft - hop_size) // 2

    mels = []
    for index in range(audio.shape[0]):
        spec = stft(
            reflect_pad_1d(audio[index], padding),
            n_fft=n_fft,
            hop_length=hop_size,
            win_length=win_size,
            center=False,
        )
        magnitude = mx.sqrt(mx.abs(spec) ** 2 + 1e-9)
        mel = mx.matmul(magnitude, basis.T)
        mels.append(mx.log(mx.clip(mel, 1e-5, None)))

    return mx.stack(mels, axis=0)
