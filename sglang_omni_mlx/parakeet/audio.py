# SPDX-License-Identifier: Apache-2.0
"""Parakeet audio front end in numpy: normalized log-mel features and long-audio splitting."""

from __future__ import annotations

import numpy as np

from sglang_omni_mlx.wav import SAMPLE_RATE

FFT_SIZE = 512
WINDOW_LENGTH = 400
HOP_LENGTH = 160
PREEMPHASIS = 0.97
LOG_GUARD = 2.0**-24
NORMALIZE_EPS = 1e-5
# Slaney mel scale: linear below 1 kHz, logarithmic above.
SLANEY_LINEAR_HZ_PER_MEL = 200.0 / 3
SLANEY_LOG_START_HZ = 1000.0
SLANEY_LOG_STEP = np.log(6.4) / 27.0
# Long uploads are transcribed in chunks of at most this many seconds, cut at the
# quietest 100 ms window in the last two seconds before each boundary.
MAX_CHUNK_SECONDS = 120.0
CUT_SEARCH_SECONDS = 2.0
CUT_ENERGY_WINDOW = 1600
MAX_AUDIO_SECONDS = 3600.0
# A chunk whose peak stays below -60 dBFS holds no speech,
# and decoding it anyway can emit hallucinated tokens.
SILENT_PEAK = 1e-3


def hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=np.float64)
    linear = frequencies / SLANEY_LINEAR_HZ_PER_MEL
    log_start_mel = SLANEY_LOG_START_HZ / SLANEY_LINEAR_HZ_PER_MEL
    logarithmic = (
        log_start_mel
        + np.log(np.maximum(frequencies, SLANEY_LOG_START_HZ) / SLANEY_LOG_START_HZ)
        / SLANEY_LOG_STEP
    )
    return np.where(frequencies >= SLANEY_LOG_START_HZ, logarithmic, linear)


def mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    log_start_mel = SLANEY_LOG_START_HZ / SLANEY_LINEAR_HZ_PER_MEL
    linear = mels * SLANEY_LINEAR_HZ_PER_MEL
    logarithmic = SLANEY_LOG_START_HZ * np.exp(SLANEY_LOG_STEP * (mels - log_start_mel))
    return np.where(mels >= log_start_mel, logarithmic, linear)


def mel_filter_bank(mel_bin_count: int) -> np.ndarray:
    """Slaney-scale, Slaney-normalized triangular filters, [mel_bins, FFT_SIZE // 2 + 1]."""
    fft_frequencies = np.linspace(0, SAMPLE_RATE / 2, 1 + FFT_SIZE // 2)
    mel_frequencies = mel_to_hz(
        np.linspace(hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE / 2), mel_bin_count + 2)
    )
    widths = np.diff(mel_frequencies)
    ramps = mel_frequencies[:, None] - fft_frequencies[None, :]
    rising = -ramps[:-2] / widths[:-1, None]
    falling = ramps[2:] / widths[1:, None]
    filters = np.maximum(0.0, np.minimum(rising, falling))
    filters *= (2.0 / (mel_frequencies[2:] - mel_frequencies[:-2]))[:, None]
    return filters.astype(np.float32)


def symmetric_hann_window() -> np.ndarray:
    """The 400-sample symmetric Hann window, centered in an FFT_SIZE frame."""
    window = np.hanning(WINDOW_LENGTH).astype(np.float32)
    left = (FFT_SIZE - WINDOW_LENGTH) // 2
    return np.pad(window, (left, FFT_SIZE - WINDOW_LENGTH - left))


def log_mel_features(samples: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Per-bin normalized log-mel features of one clip, [frames, mel_bins].

    Frames are centered with zero padding. The frame centered past the last
    hop is dropped; mean and variance come from the frames that remain.
    """
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(waveform) < FFT_SIZE:
        waveform = np.pad(waveform, (0, FFT_SIZE - len(waveform)))
    else:
        pass
    emphasized = np.concatenate(
        [waveform[:1], waveform[1:] - np.float32(PREEMPHASIS) * waveform[:-1]]
    )
    padded = np.pad(emphasized, FFT_SIZE // 2)
    frame_count = 1 + len(emphasized) // HOP_LENGTH
    frames = np.lib.stride_tricks.sliding_window_view(padded, FFT_SIZE)[::HOP_LENGTH][
        :frame_count
    ]
    spectrum = np.fft.rfft(frames * symmetric_hann_window(), axis=-1)
    power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
    log_mel = np.log(power @ mel_filters.T + np.float32(LOG_GUARD))
    valid_count = len(waveform) // HOP_LENGTH
    valid = log_mel[:valid_count]
    mean = valid.mean(axis=0)
    std = np.sqrt(((valid - mean) ** 2).sum(axis=0) / (valid_count - 1))
    return ((valid - mean) / (std + np.float32(NORMALIZE_EPS))).astype(np.float32)


def chunk_spans(samples: np.ndarray) -> list[tuple[int, int]]:
    """Half-open sample ranges of at most MAX_CHUNK_SECONDS covering the clip."""
    total = len(samples)
    chunk = int(MAX_CHUNK_SECONDS * SAMPLE_RATE)
    search = int(CUT_SEARCH_SECONDS * SAMPLE_RATE)
    spans: list[tuple[int, int]] = []
    start = 0
    while total - start > chunk:
        boundary = start + chunk
        region = samples[boundary - search : boundary]
        window_count = len(region) // CUT_ENERGY_WINDOW
        energies = np.mean(
            np.square(
                region[: window_count * CUT_ENERGY_WINDOW]
                .reshape(window_count, CUT_ENERGY_WINDOW)
                .astype(np.float64)
            ),
            axis=1,
        )
        cut = (
            boundary
            - search
            + int(np.argmin(energies)) * CUT_ENERGY_WINDOW
            + CUT_ENERGY_WINDOW // 2
        )
        spans.append((start, cut))
        start = cut
    spans.append((start, total))
    return spans


def is_silent(samples: np.ndarray) -> bool:
    return len(samples) == 0 or float(np.max(np.abs(samples))) < SILENT_PEAK
