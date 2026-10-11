# SPDX-License-Identifier: Apache-2.0
"""Whisper log-mel features and window token lengths for MOSS-TD."""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from sglang_omni_mlx.wav import SAMPLE_RATE, read_wav

FFT_SIZE = 400
HOP_LENGTH = 160
MEL_BIN_COUNT = 80
WINDOW_SAMPLE_COUNT = 30 * SAMPLE_RATE
ENCODER_STRIDE = 2
AUDIO_MERGE_SIZE = 4
LOG_MEL_FLOOR = 1e-10
LOG_MEL_DYNAMIC_RANGE = 8.0


def slaney_mel_filter_bank() -> np.ndarray:
    """Slaney-scale, Slaney-normalized filters as used by Whisper."""
    frequency_bin_count = FFT_SIZE // 2 + 1
    linear_step_hz = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / linear_step_hz
    log_step = np.log(6.4) / 27.0

    def hertz_to_mel(frequency_hz: float) -> float:
        if frequency_hz < min_log_hz:
            return frequency_hz / linear_step_hz
        else:
            return min_log_mel + np.log(frequency_hz / min_log_hz) / log_step

    def mel_to_hertz(mel: float) -> float:
        if mel < min_log_mel:
            return linear_step_hz * mel
        else:
            return min_log_hz * np.exp(log_step * (mel - min_log_mel))

    bin_frequencies_hz = np.arange(frequency_bin_count) * SAMPLE_RATE / FFT_SIZE
    mel_max = hertz_to_mel(SAMPLE_RATE / 2.0)
    edges_hz = np.array(
        [
            mel_to_hertz(index * mel_max / (MEL_BIN_COUNT + 1))
            for index in range(MEL_BIN_COUNT + 2)
        ]
    )
    filters = np.zeros((frequency_bin_count, MEL_BIN_COUNT), dtype=np.float32)
    for mel_bin in range(MEL_BIN_COUNT):
        low, center, high = edges_hz[mel_bin : mel_bin + 3]
        lower = (bin_frequencies_hz - low) / (center - low)
        upper = (high - bin_frequencies_hz) / (high - center)
        filters[:, mel_bin] = (
            np.maximum(0.0, np.minimum(lower, upper)) * 2.0 / (high - low)
        )
    return filters


MEL_FILTERS = slaney_mel_filter_bank()
HANN_WINDOW = np.hanning(FFT_SIZE + 1)[:-1].astype(np.float32)


def log_mel(samples: np.ndarray) -> np.ndarray:
    """One padded 30-second window to [mel_bins, frames] log-mel features."""
    padded = np.pad(samples, FFT_SIZE // 2, mode="reflect")
    frame_count = 1 + (len(padded) - FFT_SIZE) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(frame_count, FFT_SIZE),
        strides=(padded.strides[0] * HOP_LENGTH, padded.strides[0]),
        writeable=False,
    )
    power = np.abs(np.fft.rfft(frames * HANN_WINDOW, axis=1)) ** 2
    spectrum = np.log10(np.maximum(power @ MEL_FILTERS, LOG_MEL_FLOOR))[:-1]
    spectrum = np.maximum(spectrum, spectrum.max() - LOG_MEL_DYNAMIC_RANGE)
    return ((spectrum + 4.0) / 4.0).T.astype(np.float32)


def audio_features(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Waveform to padded log-mel windows and their post-merge token lengths."""
    if samples.size == 0:
        raise ValueError("audio must contain at least one sample")
    else:
        pass
    features: list[np.ndarray] = []
    token_lengths: list[int] = []
    token_stride = HOP_LENGTH * ENCODER_STRIDE * AUDIO_MERGE_SIZE
    for start in range(0, len(samples), WINDOW_SAMPLE_COUNT):
        window = samples[start : start + WINDOW_SAMPLE_COUNT]
        token_lengths.append((len(window) - 1) // token_stride + 1)
        if len(window) < WINDOW_SAMPLE_COUNT:
            window = np.pad(window, (0, WINDOW_SAMPLE_COUNT - len(window)))
        else:
            pass
        features.append(log_mel(window.astype(np.float32, copy=False)))
    return np.stack(features), np.asarray(token_lengths, dtype=np.int32)


def decode_audio_wav(wav_bytes: bytes) -> np.ndarray:
    """Mono WAV at any sample rate to 16 kHz float32 samples."""
    samples, sample_rate = read_wav(wav_bytes)
    if sample_rate == SAMPLE_RATE:
        return samples
    else:
        return resample_poly(samples, SAMPLE_RATE, sample_rate).astype(np.float32)
