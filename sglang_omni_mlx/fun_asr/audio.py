# SPDX-License-Identifier: Apache-2.0
"""Fun-ASR audio front end in numpy: Kaldi log-mel fbank, low-frame-rate stacking, token counts.

Matches funasr's WavFrontend (torchaudio.compliance.kaldi.fbank, then apply_lfr)
without PyTorch.
"""

from __future__ import annotations

import math

import numpy as np

SAMPLE_RATE = 16000
MEL_BIN_COUNT = 80
FRAME_LENGTH_MS = 25.0
FRAME_SHIFT_MS = 10.0
PREEMPHASIS = 0.97
MEL_LOW_HZ = 20.0
# Low frame rate: stack 7 fbank frames every 6, so the encoder sees 560 features.
STACKED_FRAME_COUNT = 7
STACK_STRIDE = 6
# The encoder and adaptor halve the stacked frame rate three times.
ADAPTOR_HALVING_COUNT = 3
# funasr scales float audio to the int16 range before the fbank.
PCM16_SCALE = float(1 << 15)
LOG_FLOOR = float(np.finfo(np.float32).eps)
MAX_AUDIO_SECONDS = 30.0


def round_up_to_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def mel_scale(frequency_hz: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log1p(frequency_hz / 700.0)


def kaldi_mel_banks(padded_window_size: int) -> np.ndarray:
    """Kaldi triangular mel filters, [mel_bins, padded_window_size // 2 + 1]."""
    fft_bin_count = padded_window_size // 2
    fft_bin_width = SAMPLE_RATE / padded_window_size
    mel_low = mel_scale(np.float64(MEL_LOW_HZ))
    mel_high = mel_scale(np.float64(SAMPLE_RATE / 2))
    mel_delta = (mel_high - mel_low) / (MEL_BIN_COUNT + 1)
    left = mel_low + np.arange(MEL_BIN_COUNT)[:, None] * mel_delta
    center = left + mel_delta
    right = center + mel_delta
    mel = mel_scale(fft_bin_width * np.arange(fft_bin_count))[None, :]
    up_slope = (mel - left) / (center - left)
    down_slope = (right - mel) / (right - center)
    banks = np.maximum(0.0, np.minimum(up_slope, down_slope))
    # Kaldi leaves the Nyquist bin out of every filter.
    return np.pad(banks, ((0, 0), (0, 1))).astype(np.float32)


def log_fbank(samples: np.ndarray) -> np.ndarray:
    """80-bin Kaldi log-mel fbank of 16 kHz float samples, [frames, mel_bins].

    Hamming window, no dither, DC offset removed, snip_edges framing. Like
    WavFrontend, a clip shorter than one frame uses the whole clip as its window.
    """
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1) * PCM16_SCALE
    frame_length_ms = min(FRAME_LENGTH_MS, len(waveform) / SAMPLE_RATE * 1000)
    window_size = int(SAMPLE_RATE * frame_length_ms * 0.001)
    window_shift = int(SAMPLE_RATE * FRAME_SHIFT_MS * 0.001)
    if window_size < 2 or len(waveform) < window_size:
        raise ValueError("audio is too short to transcribe")
    else:
        pass
    padded_window_size = round_up_to_power_of_two(window_size)
    frame_count = 1 + (len(waveform) - window_size) // window_shift
    frames = np.lib.stride_tricks.sliding_window_view(waveform, window_size)[
        ::window_shift
    ][:frame_count]
    frames = frames - frames.mean(axis=1, keepdims=True)
    previous = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = frames - PREEMPHASIS * previous
    window = np.hamming(window_size).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, n=padded_window_size)
    power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
    mel_energies = power @ kaldi_mel_banks(padded_window_size).T
    return np.log(np.maximum(mel_energies, LOG_FLOOR)).astype(np.float32)


def stack_low_frame_rate(fbank: np.ndarray) -> np.ndarray:
    """funasr apply_lfr: 7 frames every 6, edges repeated, [stacked_frames, 7 * mel_bins]."""
    frame_count = fbank.shape[0]
    stacked_frame_count = math.ceil(frame_count / STACK_STRIDE)
    left_pad = (STACKED_FRAME_COUNT - 1) // 2
    padded = np.concatenate([np.repeat(fbank[:1], left_pad, axis=0), fbank])
    # A window running past the end repeats the last frame.
    indices = np.minimum(
        np.arange(stacked_frame_count)[:, None] * STACK_STRIDE
        + np.arange(STACKED_FRAME_COUNT)[None, :],
        padded.shape[0] - 1,
    )
    return padded[indices].reshape(stacked_frame_count, -1)


def audio_features(samples: np.ndarray) -> np.ndarray:
    """Encoder input for one clip, [stacked_frames, 560]."""
    return stack_low_frame_rate(log_fbank(samples))


def token_count(stacked_frame_count: int) -> int:
    """Audio placeholder tokens for a clip of stacked_frame_count encoder frames."""
    count = stacked_frame_count
    for _ in range(ADAPTOR_HALVING_COUNT):
        count = (count + 1) // 2
    return count
