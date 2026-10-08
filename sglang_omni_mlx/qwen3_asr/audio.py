# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR audio front end: WAV decoding, log-mel, audio token counts."""

from __future__ import annotations

import enum
import struct

import mlx.core as mx
import numpy as np

SAMPLE_RATE = 16000
HOP_LENGTH = 160
FFT_SIZE = 400
MEL_BIN_COUNT = 128
# Qwen3-ASR encodes mel frames in chunks of 100 frames, 13 audio tokens each.
CHUNK_FRAME_COUNT = 100
CHUNK_TOKEN_COUNT = 13
LOG_MEL_FLOOR = 1e-10
LOG_MEL_DYNAMIC_RANGE = 8.0
WAV_FORMAT_PCM = 1
WAV_FORMAT_FLOAT = 3
PCM16_FULL_SCALE = 32768.0


class AudioLayout(enum.Enum):
    """How the audio part of the prompt is built.

    REFERENCE follows the checkpoint's processor. VOXT_SWIFT reproduces Voxt's
    Swift port: one more mel frame, and a token count computed with true
    division, so a partial final chunk is credited with extra tokens.
    """

    REFERENCE = "reference"
    VOXT_SWIFT = "voxt_swift"


def decode_wav(wav_bytes: bytes) -> np.ndarray:
    """16 kHz mono PCM16 or float32 WAV to float32 samples."""
    if len(wav_bytes) < 12 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("audio must be a RIFF/WAVE file")
    else:
        pass
    offset = 12
    audio_format = channel_count = sample_rate = bits_per_sample = None
    while offset + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[offset : offset + 4]
        chunk_size = struct.unpack("<I", wav_bytes[offset + 4 : offset + 8])[0]
        body = wav_bytes[offset + 8 : offset + 8 + chunk_size]
        if chunk_id == b"fmt ":
            audio_format, channel_count, sample_rate = struct.unpack("<HHI", body[:8])
            bits_per_sample = struct.unpack("<H", body[14:16])[0]
        elif chunk_id == b"data":
            if (channel_count, sample_rate) != (1, SAMPLE_RATE):
                raise ValueError("audio must be 16 kHz mono")
            elif (audio_format, bits_per_sample) == (WAV_FORMAT_PCM, 16):
                return (
                    np.frombuffer(body, dtype="<i2").astype(np.float32)
                    / PCM16_FULL_SCALE
                )
            elif (audio_format, bits_per_sample) == (WAV_FORMAT_FLOAT, 32):
                return np.frombuffer(body, dtype="<f4").astype(np.float32)
            else:
                raise ValueError("audio must be PCM16 or float32 WAV")
        else:
            pass
        offset += 8 + chunk_size + (chunk_size & 1)
    raise ValueError("WAV file has no data chunk")


def slaney_mel_filter_bank() -> np.ndarray:
    """Slaney-scale, Slaney-normalized triangular filters, [frequency_bins, mel_bins].

    Built in float32 in the same order as Voxt's Swift front end (MLXAudioCore
    melFilters), so both produce the same filter values.
    """
    float32 = np.float32
    frequency_bin_count = FFT_SIZE // 2 + 1
    linear_step_hz = float32(200.0) / float32(3.0)
    min_log_hz = float32(1000.0)
    min_log_mel = min_log_hz / linear_step_hz
    log_step = float32(np.log(float32(6.4))) / float32(27.0)

    def hertz_to_mel(frequency_hz: np.float32) -> np.float32:
        if frequency_hz < min_log_hz:
            return frequency_hz / linear_step_hz
        else:
            return min_log_mel + float32(np.log(frequency_hz / min_log_hz)) / log_step

    def mel_to_hertz(mel: np.float32) -> np.float32:
        if mel < min_log_mel:
            return linear_step_hz * mel
        else:
            return min_log_hz * float32(np.exp(log_step * (mel - min_log_mel)))

    bin_frequencies_hz = [
        float32(i) * float32(SAMPLE_RATE) / float32(FFT_SIZE)
        for i in range(frequency_bin_count)
    ]
    mel_max = hertz_to_mel(float32(SAMPLE_RATE) / float32(2.0))
    edges_hz = [
        mel_to_hertz(float32(i) * mel_max / float32(MEL_BIN_COUNT + 1))
        for i in range(MEL_BIN_COUNT + 2)
    ]
    filters = np.zeros((frequency_bin_count, MEL_BIN_COUNT), dtype=np.float32)
    for mel_bin in range(MEL_BIN_COUNT):
        low, center, high = (
            edges_hz[mel_bin],
            edges_hz[mel_bin + 1],
            edges_hz[mel_bin + 2],
        )
        normalization = float32(2.0) / (high - low)
        for frequency_bin, frequency_hz in enumerate(bin_frequencies_hz):
            if low <= frequency_hz < center:
                weight = (frequency_hz - low) / (center - low)
            elif center <= frequency_hz <= high:
                weight = (high - frequency_hz) / (high - center)
            else:
                weight = float32(0.0)
            filters[frequency_bin, mel_bin] = weight * normalization
    return filters


def periodic_hann_window() -> np.ndarray:
    denominator = np.float32(FFT_SIZE)
    return np.array(
        [
            np.float32(0.5)
            * (
                np.float32(1.0)
                - np.cos(
                    np.float32(2.0) * np.float32(np.pi) * np.float32(n) / denominator
                )
            )
            for n in range(FFT_SIZE)
        ],
        dtype=np.float32,
    )


MEL_FILTERS = mx.array(slaney_mel_filter_bank())
PERIODIC_HANN_WINDOW = mx.array(periodic_hann_window())


def log_mel(samples: np.ndarray, layout: AudioLayout) -> mx.array:
    """Whisper-style log-mel in float32 on MLX, [mel_bins, frames].

    The same operations as Voxt's Swift front end. The reference drops the final
    centered STFT frame; the Swift layout keeps it.
    """
    if len(samples) < FFT_SIZE:
        # Reflect padding needs more samples than half a window; a tail this
        # short (a realtime cut or a stop right after one) is zero-filled to
        # one window.
        samples = np.pad(samples, (0, FFT_SIZE - len(samples)))
    else:
        pass
    audio = mx.array(samples.astype(np.float32))
    padding = FFT_SIZE // 2
    padded = mx.concatenate(
        [audio[1 : padding + 1][::-1], audio, audio[-padding - 1 : -1][::-1]]
    )
    frame_count = 1 + (padded.shape[0] - FFT_SIZE) // HOP_LENGTH
    frames = mx.as_strided(padded, (frame_count, FFT_SIZE), (HOP_LENGTH, 1))
    power = mx.square(mx.abs(mx.fft.rfft(frames * PERIODIC_HANN_WINDOW, axis=1)))
    if layout is AudioLayout.REFERENCE:
        power = power[:-1]
    else:
        pass
    log_spectrum = mx.log10(mx.maximum(mx.matmul(power, MEL_FILTERS), LOG_MEL_FLOOR))
    log_spectrum = mx.maximum(log_spectrum, log_spectrum.max() - LOG_MEL_DYNAMIC_RANGE)
    return ((log_spectrum + 4.0) / 4.0).T


def reference_token_count(frame_count: int) -> int:
    """Encoder output tokens for frame_count mel frames."""
    remainder_frames = frame_count % CHUNK_FRAME_COUNT
    after_first_conv = (remainder_frames - 1) // 2 + 1
    remainder_tokens = ((after_first_conv - 1) // 2 + 1 - 1) // 2 + 1
    return remainder_tokens + (frame_count // CHUNK_FRAME_COUNT) * CHUNK_TOKEN_COUNT


def swift_token_count(frame_count: int) -> int:
    """Voxt's Swift formula: chunks counted with float32 true division, then truncated."""
    remainder_frames = frame_count % CHUNK_FRAME_COUNT
    after_first_conv = (remainder_frames - 1) // 2 + 1
    remainder_tokens = ((after_first_conv - 1) // 2 + 1 - 1) // 2 + 1
    chunks = np.float32(frame_count) / np.float32(CHUNK_FRAME_COUNT)
    return int(
        np.trunc(np.float32(remainder_tokens) + chunks * np.float32(CHUNK_TOKEN_COUNT))
    )


def token_count(frame_count: int, layout: AudioLayout) -> int:
    if layout is AudioLayout.REFERENCE:
        return reference_token_count(frame_count)
    else:
        return swift_token_count(frame_count)


def peak_is_silent(samples: np.ndarray, peak_threshold: float) -> bool:
    return samples.size == 0 or float(np.max(np.abs(samples))) < peak_threshold
