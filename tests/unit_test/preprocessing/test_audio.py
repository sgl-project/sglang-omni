# SPDX-License-Identifier: Apache-2.0
"""Tests for model-agnostic audio decoding."""

import io

import numpy as np
import pytest
import soundfile as sf

from sglang_omni.preprocessing.audio import _decode_audio_bytes_av, _parse_wav_bytes


def _encode_audio(
    audio: np.ndarray, sample_rate: int, container_format: str, subtype: str
) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=container_format, subtype=subtype)
    return buffer.getvalue()


@pytest.mark.parametrize("channels", [1, 2])
def test_flac_decode_matches_pcm_wav_duration_and_amplitude(channels: int) -> None:
    sample_rate = 16_000
    signal = np.where(np.arange(sample_rate) % 2, -0.125, 0.125).astype(np.float32)
    source = signal if channels == 1 else np.column_stack((signal, signal / 2))

    flac = _encode_audio(source, sample_rate, "FLAC", "PCM_16")
    wav = _encode_audio(source, sample_rate, "WAV", "PCM_16")
    decoded_flac, flac_rate = _decode_audio_bytes_av(flac)
    decoded_wav, wav_rate = _parse_wav_bytes(wav)

    assert flac_rate == wav_rate == sample_rate
    assert decoded_flac.shape == decoded_wav.shape == (sample_rate,)
    np.testing.assert_array_equal(decoded_flac, decoded_wav)


def test_av_decode_preserves_float_amplitude() -> None:
    sample_rate = 8_000
    source = np.where(np.arange(sample_rate) % 2, -1.5, 1.5).astype(np.float32)
    wav = _encode_audio(source, sample_rate, "WAV", "FLOAT")

    decoded, decoded_rate = _decode_audio_bytes_av(wav)

    assert decoded_rate == sample_rate
    np.testing.assert_array_equal(decoded, source)
