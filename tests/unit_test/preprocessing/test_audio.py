# SPDX-License-Identifier: Apache-2.0
"""Tests for model-agnostic audio decoding."""

import io

import numpy as np
import pytest
import soundfile as sf

from sglang_omni.preprocessing.audio import decode_audio_bytes_av

SAMPLE_RATE = 16_000
# note (ratish): lossy codecs decode through different float paths; one 16-bit step bounds them.
LOSSY_TOLERANCE = 1 / 32768


def encode_audio(
    audio: np.ndarray, container_format: str, subtype: str, sample_rate: int
) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=container_format, subtype=subtype)
    return buffer.getvalue()


def libsndfile_mono(data: bytes) -> np.ndarray:
    """The reference loader's decode: libsndfile floats averaged over channels."""
    audio, _ = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio.mean(axis=1)


def tone(channels: int) -> np.ndarray:
    time = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    left = 0.5 * np.sin(2 * np.pi * 440 * time)
    right = 0.25 * np.cos(2 * np.pi * 660 * time)
    return left if channels == 1 else np.column_stack((left, right))


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize(
    ("container_format", "subtype"),
    [
        ("WAV", "PCM_U8"),
        ("WAV", "PCM_16"),
        ("WAV", "PCM_24"),
        ("WAV", "PCM_32"),
        ("WAV", "FLOAT"),
        ("FLAC", "PCM_16"),
        ("FLAC", "PCM_24"),
    ],
)
def test_lossless_decode_equals_libsndfile(
    container_format: str, subtype: str, channels: int
) -> None:
    data = encode_audio(tone(channels), container_format, subtype, SAMPLE_RATE)

    decoded, sample_rate = decode_audio_bytes_av(data)

    assert sample_rate == SAMPLE_RATE
    np.testing.assert_array_equal(decoded, libsndfile_mono(data))


@pytest.mark.parametrize("channels", [1, 2])
def test_planar_lossy_decode_matches_libsndfile(channels: int) -> None:
    data = encode_audio(tone(channels), "OGG", "VORBIS", SAMPLE_RATE)

    decoded, sample_rate = decode_audio_bytes_av(data)

    reference = libsndfile_mono(data)
    assert sample_rate == SAMPLE_RATE
    assert decoded.shape == reference.shape
    np.testing.assert_allclose(decoded, reference, rtol=0, atol=LOSSY_TOLERANCE)


def test_float_audio_above_full_scale_keeps_its_amplitude() -> None:
    source = np.where(np.arange(8_000) % 2, -1.5, 1.5).astype(np.float32)
    data = encode_audio(source, "WAV", "FLOAT", 8_000)

    decoded, sample_rate = decode_audio_bytes_av(data)

    assert sample_rate == 8_000
    np.testing.assert_array_equal(decoded, source)
