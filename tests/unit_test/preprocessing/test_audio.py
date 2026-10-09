# SPDX-License-Identifier: Apache-2.0
"""Tests for model-agnostic audio decoding."""

import io

import av
import numpy as np
import pytest
import soundfile as sf

from sglang_omni.preprocessing.audio import (
    decode_audio_bytes,
    decode_audio_bytes_av,
    parse_wav_bytes,
)

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


def tone(channels: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    left = 0.5 * np.sin(2 * np.pi * 440 * time)
    right = 0.25 * np.cos(2 * np.pi * 660 * time)
    return left if channels == 1 else np.column_stack((left, right))


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize(
    ("container_format", "subtype"),
    [
        ("WAV", "PCM_16"),
        ("WAV", "PCM_24"),
        ("WAVEX", "PCM_16"),
        ("FLAC", "PCM_16"),
        ("OGG", "VORBIS"),
        ("MP3", "MPEG_LAYER_III"),
    ],
)
def test_request_audio_decodes_as_the_reference_loader(
    container_format: str, subtype: str, channels: int
) -> None:
    # One second of Vorbis is where PyAV's end trimming differs from libsndfile's.
    data = encode_audio(tone(channels), container_format, subtype, SAMPLE_RATE)

    decoded, sample_rate = decode_audio_bytes(data)

    assert sample_rate == SAMPLE_RATE
    np.testing.assert_array_equal(decoded, libsndfile_mono(data))


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("subtype", ["PCM_U8", "PCM_16", "PCM_32", "FLOAT", "DOUBLE"])
def test_wav_the_parser_reads_decodes_as_the_wav_parser(
    subtype: str, channels: int
) -> None:
    data = encode_audio(tone(channels), "WAV", subtype, SAMPLE_RATE)

    decoded, sample_rate = decode_audio_bytes(data)
    parsed, parsed_rate = parse_wav_bytes(data)

    assert sample_rate == parsed_rate == SAMPLE_RATE
    np.testing.assert_array_equal(decoded, parsed)


def test_containers_libsndfile_rejects_decode_through_pyav() -> None:
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="adts") as container:
        stream = container.add_stream("aac", rate=SAMPLE_RATE, layout="mono")
        frame = av.AudioFrame.from_ndarray(
            tone(1)[None, :], format="fltp", layout="mono"
        )
        frame.sample_rate = SAMPLE_RATE
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            container.mux(packet)
    data = buffer.getvalue()

    with pytest.raises(sf.LibsndfileError):
        sf.read(io.BytesIO(data))
    decoded, sample_rate = decode_audio_bytes(data)
    pyav_decoded, _ = decode_audio_bytes_av(data)

    assert sample_rate == SAMPLE_RATE
    np.testing.assert_array_equal(decoded, pyav_decoded)


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
def test_pyav_lossless_decode_equals_libsndfile(
    container_format: str, subtype: str, channels: int
) -> None:
    data = encode_audio(tone(channels), container_format, subtype, SAMPLE_RATE)

    decoded, sample_rate = decode_audio_bytes_av(data)

    assert sample_rate == SAMPLE_RATE
    np.testing.assert_array_equal(decoded, libsndfile_mono(data))


@pytest.mark.parametrize("channels", [1, 2])
def test_pyav_planar_decode_matches_libsndfile(channels: int) -> None:
    data = encode_audio(tone(channels, 48_000), "OGG", "OPUS", 48_000)

    decoded, sample_rate = decode_audio_bytes_av(data)

    reference = libsndfile_mono(data)
    assert sample_rate == 48_000
    assert decoded.shape == reference.shape
    np.testing.assert_allclose(decoded, reference, rtol=0, atol=LOSSY_TOLERANCE)


def test_pyav_float_audio_above_full_scale_keeps_its_amplitude() -> None:
    source = np.where(np.arange(8_000) % 2, -1.5, 1.5).astype(np.float32)
    data = encode_audio(source, "WAV", "FLOAT", 8_000)

    decoded, sample_rate = decode_audio_bytes_av(data)

    assert sample_rate == 8_000
    np.testing.assert_array_equal(decoded, source)
