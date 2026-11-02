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


def test_ogg_vorbis_decodes_as_the_reference_loader() -> None:
    data = encode_audio(tone(2), "OGG", "VORBIS", SAMPLE_RATE)

    decoded, sample_rate = decode_audio_bytes(data)

    assert sample_rate == SAMPLE_RATE
    np.testing.assert_array_equal(decoded, libsndfile_mono(data))


@pytest.mark.parametrize("subtype", ["PCM_U8", "PCM_16", "PCM_32", "FLOAT", "DOUBLE"])
def test_every_wav_the_parser_read_decodes_unchanged(subtype: str) -> None:
    data = encode_audio(tone(2), "WAV", subtype, SAMPLE_RATE)

    decoded, sample_rate = decode_audio_bytes(data)
    parsed, parsed_rate = parse_wav_bytes(data)

    assert sample_rate == parsed_rate == SAMPLE_RATE
    np.testing.assert_array_equal(decoded, parsed)


def test_wav_whose_header_states_no_data_decodes_to_its_end() -> None:
    data = encode_audio(tone(1), "WAV", "PCM_16", SAMPLE_RATE)
    data_at = data.index(b"data")
    unsized = data[: data_at + 4] + bytes(4) + data[data_at + 8 :]

    decoded, sample_rate = decode_audio_bytes(unsized)

    assert sample_rate == SAMPLE_RATE
    np.testing.assert_array_equal(decoded, libsndfile_mono(data))


def test_flac_whose_header_states_no_length_decodes_to_its_end() -> None:
    data = encode_audio(tone(2), "FLAC", "PCM_16", SAMPLE_RATE)
    streaminfo = int.from_bytes(data[18:26], "big")
    unsized = data[:18] + (streaminfo & ~((1 << 36) - 1)).to_bytes(8, "big") + data[26:]

    decoded, sample_rate = decode_audio_bytes(unsized)

    assert sample_rate == SAMPLE_RATE
    np.testing.assert_array_equal(decoded, libsndfile_mono(data))


def test_wav_without_samples_is_an_error() -> None:
    data = encode_audio(np.zeros(0, dtype=np.float32), "WAV", "PCM_16", SAMPLE_RATE)

    with pytest.raises(ValueError, match="No audio frames decoded"):
        decode_audio_bytes(data)


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


@pytest.mark.parametrize("subtype", ["PCM_U8", "PCM_16", "FLOAT"])
def test_pyav_packed_decode_equals_libsndfile(subtype: str) -> None:
    data = encode_audio(tone(2), "WAV", subtype, SAMPLE_RATE)

    decoded, sample_rate = decode_audio_bytes_av(data)

    assert sample_rate == SAMPLE_RATE
    np.testing.assert_array_equal(decoded, libsndfile_mono(data))


def test_pyav_planar_decode_matches_libsndfile() -> None:
    data = encode_audio(tone(2, 48_000), "OGG", "OPUS", 48_000)

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
