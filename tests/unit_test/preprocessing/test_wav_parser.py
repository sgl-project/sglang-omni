# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the dependency-free WAV fast-path parser."""

import struct

import numpy as np
import pytest

from sglang_omni.preprocessing.audio import _parse_wav_bytes

PCM = 1
IEEE_FLOAT = 3
EXTENSIBLE = 0xFFFE

# Fixed 14-byte GUID suffix shared by every KSDATAFORMAT_SUBTYPE_* value.
_GUID_SUFFIX = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"


def _build_wav(fmt_tag, bits, data_bytes, *, channels=1, sample_rate=16000):
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    if fmt_tag == EXTENSIBLE:
        raise ValueError("use sub_format to build EXTENSIBLE fixtures")
    fmt = struct.pack(
        "<HHIIHH", fmt_tag, channels, sample_rate, byte_rate, block_align, bits
    )
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(data_bytes)) + data_bytes
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def _build_extensible_wav(
    sub_format, bits, data_bytes, *, channels=1, sample_rate=16000
):
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    fmt = struct.pack(
        "<HHIIHH", EXTENSIBLE, channels, sample_rate, byte_rate, block_align, bits
    )
    fmt += struct.pack("<HHI", 22, bits, 0)  # cbSize, wValidBitsPerSample, channelMask
    fmt += struct.pack("<H", sub_format) + _GUID_SUFFIX
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(data_bytes)) + data_bytes
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def _pack_s24(values):
    out = bytearray()
    for v in values:
        out += struct.pack("<i", v)[:3]
    return bytes(out)


def test_pcm24_mono_scales_to_unit_range():
    audio, sr = _parse_wav_bytes(
        _build_wav(PCM, 24, _pack_s24([0, 0x400000, -0x400000]))
    )
    assert sr == 16000
    np.testing.assert_allclose(audio, [0.0, 0.5, -0.5], atol=1e-6)
    assert audio.dtype == np.float32


def test_pcm24_multichannel_downmix():
    # Two frames, stereo: [(0, +1.0), (-1.0, 0)] -> mono means [0.5, -0.5].
    samples = _pack_s24([0, 0x7FFFFF, -0x800000, 0])
    audio, _ = _parse_wav_bytes(_build_wav(PCM, 24, samples, channels=2))
    np.testing.assert_allclose(audio, [0.5, -0.5], atol=1e-6)


def test_extensible_wraps_pcm16():
    data = np.array([0, 16384, -16384], dtype="<i2").tobytes()
    audio, sr = _parse_wav_bytes(_build_extensible_wav(PCM, 16, data))
    assert sr == 16000
    np.testing.assert_allclose(audio, [0.0, 0.5, -0.5], atol=1e-6)


def test_extensible_wraps_ieee_float32():
    data = np.array([0.0, 0.25, -0.75], dtype="<f4").tobytes()
    audio, _ = _parse_wav_bytes(_build_extensible_wav(IEEE_FLOAT, 32, data))
    np.testing.assert_allclose(audio, [0.0, 0.25, -0.75], atol=1e-6)


def test_extensible_wraps_pcm24():
    audio, _ = _parse_wav_bytes(
        _build_extensible_wav(PCM, 24, _pack_s24([0, 0x400000]))
    )
    np.testing.assert_allclose(audio, [0.0, 0.5], atol=1e-6)


def test_pcm16_still_parses():
    data = np.array([0, 16384, -16384], dtype="<i2").tobytes()
    audio, _ = _parse_wav_bytes(_build_wav(PCM, 16, data))
    np.testing.assert_allclose(audio, [0.0, 0.5, -0.5], atol=1e-6)


def test_unsupported_bit_depth_still_raises():
    with pytest.raises(ValueError):
        _parse_wav_bytes(_build_wav(PCM, 12, b"\x00\x00\x00"))


def _build_extensible_wav_raw_guid(guid16, bits, data_bytes, *, sample_rate=16000):
    assert len(guid16) == 16
    block_align = bits // 8
    byte_rate = sample_rate * block_align
    fmt = struct.pack(
        "<HHIIHH", EXTENSIBLE, 1, sample_rate, byte_rate, block_align, bits
    )
    fmt += struct.pack("<HHI", 22, bits, 0) + guid16
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(data_bytes)) + data_bytes
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def test_extensible_custom_guid_is_not_decoded_as_pcm():
    # A vendor GUID whose first two bytes look like PCM (0x0001) but whose tail
    # differs must NOT be unwrapped; it should raise so PyAV can handle it.
    bogus = b"\x01\x00" + b"\xde\xad\xbe\xef" + b"\x00" * 10
    data = np.array([0, 16384, -16384], dtype="<i2").tobytes()
    with pytest.raises(ValueError):
        _parse_wav_bytes(_build_extensible_wav_raw_guid(bogus, 16, data))
