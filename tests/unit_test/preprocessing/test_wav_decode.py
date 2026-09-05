# SPDX-License-Identifier: Apache-2.0
"""WAV byte parsing: multi-channel downmix stays numerically identical."""

from __future__ import annotations

import base64
import io
import struct
import wave

import numpy as np

from sglang_omni.preprocessing.audio import _downmix_to_mono, _parse_wav_bytes


def _wav_bytes(pcm: np.ndarray, channels: int, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


def test_stereo_downmix_matches_numpy_mean():
    rng = np.random.default_rng(0)
    frames = rng.standard_normal((480003, 2)).astype(np.float32)
    expected = frames.mean(axis=1)
    got = _downmix_to_mono(frames.reshape(-1), 2)
    assert got.dtype == np.float32
    assert np.array_equal(got, expected)


def test_multichannel_downmix_is_close_to_numpy_mean():
    rng = np.random.default_rng(1)
    frames = rng.standard_normal((1000, 6)).astype(np.float32)
    got = _downmix_to_mono(frames.reshape(-1), 6)
    np.testing.assert_allclose(got, frames.mean(axis=1), rtol=0, atol=1e-6)


def test_parse_stereo_pcm16_wav():
    rng = np.random.default_rng(2)
    pcm = rng.integers(-32768, 32767, size=(1200, 2), dtype=np.int16)
    audio, rate = _parse_wav_bytes(_wav_bytes(pcm, channels=2))
    assert rate == 16000
    assert audio.shape == (1200,)
    expected = (pcm.astype(np.float32) / 32768.0).mean(axis=1)
    assert np.array_equal(audio, expected.astype(np.float32))


def test_parse_mono_pcm16_wav_unchanged():
    pcm = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    audio, _ = _parse_wav_bytes(_wav_bytes(pcm, channels=1))
    assert np.array_equal(audio, pcm.astype(np.float32) / 32768.0)


def test_base64_loader_decodes_standard_payload():
    from sglang_omni.preprocessing.audio import AudioMediaIO

    pcm = np.zeros((16000, 2), dtype=np.int16)
    data = base64.b64encode(_wav_bytes(pcm, channels=2)).decode()
    audio, sr = AudioMediaIO(target_sr=16000).load_base64("audio/wav", data)
    assert sr == 16000.0
    assert audio.shape == (16000,)
    assert struct.calcsize("<h") == 2  # sanity: 16-bit samples as written
