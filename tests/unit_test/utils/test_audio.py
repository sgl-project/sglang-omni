# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import wave

import pybase64
import pytest

from sglang_omni.utils import audio
from sglang_omni.utils.audio import load_audio


def _wav_bytes(
    num_samples: int = 1600, sample_rate: int = 16000, num_channels: int = 1
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(num_channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * num_samples * num_channels)
    return buffer.getvalue()


def test_load_audio_accepts_base64_data_uri() -> None:
    encoded = pybase64.b64encode(_wav_bytes()).decode("ascii")

    samples = load_audio(f"data:audio/wav;base64,{encoded}")

    assert samples.shape == (1600,)


def test_load_audio_rejects_non_base64_data_uri() -> None:
    with pytest.raises(ValueError, match="Invalid base64 audio data URI"):
        load_audio("data:audio/wav,not-base64")


def test_load_audio_can_preserve_channels() -> None:
    encoded = pybase64.b64encode(_wav_bytes(num_channels=2)).decode("ascii")

    samples = load_audio(f"data:audio/wav;base64,{encoded}", mono=False)

    assert samples.shape == (2, 1600)


def test_load_audio_accepts_file_uri(tmp_path) -> None:
    path = tmp_path / "audio.wav"
    path.write_bytes(_wav_bytes())

    samples = load_audio(path.as_uri())

    assert samples.shape == (1600,)


class _FakeHTTPResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.raise_checked = False

    def raise_for_status(self) -> None:
        self.raise_checked = True


def test_load_audio_accepts_http_url(monkeypatch) -> None:
    response = _FakeHTTPResponse(_wav_bytes())
    calls = []

    def fake_get(url: str, *, timeout: int, follow_redirects: bool):
        calls.append(
            {
                "url": url,
                "timeout": timeout,
                "follow_redirects": follow_redirects,
            }
        )
        return response

    monkeypatch.setenv("REQUEST_TIMEOUT", "7")
    monkeypatch.setattr(audio.httpx, "get", fake_get)

    samples = load_audio("https://example.test/audio.wav")

    assert samples.shape == (1600,)
    assert calls == [
        {
            "url": "https://example.test/audio.wav",
            "timeout": 7,
            "follow_redirects": True,
        }
    ]
    assert response.raise_checked


def test_load_audio_uses_default_timeout_for_invalid_env(monkeypatch) -> None:
    response = _FakeHTTPResponse(_wav_bytes())
    calls = []

    def fake_get(url: str, *, timeout: int, follow_redirects: bool):
        calls.append(timeout)
        return response

    monkeypatch.setenv("REQUEST_TIMEOUT", "abc")
    monkeypatch.setattr(audio.httpx, "get", fake_get)

    samples = load_audio("https://example.test/audio.wav")

    assert samples.shape == (1600,)
    assert calls == [audio._DEFAULT_REQUEST_TIMEOUT]
