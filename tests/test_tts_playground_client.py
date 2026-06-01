# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the S2-Pro TTS playground HTTP client."""

from __future__ import annotations

import base64
import json

import pytest

from playground.tts.api_client import SpeechDemoClient, SpeechDemoClientError
from playground.tts.models import GenerationSettings, SpeechSynthesisRequest
from sglang_omni.client.audio import encode_wav


def _request() -> SpeechSynthesisRequest:
    return SpeechSynthesisRequest(
        text="hello world",
        reference_audio_path=None,
        reference_text="",
        settings=GenerationSettings(
            temperature=0.8,
            top_p=0.8,
            top_k=30,
            max_new_tokens=256,
        ),
    )


def _stream_lines(include_terminal: bool = True) -> list[str]:
    audio = encode_wav([0.0, 0.1], 24000)
    payload = {
        "id": "speech-1",
        "object": "audio.speech.chunk",
        "index": 0,
        "audio": {
            "data": base64.b64encode(audio).decode("ascii"),
            "format": "wav",
            "mime_type": "audio/wav",
            "sample_rate": 24000,
        },
        "finish_reason": None,
    }
    lines = [f"data: {json.dumps(payload)}"]
    if include_terminal:
        terminal = {
            "id": "speech-1",
            "object": "audio.speech.chunk",
            "index": 1,
            "audio": None,
            "finish_reason": "stop",
        }
        lines.extend([f"data: {json.dumps(terminal)}", "data: [DONE]"])
    return lines


class _DummyStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        for line in self._lines:
            yield line

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False


def test_stream_synthesize_requires_terminal_markers(monkeypatch) -> None:
    monkeypatch.setattr(
        "playground.tts.api_client.httpx.stream",
        lambda *args, **kwargs: _DummyStreamResponse(
            _stream_lines(include_terminal=False)
        ),
    )

    client = SpeechDemoClient("http://localhost:8000")

    with pytest.raises(SpeechDemoClientError, match="terminal completion markers"):
        list(client.stream_synthesize(_request()))


def test_stream_synthesize_yields_audio_and_completes(monkeypatch) -> None:
    monkeypatch.setattr(
        "playground.tts.api_client.httpx.stream",
        lambda *args, **kwargs: _DummyStreamResponse(
            _stream_lines(include_terminal=True)
        ),
    )

    client = SpeechDemoClient("http://localhost:8000")
    events = list(client.stream_synthesize(_request()))

    assert len(events) == 2
    assert events[0].audio_bytes is not None
    assert events[1].finish_reason == "stop"
