# SPDX-License-Identifier: Apache-2.0
"""Tests for the minimal OpenAI-compatible adapter."""

from __future__ import annotations

import base64
import io
import json
import wave
from typing import Any

from fastapi.testclient import TestClient

from sglang_omni.client import (
    Client,
    CompletionResult,
    CompletionStreamChunk,
    GenerateChunk,
)
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import (
    _build_speech_generate_request,
    _select_speech_audio_delta,
)
from sglang_omni.serve.protocol import CreateSpeechRequest


class DummyClient:
    """Minimal stand-in for ``Client`` that replays pre-built chunks."""

    def __init__(self, chunks: list[GenerateChunk]):
        self._chunks = chunks

    async def generate(self, request: Any, request_id: str | None = None):
        for chunk in self._chunks:
            yield chunk

    async def completion(
        self, request: Any, *, request_id: str, audio_format: str = "wav"
    ) -> CompletionResult:
        text_parts: list[str] = []
        finish_reason: str | None = None
        async for chunk in self.generate(request, request_id=request_id):
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
        return CompletionResult(
            request_id=request_id,
            text="".join(text_parts),
            finish_reason=finish_reason or "stop",
        )

    async def completion_stream(
        self, request: Any, *, request_id: str, audio_format: str = "wav"
    ):
        async for chunk in self.generate(request, request_id=request_id):
            yield CompletionStreamChunk(
                request_id=request_id,
                text=chunk.text,
                modality=chunk.modality,
                finish_reason=chunk.finish_reason,
            )

    def health(self) -> dict[str, Any]:
        return {"running": True}


class MockSpeechCoordinator:
    """Minimal coordinator mock for speech E2E tests."""

    def __init__(self) -> None:
        self.last_request = None
        self.last_request_id = None

    async def submit(self, request_id: str, request: Any) -> dict[str, Any]:
        self.last_request_id = request_id
        self.last_request = request
        return {
            "audio_data": [0.0, 0.1, -0.1, 0.0],
            "sample_rate": 24000,
            "modality": "audio",
            "finish_reason": "stop",
        }

    def health(self) -> dict[str, Any]:
        return {"running": True}


def test_chat_completions_non_stream() -> None:
    dummy = DummyClient(
        [GenerateChunk(request_id="req-1", text="hello", finish_reason="stop")]
    )
    client = TestClient(create_app(dummy))

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hello"


def test_chat_completions_stream() -> None:
    dummy = DummyClient(
        [
            GenerateChunk(request_id="req-1", text="hi"),
            GenerateChunk(request_id="req-1", finish_reason="stop"),
        ]
    )
    client = TestClient(create_app(dummy))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        timeout=5.0,
    ) as resp:
        assert resp.status_code == 200
        events = []
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            events.append(json.loads(payload))

    deltas = [event["choices"][0]["delta"] for event in events]
    assert any("content" in delta for delta in deltas)


def test_build_speech_request_supports_references() -> None:
    req = CreateSpeechRequest(
        model="s2-pro",
        input="hello",
        references=[{"audio_path": "nan1.wav", "text": "male voice reference"}],
    )

    gen_req = _build_speech_generate_request(req, default_model="fallback-model")

    assert isinstance(gen_req.prompt, dict)
    assert gen_req.prompt["text"] == "hello"
    assert gen_req.prompt["references"] == [
        {"audio_path": "nan1.wav", "text": "male voice reference"}
    ]


def test_build_speech_request_keeps_ref_audio_compatibility() -> None:
    req = CreateSpeechRequest(
        model="s2-pro",
        input="hello",
        ref_audio="ref.wav",
        ref_text="reference transcript",
    )

    gen_req = _build_speech_generate_request(req, default_model="fallback-model")

    assert isinstance(gen_req.prompt, dict)
    assert gen_req.prompt["references"] == [
        {"audio_path": "ref.wav", "text": "reference transcript"}
    ]


def test_build_speech_request_preserves_stream_flag() -> None:
    req = CreateSpeechRequest(
        model="s2-pro",
        input="hello",
        stream=True,
    )

    gen_req = _build_speech_generate_request(req, default_model="fallback-model")
    assert gen_req.stream is True


def test_speech_endpoint_e2e_with_mock_pipeline_references() -> None:
    coordinator = MockSpeechCoordinator()
    client = Client(coordinator=coordinator)
    app_client = TestClient(create_app(client, model_name="s2-pro"))

    response = app_client.post(
        "/v1/audio/speech",
        json={
            "model": "s2-pro",
            "input": "hello",
            "references": [{"audio_path": "nan1.wav", "text": "male voice reference"}],
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content.startswith(b"RIFF")
    assert coordinator.last_request is not None
    assert coordinator.last_request.inputs == {
        "text": "hello",
        "references": [{"audio_path": "nan1.wav", "text": "male voice reference"}],
    }


def test_speech_endpoint_stream_returns_sse_audio_chunks() -> None:
    dummy = DummyClient(
        [
            GenerateChunk(
                request_id="speech-1",
                modality="audio",
                audio_data=[0.0, 0.1, -0.1, 0.0],
                sample_rate=24000,
            ),
            GenerateChunk(
                request_id="speech-1",
                modality="audio",
                audio_data=[0.0, 0.1, -0.1, 0.0, 0.2, -0.2],
                sample_rate=24000,
                finish_reason="stop",
            ),
        ]
    )
    client = TestClient(create_app(dummy, model_name="s2-pro"))

    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={
            "model": "s2-pro",
            "input": "hello",
            "stream": True,
            "response_format": "wav",
        },
        timeout=5.0,
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            events.append(json.loads(payload))

    audio_events = [event for event in events if event.get("audio")]
    assert len(audio_events) == 2
    sample_counts = [
        _decode_wav_frame_count(event["audio"]["data"]) for event in audio_events
    ]
    assert sample_counts == [4, 2]
    assert events[-1]["finish_reason"] == "stop"


def test_select_speech_audio_delta_suppresses_empty_chunks() -> None:
    audio, emitted = _select_speech_audio_delta(
        [],
        emitted_samples=5,
        is_terminal=False,
    )

    assert audio is None
    assert emitted == 5


def test_select_speech_audio_delta_returns_only_new_terminal_samples() -> None:
    audio, emitted = _select_speech_audio_delta(
        [0.0, 0.1, -0.1, 0.0, 0.2, -0.2],
        emitted_samples=4,
        is_terminal=True,
    )

    assert audio is not None
    assert list(audio) == [0.2, -0.2]
    assert emitted == 6


def _decode_wav_frame_count(audio_b64: str) -> int:
    wav_bytes = base64.b64decode(audio_b64)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        return wav_file.getnframes()


def test_speech_stream_finishes_without_terminal_audio_chunk() -> None:
    dummy = DummyClient(
        [
            GenerateChunk(
                request_id="speech-1",
                modality="audio",
                audio_data=[0.0, 0.1, -0.1, 0.0],
                sample_rate=24000,
            ),
            GenerateChunk(
                request_id="speech-1",
                finish_reason="stop",
            ),
        ]
    )
    client = TestClient(create_app(dummy, model_name="s2-pro"))

    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={
            "model": "s2-pro",
            "input": "hello",
            "stream": True,
            "response_format": "wav",
        },
        timeout=5.0,
    ) as resp:
        assert resp.status_code == 200
        events = []
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            events.append(json.loads(payload))

    assert len(events) == 2
    assert events[0]["audio"] is not None
    assert events[0]["finish_reason"] is None
    assert events[1]["audio"] is None
    assert events[1]["finish_reason"] == "stop"


def test_speech_stream_uses_actual_audio_format_on_fallback(monkeypatch) -> None:
    dummy = DummyClient(
        [
            GenerateChunk(
                request_id="speech-1",
                modality="audio",
                audio_data=[0.0, 0.1, -0.1, 0.0],
                sample_rate=24000,
                finish_reason="stop",
            ),
        ]
    )
    client = TestClient(create_app(dummy, model_name="s2-pro"))

    def _fake_encode_audio(*args, **kwargs):
        del args, kwargs
        return b"RIFF....", "audio/wav"

    monkeypatch.setattr("sglang_omni.serve.openai_api.encode_audio", _fake_encode_audio)

    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={
            "model": "s2-pro",
            "input": "hello",
            "stream": True,
            "response_format": "mp3",
        },
        timeout=5.0,
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("audio"):
                assert event["audio"]["mime_type"] == "audio/wav"
                assert event["audio"]["format"] == "wav"
                break
