# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from sglang_omni.client import GenerateChunk
from sglang_omni.client.types import CompletionResult, GenerateRequest
from sglang_omni.serve import create_app

WHISPER_MODEL = "openai/whisper-large-v3"
# Per-model capability truth lives in test_translation_capability.py; the
# route only sees a boolean, so one unsupported model exercises the path.
UNSUPPORTED_MODEL = "Qwen/Qwen3-ASR-1.7B"


class RecordingTranslationClient:
    def __init__(
        self,
        detected_language: str = "zh",
        timestamp_text: str = "<|0.00|>hello world<|1.50|>",
    ) -> None:
        self.requests: list[GenerateRequest] = []
        self.detected_language = detected_language
        self.timestamp_text = timestamp_text

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        del audio_format
        self.requests.append(request)
        if request.extra_params.get("detect_language"):
            return CompletionResult(request_id=request_id, text=self.detected_language)
        if request.extra_params.get("segment_timestamps"):
            return CompletionResult(
                request_id=request_id,
                text=self.timestamp_text,
            )
        return CompletionResult(request_id=request_id, text="hello world")

    async def generate(
        self,
        request: GenerateRequest,
        request_id: str | None = None,
    ):
        self.requests.append(request)
        yield GenerateChunk(request_id=request_id or "translation-1", text="hello ")
        yield GenerateChunk(request_id=request_id or "translation-1", text="world")
        yield GenerateChunk(
            request_id=request_id or "translation-1",
            text="hello world",
            finish_reason="stop",
        )


def _translation_client(
    *,
    model_name: str = WHISPER_MODEL,
    supports_audio_translation: bool = True,
) -> tuple[FastAPI, RecordingTranslationClient]:
    backend = RecordingTranslationClient()
    app = create_app(
        backend,
        model_name=model_name,
        architectures=["WhisperForConditionalGeneration"],
        supports_audio_translation=supports_audio_translation,
    )
    return app, backend


def _post_translation(
    app: FastAPI,
    *,
    model: str = WHISPER_MODEL,
    response_format: str = "json",
    stream: bool = False,
    language: str | None = "fr",
    audio: bytes = b"RIFF",
) -> httpx.Response:
    data: dict[str, str] = {
        "model": model,
        "response_format": response_format,
        "stream": str(stream).lower(),
    }
    if language is not None:
        data["language"] = language

    async def _post() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/audio/translations",
                data=data,
                files={"file": ("sample.wav", audio, "audio/wav")},
            )

    return asyncio.run(_post())


def test_whisper_translation_sets_translate_task_and_preserves_language() -> None:
    client, backend = _translation_client()

    response = _post_translation(client, language="fr")

    assert response.status_code == 200
    assert response.json() == {"text": "hello world"}
    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.model == WHISPER_MODEL
    assert request.extra_params["task"] == "translate"
    assert request.extra_params["language"] == "fr"


def test_unsupported_model_returns_openai_shaped_400_before_audio_read() -> None:
    model_name = UNSUPPORTED_MODEL
    client, backend = _translation_client(
        model_name=model_name,
        supports_audio_translation=False,
    )

    response = _post_translation(client, model=model_name, audio=b"")

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "model"
    assert model_name in error["message"]
    assert "/v1/audio/translations" in error["message"]
    assert "use /v1/audio/transcriptions" in error["message"]
    assert backend.requests == []


@pytest.mark.parametrize(
    "response_format,expected_status,expected_content_type",
    [
        ("json", 200, "application/json"),
        ("verbose_json", 200, "application/json"),
        ("text", 200, "text/plain"),
        ("srt", 200, "text/plain"),
        ("vtt", 200, "text/plain"),
    ],
)
def test_translation_response_format_matrix(
    response_format: str,
    expected_status: int,
    expected_content_type: str,
) -> None:
    client, backend = _translation_client()

    response = _post_translation(client, response_format=response_format)

    assert response.status_code == expected_status
    assert response.headers["content-type"].startswith(expected_content_type)
    if response_format == "json":
        assert response.json() == {"text": "hello world"}
    elif response_format == "verbose_json":
        body = response.json()
        assert body["task"] == "translate"
        assert body["text"] == "hello world"
        assert body["segments"] == [
            {"id": 0, "start": 0.0, "end": 0.0, "text": "hello world"}
        ]
    elif response_format == "text":
        assert response.text == "hello world"
        assert not response.text.startswith("{")
    elif response_format == "srt":
        assert response.text == "1\n00:00:00,000 --> 00:00:01,500\nhello world\n\n"
    else:
        assert response.text == "WEBVTT\n\n00:00.000 --> 00:01.500\nhello world\n\n"
    if response_format in {"srt", "vtt"}:
        assert backend.requests[-1].extra_params["task"] == "translate"
        assert backend.requests[-1].extra_params["segment_timestamps"] is True


def test_omitted_language_detects_the_source() -> None:
    client, backend = _translation_client()

    response = _post_translation(client, language=None)

    assert response.status_code == 200
    assert response.json()["text"] == "hello world"
    assert backend.requests[-1].extra_params["language"] == "zh"


def test_invalid_response_format_returns_openai_400() -> None:
    client, backend = _translation_client()

    response = _post_translation(client, response_format="xml")

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "response_format"
    assert "'xml'" in error["message"]
    assert backend.requests == []


def test_segment_format_without_adapter_timestamps_returns_400() -> None:
    backend = RecordingTranslationClient()
    app = create_app(
        backend,
        model_name=WHISPER_MODEL,
        architectures=None,
        supports_audio_translation=True,
    )

    response = _post_translation(app, response_format="SRT")

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "response_format"
    assert "segment-timestamp capability" in error["message"]
    assert backend.requests == []


def test_translation_timestamp_decode_error_uses_openai_envelope() -> None:
    """Keep post-decode subtitle failures in translation's OpenAI error schema."""
    backend = RecordingTranslationClient(timestamp_text="hello world")
    app = create_app(
        backend,
        model_name=WHISPER_MODEL,
        architectures=["WhisperForConditionalGeneration"],
        supports_audio_translation=True,
    )

    response = _post_translation(app, response_format="srt")

    assert response.status_code == 400
    assert "detail" not in response.json()
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] is None
    assert error["message"] == "model did not produce segment timestamps"


def test_whisper_translation_stream_matches_transcription_sse_lifecycle() -> None:
    client, backend = _translation_client()

    response = _post_translation(client, stream=True, language="de")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-request-id"].startswith("translation-")
    body = response.text
    assert (
        body.index('"transcript.text.delta"')
        < body.index('"transcript.text.done"')
        < body.index("data: [DONE]")
    )
    assert '"delta":"hello "' in body
    assert '"delta":"world"' in body
    assert '"text":"hello world"' in body
    assert backend.requests[0].extra_params["task"] == "translate"
    assert backend.requests[0].extra_params["language"] == "de"


@pytest.mark.parametrize("response_format", ["srt", "vtt"])
def test_translation_stream_rejects_formats_without_segment_timestamps(
    response_format: str,
) -> None:
    client, backend = _translation_client()

    response = _post_translation(
        client,
        response_format=response_format,
        stream=True,
    )

    assert response.status_code == 400
    assert "stream=true" in response.json()["error"]["message"]
    assert backend.requests == []


def test_unknown_model_rejected_with_404() -> None:
    client, backend = _translation_client()

    response = _post_translation(client, model="unknown/model")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "model_not_found"
    assert "unknown/model" in error["message"]
    assert backend.requests == []
