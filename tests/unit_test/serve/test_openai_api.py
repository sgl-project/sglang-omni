# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import ClientError, CompletionStreamChunk, GenerateChunk
from sglang_omni.client.types import CompletionResult, GenerateRequest
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import (
    _build_speech_generate_request,
    _chat_stream,
    _speech_stream,
    build_speech_generate_request,
)
from sglang_omni.serve.protocol import ChatCompletionRequest, CreateSpeechRequest

MODEL_FAMILIES = ("qwen3-omni", "ming-omni", "s2-pro", "voxtral")


class FaultInjectionClient:
    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def completion(self, *args, **kwargs) -> CompletionResult:
        del args, kwargs
        raise ClientError("cuda out of memory")

    async def completion_stream(self, *args, **kwargs):
        del args, kwargs
        yield CompletionStreamChunk(
            request_id="chat-oom",
            text="partial",
            modality="text",
        )
        raise ClientError("cuda out of memory")

    async def speech(self, *args, **kwargs):
        del args, kwargs
        raise ClientError("cuda out of memory")

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-oom",
            modality="audio",
            audio_data=[0.0, 0.1],
            sample_rate=24000,
        )
        raise ClientError("cuda out of memory")


class SuccessfulSpeechClient:
    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-1",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=24000,
            finish_reason="stop",
        )


class FailingSpeechClient:
    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request, request_id
        yield GenerateChunk(
            request_id="speech-1",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=24000,
        )
        raise ClientError("stream failed")


@pytest.mark.parametrize("model_name", MODEL_FAMILIES)
def test_non_streaming_http_faults_return_500(model_name: str) -> None:
    client = TestClient(create_app(FaultInjectionClient(), model_name=model_name))

    chat_resp = client.post(
        "/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert chat_resp.status_code == 500
    assert "cuda out of memory" in chat_resp.json()["detail"]

    speech_resp = client.post(
        "/v1/audio/speech",
        json={
            "model": model_name,
            "input": "hello",
            "stream": False,
            "response_format": "wav",
        },
    )
    assert speech_resp.status_code == 500
    assert "cuda out of memory" in speech_resp.json()["detail"]


def test_chat_stream_failure_closes_without_done_sentinel() -> None:
    chunks: list[str] = []
    req = ChatCompletionRequest(
        model="qwen3-omni",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

    async def _drive() -> None:
        async for chunk in _chat_stream(
            client=FaultInjectionClient(),
            gen_req=GenerateRequest(model="qwen3-omni", prompt="hello", stream=True),
            request_id="req-1",
            response_id="chatcmpl-req-1",
            created=0,
            model="qwen3-omni",
            req=req,
            audio_format="wav",
        ):
            chunks.append(chunk)

    with pytest.raises(ClientError, match="cuda out of memory"):
        asyncio.run(_drive())

    assert chunks
    assert all(chunk != "data: [DONE]\n\n" for chunk in chunks)


async def _collect_speech_stream(client: Any) -> list[str]:
    chunks: list[str] = []
    async for chunk in _speech_stream(
        client=client,
        gen_req=GenerateRequest(model="s2-pro", prompt="hello", stream=True),
        request_id="req-1",
        response_format="wav",
        speed=1.0,
    ):
        chunks.append(chunk)
    return chunks


def test_speech_stream_success_emits_done_sentinel() -> None:
    chunks = asyncio.run(_collect_speech_stream(SuccessfulSpeechClient()))

    assert chunks[-1] == "data: [DONE]\n\n"
    payload = json.loads(chunks[-2][len("data: ") :])
    assert payload["audio"] is None
    assert payload["finish_reason"] == "stop"


def test_speech_stream_failure_closes_without_done_sentinel() -> None:
    """A mid-stream failure must not be reported as a successful SSE finish."""

    chunks: list[str] = []

    async def _drive() -> None:
        async for chunk in _speech_stream(
            client=FailingSpeechClient(),
            gen_req=GenerateRequest(model="s2-pro", prompt="hello", stream=True),
            request_id="req-1",
            response_format="wav",
            speed=1.0,
        ):
            chunks.append(chunk)

    with pytest.raises(ClientError, match="stream failed"):
        asyncio.run(_drive())

    assert chunks
    assert all(chunk != "data: [DONE]\n\n" for chunk in chunks)
    payload = json.loads(chunks[0][len("data: ") :])
    assert payload["audio"] is not None
    assert payload["finish_reason"] is None


def test_speech_request_records_explicit_generation_params() -> None:
    req = CreateSpeechRequest(
        input="hello",
        temperature=0.8,
        top_k=30,
        seed=123,
    )

    gen_req = build_speech_generate_request(req, "qwen3-tts")

    assert _build_speech_generate_request is build_speech_generate_request
    assert gen_req.sampling.temperature == 0.8
    assert gen_req.sampling.top_k == 30
    assert gen_req.sampling.seed == 123
    assert gen_req.metadata["tts_params"]["explicit_generation_params"] == [
        "seed",
        "temperature",
        "top_k",
    ]
