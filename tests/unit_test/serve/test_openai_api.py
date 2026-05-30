# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import Client, GenerateChunk
from sglang_omni.client.types import GenerateRequest
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import (
    _build_speech_generate_request,
    _chat_stream,
    _speech_stream,
    build_speech_generate_request,
    build_transcription_generate_request,
)
from sglang_omni.serve.protocol import ChatCompletionRequest, CreateSpeechRequest
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane

MODEL_FAMILIES = {
    "qwen3-omni": "code2wav",
    "ming-omni": "talker",
    "s2-pro": "vocoder",
    "voxtral": "vocoder",
}


class FaultInjectingCoordinator(Coordinator):
    """Inject a model-stage failure through the real Coordinator/Client path."""

    def __init__(self, terminal_stage: str):
        super().__init__(
            completion_endpoint="inproc://complete",
            abort_endpoint="inproc://abort",
            entry_stage="preprocess",
            terminal_stages=[terminal_stage],
        )
        self.control_plane = RecordingCoordinatorControlPlane()
        self.terminal_stage = terminal_stage
        self.register_stage("preprocess", "inproc://preprocess")

    async def _submit_request(
        self, request_id: str, request: OmniRequest | Any
    ) -> None:
        await super()._submit_request(request_id, request)
        if not isinstance(request, OmniRequest):
            request = OmniRequest(inputs=request)
        if bool(request.params.get("stream", False)):
            await self._handle_stream(self._partial_stream_message(request_id, request))
        await self._handle_completion(
            CompleteMessage(
                request_id=request_id,
                from_stage=self.terminal_stage,
                success=False,
                error="cuda out of memory",
            )
        )

    def _partial_stream_message(
        self, request_id: str, request: OmniRequest
    ) -> StreamMessage:
        if "tts_params" in request.metadata:
            chunk = {
                "audio_data": [0.0, 0.1],
                "sample_rate": 24000,
                "modality": "audio",
            }
            modality = "audio"
        else:
            chunk = {"text": "partial", "modality": "text"}
            modality = "text"
        return StreamMessage(
            request_id=request_id,
            from_stage=self.terminal_stage,
            chunk=chunk,
            stage_name=self.terminal_stage,
            modality=modality,
        )


def _fault_client(model_name: str) -> Client:
    return Client(FaultInjectingCoordinator(MODEL_FAMILIES[model_name]))


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


class SuccessfulTranscriptionClient:
    def __init__(self) -> None:
        self.requests: list[GenerateRequest] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ):
        from sglang_omni.client.types import CompletionResult

        del request_id, audio_format
        self.requests.append(request)
        return CompletionResult(request_id="transcription-1", text="hello world")


@pytest.mark.parametrize("model_name", MODEL_FAMILIES)
def test_non_streaming_http_faults_return_500(model_name: str) -> None:
    client = TestClient(create_app(_fault_client(model_name), model_name=model_name))

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
    client = _fault_client("qwen3-omni")
    req = ChatCompletionRequest(
        model="qwen3-omni",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

    async def _drive() -> None:
        async for chunk in _chat_stream(
            client=client,
            gen_req=GenerateRequest(model="qwen3-omni", prompt="hello", stream=True),
            request_id="req-1",
            response_id="chatcmpl-req-1",
            created=0,
            model="qwen3-omni",
            req=req,
            audio_format="wav",
        ):
            chunks.append(chunk)

    with pytest.raises(RuntimeError, match="cuda out of memory"):
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
    client = _fault_client("s2-pro")

    async def _drive() -> None:
        async for chunk in _speech_stream(
            client=client,
            gen_req=GenerateRequest(
                model="s2-pro",
                prompt="hello",
                stream=True,
                metadata={"tts_params": {}},
            ),
            request_id="req-1",
            response_format="wav",
            speed=1.0,
        ):
            chunks.append(chunk)

    with pytest.raises(RuntimeError, match="cuda out of memory"):
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


def test_transcription_request_builds_asr_generate_request() -> None:
    gen_req = build_transcription_generate_request(
        audio_bytes=b"RIFF",
        filename="sample.wav",
        content_type="audio/wav",
        model="openai/whisper-large-v3",
        language="en",
        prompt=None,
        temperature=None,
    )

    assert gen_req.model == "openai/whisper-large-v3"
    assert gen_req.prompt == {
        "audio_bytes": b"RIFF",
        "filename": "sample.wav",
        "content_type": "audio/wav",
    }
    assert gen_req.extra_params == {"task": "transcribe", "language": "en"}
    assert gen_req.metadata == {"task": "asr"}
    assert gen_req.output_modalities == ["text"]
    assert gen_req.stream is False


def test_transcription_endpoint_returns_text_json() -> None:
    transcription_client = SuccessfulTranscriptionClient()
    client = TestClient(
        create_app(transcription_client, model_name="openai/whisper-large-v3")
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "openai/whisper-large-v3", "language": "en"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "hello world"}
    assert transcription_client.requests
    request = transcription_client.requests[0]
    assert request.model == "openai/whisper-large-v3"
    assert request.prompt["filename"] == "sample.wav"
    assert request.extra_params["language"] == "en"
