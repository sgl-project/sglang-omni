# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import Client, GenerateChunk
from sglang_omni.client.audio import encode_pcm
from sglang_omni.client.types import GenerateRequest
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import (
    _await_speech_response,
    _chat_stream,
    _speech_audio_response,
    build_transcription_generate_request,
)
from sglang_omni.serve.protocol import ChatCompletionRequest, CreateSpeechRequest
from sglang_omni.serve.speech_service import SpeechRequestValidator
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
    def __init__(self, *, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-1",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=self.sample_rate,
            finish_reason="stop",
        )

    async def speech(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ):
        from sglang_omni.client.types import SpeechResult

        del request, request_id, speed, allow_format_fallback
        return SpeechResult(
            audio_bytes=b"RIFF",
            mime_type=f"audio/{response_format}",
            format=response_format,
        )


class EmptyStreamingSpeechClient:
    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-1",
            modality="audio",
            audio_data=None,
            sample_rate=24000,
            finish_reason="stop",
        )


class EmptyDeltaStreamingSpeechClient:
    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-1",
            modality="audio",
            audio_data=[],
            sample_rate=24000,
            finish_reason=None,
        )
        yield GenerateChunk(
            request_id=request_id or "speech-1",
            modality="audio",
            audio_data=None,
            sample_rate=24000,
            finish_reason="stop",
        )


class BlockingStreamingSpeechClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.aborted: list[str] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        self.started.set()
        await asyncio.Future()
        yield GenerateChunk(request_id=request_id or "speech-1")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class PrefetchedBlockingStreamingSpeechClient:
    def __init__(self) -> None:
        self.aborted: list[str] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-1",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=24000,
            finish_reason=None,
        )
        await asyncio.Future()

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class BlockingNonStreamingSpeechClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.aborted: list[str] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def speech(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
        allow_format_fallback: bool = True,
    ):
        del request, request_id, response_format, speed, allow_format_fallback
        self.started.set()
        await asyncio.Future()

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class DisconnectingRequest:
    def __init__(self) -> None:
        self.disconnected = asyncio.Event()

    async def is_disconnected(self) -> bool:
        return self.disconnected.is_set()


class ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


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
    assert speech_resp.json()["error"]["type"] == "server_error"
    assert "cuda out of memory" in speech_resp.json()["error"]["message"]


def test_speech_endpoint_rejects_invalid_request_with_openai_error() -> None:
    client = TestClient(create_app(SuccessfulSpeechClient(), model_name="tts"))

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "hello",
            "stream": True,
            "response_format": "wav",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "stream=true requires response_format='pcm'",
            "type": "BadRequestError",
            "param": "response_format",
            "code": 400,
        }
    }


def test_speech_endpoint_returns_binary_audio() -> None:
    client = TestClient(create_app(SuccessfulSpeechClient(), model_name="tts"))

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "response_format": "wav"},
    )

    assert response.status_code == 200
    assert response.content == b"RIFF"
    assert response.headers["content-type"] == "audio/wav"


def test_speech_endpoint_rejects_invalid_json_with_openai_error() -> None:
    client = TestClient(create_app(SuccessfulSpeechClient(), model_name="tts"))

    response = client.post(
        "/v1/audio/speech",
        content=b"{",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "BadRequestError"
    assert response.json()["error"]["code"] == 400


def test_speech_endpoint_stream_without_audio_returns_error() -> None:
    client = TestClient(create_app(EmptyStreamingSpeechClient(), model_name="tts"))

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "stream": True, "response_format": "pcm"},
    )

    assert response.status_code == 500
    assert response.json()["error"]["type"] == "server_error"
    assert "No audio output generated" in response.json()["error"]["message"]


def test_speech_endpoint_stream_empty_delta_is_not_success() -> None:
    client = TestClient(create_app(EmptyDeltaStreamingSpeechClient(), model_name="tts"))

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "stream": True, "response_format": "pcm"},
    )

    assert response.status_code == 500
    assert response.json()["error"]["type"] == "server_error"
    assert "No audio output generated" in response.json()["error"]["message"]


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


def test_speech_stream_defaults_to_raw_pcm() -> None:
    client = TestClient(
        create_app(SuccessfulSpeechClient(), model_name="higgs-audio-v2")
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "hello",
            "stream": True,
            "response_format": "pcm",
        },
    )

    expected = encode_pcm([0.0, 0.1, -0.1, 0.0], sample_rate=24000)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/pcm")
    assert response.headers["x-sample-rate"] == "24000"
    assert response.headers["x-channels"] == "1"
    assert response.headers["x-bit-depth"] == "16"
    assert response.content == expected


def test_speech_stream_returns_raw_pcm_bytes() -> None:
    client = TestClient(
        create_app(SuccessfulSpeechClient(), model_name="higgs-audio-v2")
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "hello",
            "stream": True,
            "response_format": "pcm",
        },
    )

    expected = encode_pcm([0.0, 0.1, -0.1, 0.0], sample_rate=24000)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/pcm")
    assert response.headers["x-sample-rate"] == "24000"
    assert response.headers["x-channels"] == "1"
    assert response.headers["x-bit-depth"] == "16"
    assert response.content == expected


def test_speech_stream_headers_use_chunk_sample_rate() -> None:
    client = TestClient(
        create_app(SuccessfulSpeechClient(sample_rate=44100), model_name="s2-pro")
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "hello",
            "stream": True,
            "response_format": "pcm",
        },
    )

    expected = encode_pcm([0.0, 0.1, -0.1, 0.0], sample_rate=44100)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/pcm")
    assert response.headers["x-sample-rate"] == "44100"
    assert response.headers["x-channels"] == "1"
    assert response.headers["x-bit-depth"] == "16"
    assert response.content == expected


def test_raw_pcm_response_close_aborts_inner_speech_stream() -> None:
    async def _drive() -> None:
        client = PrefetchedBlockingStreamingSpeechClient()
        response = await _speech_audio_response(
            client=client,
            gen_req=GenerateRequest(model="s2-pro", prompt="hello", stream=True),
            request_id="req-1",
            speed=1.0,
        )
        body = response.body_iterator
        assert await anext(body) == encode_pcm([0.0, 0.1, -0.1, 0.0], 24000)
        await body.aclose()
        assert client.aborted == ["req-1"]

    asyncio.run(_drive())


def test_speech_stream_rejects_non_pcm_response_format() -> None:
    client = TestClient(
        create_app(SuccessfulSpeechClient(), model_name="higgs-audio-v2")
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "hello",
            "stream": True,
            "response_format": "wav",
        },
    )

    assert 400 <= response.status_code < 500
    assert "response_format" in response.text
    assert "pcm" in response.text.lower()


def test_speech_request_carries_initial_codec_chunk_frames() -> None:
    req = CreateSpeechRequest(
        input="hello",
        stream=True,
        response_format="pcm",
        initial_codec_chunk_frames=4,
    )

    gen_req = SpeechRequestValidator(
        default_model="higgs-audio-v2"
    ).build_generate_request(req)

    assert gen_req.extra_params["initial_codec_chunk_frames"] == 4


def test_raw_pcm_speech_request_defaults_initial_codec_chunk_frames() -> None:
    req = CreateSpeechRequest(
        input="hello",
        stream=True,
        response_format="pcm",
    )

    gen_req = SpeechRequestValidator(
        default_model="higgs-audio-v2"
    ).build_generate_request(req)

    assert gen_req.extra_params["initial_codec_chunk_frames"] == 1


def test_raw_pcm_speech_request_respects_explicit_initial_zero() -> None:
    req = CreateSpeechRequest(
        input="hello",
        stream=True,
        response_format="pcm",
        initial_codec_chunk_frames=0,
    )

    gen_req = SpeechRequestValidator(
        default_model="higgs-audio-v2"
    ).build_generate_request(req)

    assert gen_req.extra_params["initial_codec_chunk_frames"] == 0


def test_speech_response_disconnect_aborts_active_request() -> None:
    async def _drive() -> None:
        client = BlockingNonStreamingSpeechClient()
        request = DisconnectingRequest()
        task = asyncio.create_task(
            _await_speech_response(
                request=request,
                client=client,
                gen_req=GenerateRequest(model="s2-pro", prompt="hello"),
                request_id="req-1",
                response_format="wav",
                speed=1.0,
            )
        )
        await client.started.wait()
        request.disconnected.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.aborted == ["req-1"]

    asyncio.run(_drive())


def test_speech_response_returns_when_disconnect_poll_is_false() -> None:
    async def _drive() -> None:
        result = await _await_speech_response(
            request=ConnectedRequest(),
            client=SuccessfulSpeechClient(),
            gen_req=GenerateRequest(model="s2-pro", prompt="hello"),
            request_id="req-1",
            response_format="wav",
            speed=1.0,
        )
        assert result.audio_bytes == b"RIFF"

    asyncio.run(_drive())


def test_speech_request_records_explicit_generation_params() -> None:
    req = CreateSpeechRequest(
        input="hello",
        temperature=0.8,
        top_k=30,
        seed=123,
    )

    gen_req = SpeechRequestValidator(default_model="qwen3-tts").build_generate_request(
        req
    )

    assert gen_req.sampling.temperature == 0.8
    assert gen_req.sampling.top_k == 30
    assert gen_req.sampling.seed == 123
    assert gen_req.metadata["tts_params"]["explicit_generation_params"] == [
        "seed",
        "temperature",
        "top_k",
    ]


def test_speech_request_passes_streaming_control_fields() -> None:
    req = CreateSpeechRequest(
        input="hello",
        initial_codec_chunk_frames=8,
        x_vector_only_mode=True,
        response_format="pcm",
        stream=True,
    )

    gen_req = SpeechRequestValidator(default_model="qwen3-tts").build_generate_request(
        req
    )
    tts_params = gen_req.metadata["tts_params"]

    assert tts_params["initial_codec_chunk_frames"] == 8
    assert tts_params["x_vector_only_mode"] is True
    assert tts_params["response_format"] == "pcm"
    assert gen_req.extra_params == {"initial_codec_chunk_frames": 8}


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


def test_speech_request_passes_moss_token_count() -> None:
    req = CreateSpeechRequest(input="hello", token_count=180)

    gen_req = SpeechRequestValidator(default_model="moss-tts").build_generate_request(
        req
    )

    assert gen_req.metadata["tts_params"]["token_count"] == 180
