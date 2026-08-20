# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sglang_omni.admission import QueueFullError
from sglang_omni.client import Client, GenerateChunk
from sglang_omni.client.audio import encode_pcm
from sglang_omni.client.types import GenerateRequest
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import (
    EXPLICIT_GENERATION_PARAMS_KEY,
    CompleteMessage,
    OmniRequest,
    StreamMessage,
)
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import (
    _await_speech_response,
    _build_chat_generate_request,
    _chat_stream,
    _ClosableStreamingResponse,
    _speech_audio_response,
)
from sglang_omni.serve.protocol import ChatCompletionRequest, CreateSpeechRequest
from sglang_omni.serve.speech_service import SpeechRequestValidator
from sglang_omni.serve.transcriptions import (
    _first_transcription_chunk,
    _transcription_stream,
    build_transcription_generate_request,
)
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane

MODEL_FAMILIES = {
    "qwen3-omni": "code2wav",
    "ming-omni": "talker",
    "s2-pro": "vocoder",
    "voxtral": "vocoder",
}


class FaultInjectingCoordinator(Coordinator):
    """Inject a model-stage failure through the real Coordinator/Client path."""

    def __init__(self, terminal_stage: str, error: str = "cuda out of memory"):
        super().__init__(
            completion_endpoint="inproc://complete",
            abort_endpoint="inproc://abort",
            entry_stage="preprocess",
            terminal_stages=[terminal_stage],
        )
        self.control_plane = RecordingCoordinatorControlPlane()
        self.terminal_stage = terminal_stage
        self.error = error
        self.register_stage("preprocess", "inproc://preprocess")

    async def _submit_request(
        self,
        request_id: str,
        request: OmniRequest | Any,
        *,
        stream_queue: asyncio.Queue[CompleteMessage | StreamMessage] | None = None,
    ) -> None:
        await super()._submit_request(
            request_id,
            request,
            stream_queue=stream_queue,
        )
        if not isinstance(request, OmniRequest):
            request = OmniRequest(inputs=request)
        if bool(request.params.get("stream", False)):
            await self._handle_stream(self._partial_stream_message(request_id, request))
        await self._handle_completion(
            CompleteMessage(
                request_id=request_id,
                from_stage=self.terminal_stage,
                success=False,
                error=self.error,
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


def _fault_client(model_name: str, error: str = "cuda out of memory") -> Client:
    return Client(FaultInjectingCoordinator(MODEL_FAMILIES[model_name], error=error))


class SuccessfulSpeechClient:
    def __init__(
        self, *, sample_rate: int = 24000, finish_reason: str = "stop"
    ) -> None:
        self.sample_rate = sample_rate
        self.finish_reason = finish_reason
        self.generate_requests: list[GenerateRequest] = []
        self.speech_requests: list[GenerateRequest] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        self.generate_requests.append(request)
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

        del request_id, speed, allow_format_fallback
        self.speech_requests.append(request)
        return SpeechResult(
            audio_bytes=b"RIFF",
            mime_type=f"audio/{response_format}",
            format=response_format,
            finish_reason=self.finish_reason,
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


class FailingSpeechGenerateClient:
    def __init__(self, error: str) -> None:
        self.error = error

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request, request_id
        raise RuntimeError(self.error)
        yield

    async def abort(self, request_id: str) -> None:
        del request_id


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


class BlockingFirstAudioStreamingSpeechClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.aborted: list[str] = []

    async def generate(self, request: Any, request_id: str | None = None):
        del request, request_id
        self.started.set()
        await asyncio.Future()
        yield GenerateChunk(request_id="speech-1")

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

    async def generate(
        self,
        request: GenerateRequest,
        request_id: str | None = None,
    ):
        del request_id
        self.requests.append(request)
        yield GenerateChunk(request_id="transcription-1", text="hello ")
        yield GenerateChunk(request_id="transcription-1", text="world")
        yield GenerateChunk(
            request_id="transcription-1",
            text="hello world",
            finish_reason="stop",
        )


class ChunkRecordingTranscriptionClient:
    """Records (request_id, request) pairs; can fail one chunk by index.

    Texts derive from the chunk index parsed out of the request id, so the
    "joined in span order" assertions hold however chunks are scheduled.
    """

    def __init__(
        self,
        *,
        fail_chunk: int | None = None,
        fail_message: str = "cuda out of memory",
    ) -> None:
        self.requests: list[tuple[str, GenerateRequest]] = []
        self.aborted: list[str] = []
        self.fail_chunk = fail_chunk
        self.fail_message = fail_message

    def health(self) -> dict[str, Any]:
        return {"running": True}

    @staticmethod
    def _chunk_index(request_id: str) -> int | None:
        if "-chunk-" not in request_id:
            return None
        return int(request_id.rsplit("-chunk-", 1)[-1])

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ):
        from sglang_omni.client import ClientError
        from sglang_omni.client.types import CompletionResult

        del audio_format
        arrival = len(self.requests)
        self.requests.append((request_id, request))
        index = self._chunk_index(request_id)
        if index is None:
            index = arrival
        if self.fail_chunk is not None and index == self.fail_chunk:
            raise ClientError(self.fail_message)
        return CompletionResult(request_id=request_id, text=f"part{index}")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class FailingTranscriptionClient:
    def __init__(self, message: str, exc_type: type[Exception] | None = None) -> None:
        self.message = message
        self.exc_type = exc_type

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ):
        from sglang_omni.client import ClientError

        del request, request_id, audio_format
        raise (self.exc_type or ClientError)(self.message)

    async def generate(
        self,
        request: GenerateRequest,
        request_id: str | None = None,
    ):
        from sglang_omni.client import ClientError

        del request, request_id
        raise (self.exc_type or ClientError)(self.message)
        yield  # unreachable, makes this an async generator


class _IdentityTranscriptionAdapter:
    def postprocess_text(self, text: str) -> str:
        return text


class _BlockingAbortControlPlane(RecordingCoordinatorControlPlane):
    def __init__(self) -> None:
        super().__init__()
        self.abort_started = asyncio.Event()
        self.release_abort = asyncio.Event()
        self.abort_cancelled = False

    async def broadcast_abort(self, msg: Any) -> None:
        self.aborts.append(msg)
        self.abort_started.set()
        try:
            await self.release_abort.wait()
        except asyncio.CancelledError:
            self.abort_cancelled = True
            raise


def _streaming_client(
    control_plane: RecordingCoordinatorControlPlane | None = None,
) -> tuple[Client, Coordinator, RecordingCoordinatorControlPlane]:
    coordinator = Coordinator(
        "inproc://complete",
        "inproc://abort",
        entry_stage="preprocess",
        terminal_stages=["decode"],
    )
    control_plane = control_plane or RecordingCoordinatorControlPlane()
    coordinator.control_plane = control_plane
    coordinator.register_stage("preprocess", "inproc://preprocess")
    return Client(coordinator), coordinator, control_plane


def _http_scope(*, path: str, spec_version: str) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }


class AdminClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], list[str] | None, float]] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def model_info(
        self,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        self.calls.append(("model_info", {}, stages, timeout_s))
        return {
            "success": True,
            "message": "ok",
            "results": [
                {
                    "stage": "decode",
                    "success": True,
                    "message": "ok",
                    "data": {
                        "model_path": "/tmp/current-model",
                        "load_format": "safetensors",
                        "weight_version": "v1",
                    },
                }
            ],
        }

    async def pause_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        self.calls.append(("pause_generation", payload or {}, stages, timeout_s))
        return {"success": True, "message": "ok", "results": []}

    async def continue_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        self.calls.append(("continue_generation", payload or {}, stages, timeout_s))
        return {"success": True, "message": "ok", "results": []}

    async def update_weights_from_disk(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        self.calls.append(("update_weights_from_disk", payload, stages, timeout_s))
        return {"success": True, "message": "ok", "results": []}

    async def init_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        self.calls.append(("init_weights_update_group", payload, stages, timeout_s))
        return {"success": True, "message": "ok", "results": []}

    async def destroy_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        self.calls.append(("destroy_weights_update_group", payload, stages, timeout_s))
        return {"success": True, "message": "ok", "results": []}

    async def update_weights_from_distributed(
        self,
        payload: dict[str, Any],
        *,
        stages: list[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        self.calls.append(
            ("update_weights_from_distributed", payload, stages, timeout_s)
        )
        return {"success": True, "message": "ok", "results": []}

    async def admin(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        self.calls.append((action, payload or {}, stages, timeout_s))
        return {"success": True, "message": "ok", "results": []}

    async def weights_checker(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: list[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        self.calls.append(("weights_checker", payload or {}, stages, timeout_s))
        return {"success": True, "message": "ok", "results": []}


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
            "voice": "default",
            "stream": False,
            "response_format": "wav",
        },
    )
    assert speech_resp.status_code == 500
    assert speech_resp.json()["error"]["type"] == "server_error"
    assert "cuda out of memory" in speech_resp.json()["error"]["message"]


def test_speech_stream_admission_reject_returns_503_without_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = TestClient(
        create_app(
            FailingSpeechGenerateClient(QueueFullError.MESSAGE),
            model_name="s2-pro",
        )
    )

    with caplog.at_level(logging.WARNING, logger="sglang_omni.serve.openai_api"):
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "s2-pro",
                "input": "hello",
                "voice": "default",
                "stream": True,
                "response_format": "pcm",
            },
        )

    assert response.status_code == 503
    assert QueueFullError.MESSAGE in response.json()["error"]["message"]
    assert any(
        rec.levelno == logging.WARNING and "Rejecting speech request" in rec.message
        for rec in caplog.records
    )
    assert not any(rec.exc_info for rec in caplog.records)


def test_speech_endpoint_rejects_invalid_request_with_openai_error() -> None:
    client = TestClient(create_app(SuccessfulSpeechClient(), model_name="tts"))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts",
            "input": "hello",
            "voice": "default",
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
    speech_client = SuccessfulSpeechClient(finish_reason="length")
    client = TestClient(create_app(speech_client, model_name="tts"))

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "hello",
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert response.content == b"RIFF"
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-finish-reason"] == "length"
    assert speech_client.speech_requests[0].model == "tts"
    assert speech_client.speech_requests[0].metadata["tts_params"]["voice"] == "default"


@pytest.mark.parametrize("stream", [False, True])
def test_speech_endpoint_accepts_seedtts_reference_payload_without_voice(
    stream: bool,
) -> None:
    speech_client = SuccessfulSpeechClient()
    client = TestClient(create_app(speech_client, model_name="served-model"))
    ref_audio = base64.b64encode(b"RIFF").decode("ascii")

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "seedtts",
            "input": "hello",
            "ref_audio": f"data:audio/wav;base64,{ref_audio}",
            "ref_text": "reference transcript",
            "response_format": "pcm" if stream else "wav",
            "stream": stream,
        },
    )

    assert response.status_code == 200
    request = (
        speech_client.generate_requests[0]
        if stream
        else speech_client.speech_requests[0]
    )
    assert request.model == "seedtts"
    assert request.metadata["tts_params"]["voice"] == "default"


def test_speech_endpoint_accepts_sdk_shaped_binary_request() -> None:
    speech_client = SuccessfulSpeechClient()
    client = TestClient(create_app(speech_client, model_name="default-tts"))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "voice": "alloy",
            "input": "hello from an SDK-shaped request",
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert response.content == b"RIFF"
    assert response.headers["content-type"] == "audio/wav"
    assert (
        response.headers["content-disposition"] == 'attachment; filename="speech.wav"'
    )
    assert speech_client.speech_requests[0].model == "tts-1"
    assert speech_client.speech_requests[0].metadata["tts_params"]["voice"] == "alloy"


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
        json={
            "model": "tts",
            "input": "hello",
            "voice": "default",
            "stream": True,
            "response_format": "pcm",
        },
    )

    assert response.status_code == 500
    assert response.json()["error"]["type"] == "server_error"
    assert "No audio output generated" in response.json()["error"]["message"]


def test_speech_endpoint_stream_empty_delta_is_not_success() -> None:
    client = TestClient(create_app(EmptyDeltaStreamingSpeechClient(), model_name="tts"))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts",
            "input": "hello",
            "voice": "default",
            "stream": True,
            "response_format": "pcm",
        },
    )

    assert response.status_code == 500
    assert response.json()["error"]["type"] == "server_error"
    assert "No audio output generated" in response.json()["error"]["message"]


def test_admin_routes_forward_to_client() -> None:
    admin = AdminClient()
    client = TestClient(create_app(admin, model_name="qwen3-omni"))

    info = client.get("/model_info")
    pause = client.post(
        "/pause_generation",
        json={"mode": "in_place", "stages": ["decode"], "timeout_s": 5},
    )
    update = client.post(
        "/update_weights_from_disk",
        json={
            "model_path": "/tmp/new-model",
            "load_format": "safetensors",
            "weight_version": "v2",
            "abort_all_requests": True,
        },
    )
    checksum = client.post("/weights_checker", json={"action": "checksum"})

    assert info.status_code == 200
    assert info.json()["weight_version"] == "v1"
    assert info.json()["model_path"] == "/tmp/current-model"
    assert info.json()["load_format"] == "safetensors"
    assert info.json()["stages"][0]["stage"] == "decode"
    assert pause.status_code == 200
    assert update.status_code == 200
    assert checksum.status_code == 200
    assert admin.calls == [
        ("model_info", {}, None, 30.0),
        ("pause_generation", {"mode": "in_place"}, ["decode"], 5),
        (
            "update_weights_from_disk",
            {
                "model_path": "/tmp/new-model",
                "load_format": "safetensors",
                "abort_all_requests": True,
                "weight_version": "v2",
                "is_async": False,
                "torch_empty_cache": False,
                "keep_pause": False,
                "recapture_cuda_graph": False,
                "token_step": 0,
                "flush_cache": True,
            },
            None,
            120.0,
        ),
        ("weights_checker", {"action": "checksum"}, None, 120.0),
    ]


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


def test_chat_asgi_send_failure_aborts_backend_and_cleans_state() -> None:
    async def _run() -> None:
        client, coordinator, control_plane = _streaming_client()
        request_id = "req-asgi-disconnect"
        request = ChatCompletionRequest(
            model="qwen3-omni",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        response = _ClosableStreamingResponse(
            _chat_stream(
                client=client,
                gen_req=GenerateRequest(
                    model="qwen3-omni", prompt="hello", stream=True
                ),
                request_id=request_id,
                response_id=f"chatcmpl-{request_id}",
                created=0,
                model="qwen3-omni",
                req=request,
                audio_format="wav",
            ),
            media_type="text/event-stream",
        )

        body_ready = asyncio.Event()

        async def send(message: dict[str, Any]) -> None:
            if message["type"] != "http.response.body":
                return
            body_ready.set()
            raise RuntimeError("client vanished during body send")

        async def receive() -> dict[str, Any]:
            await asyncio.Event().wait()
            return {"type": "http.disconnect"}

        scope = _http_scope(path="/v1/chat/completions", spec_version="2.4")
        response_task = asyncio.create_task(response(scope, receive, send))
        for _ in range(100):
            if request_id in coordinator._stream_queues:
                break
            await asyncio.sleep(0)
        await coordinator._handle_stream(
            StreamMessage(
                request_id=request_id,
                from_stage="decode",
                chunk={"text": "hello", "modality": "text"},
                modality="text",
            )
        )

        await body_ready.wait()
        with pytest.raises(RuntimeError, match="client vanished during body send"):
            await response_task
        assert [msg.request_id for msg in control_plane.aborts] == [request_id]
        assert request_id not in coordinator._requests
        assert request_id not in coordinator._stream_queues
        assert request_id not in coordinator._completion_futures

    asyncio.run(_run())


def test_chat_asgi_receive_disconnect_aborts_backend_and_cleans_state() -> None:
    async def _run() -> None:
        blocking_control_plane = _BlockingAbortControlPlane()
        client, coordinator, control_plane = _streaming_client(blocking_control_plane)
        request_id = "req-asgi-receive-disconnect"
        request = ChatCompletionRequest(
            model="qwen3-omni",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        response = _ClosableStreamingResponse(
            _chat_stream(
                client=client,
                gen_req=GenerateRequest(
                    model="qwen3-omni", prompt="hello", stream=True
                ),
                request_id=request_id,
                response_id=f"chatcmpl-{request_id}",
                created=0,
                model="qwen3-omni",
                req=request,
                audio_format="wav",
            ),
            media_type="text/event-stream",
        )

        first_body_sent = asyncio.Event()
        disconnected = asyncio.Event()

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.body" and message.get("body"):
                first_body_sent.set()

        async def receive() -> dict[str, Any]:
            await disconnected.wait()
            return {"type": "http.disconnect"}

        response_task = asyncio.create_task(
            response(
                _http_scope(
                    path="/v1/chat/completions",
                    spec_version="2.3",
                ),
                receive,
                send,
            )
        )
        for _ in range(100):
            if request_id in coordinator._stream_queues:
                break
            await asyncio.sleep(0)
        await coordinator._handle_stream(
            StreamMessage(
                request_id=request_id,
                from_stage="decode",
                chunk={"text": "hello", "modality": "text"},
                modality="text",
            )
        )

        await first_body_sent.wait()
        disconnected.set()
        await blocking_control_plane.abort_started.wait()
        await asyncio.wait_for(response_task, timeout=1)

        assert [msg.request_id for msg in control_plane.aborts] == [request_id]
        assert blocking_control_plane.abort_cancelled is False
        abort_task = coordinator._abort_tasks[request_id]
        assert request_id in coordinator._requests
        assert request_id not in coordinator._stream_queues
        assert request_id not in coordinator._completion_futures

        blocking_control_plane.release_abort.set()
        assert await asyncio.wait_for(asyncio.shield(abort_task), timeout=1) is True
        await asyncio.sleep(0)

        assert request_id not in coordinator._requests
        assert request_id not in coordinator._stream_queues
        assert request_id not in coordinator._completion_futures
        assert request_id not in coordinator._abort_tasks

    asyncio.run(_run())


def test_chat_asgi_task_cancellation_aborts_backend_and_stays_cancelled() -> None:
    async def _run() -> None:
        client, coordinator, control_plane = _streaming_client()
        request_id = "req-asgi-cancelled"
        request = ChatCompletionRequest(
            model="qwen3-omni",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        response = _ClosableStreamingResponse(
            _chat_stream(
                client=client,
                gen_req=GenerateRequest(
                    model="qwen3-omni", prompt="hello", stream=True
                ),
                request_id=request_id,
                response_id=f"chatcmpl-{request_id}",
                created=0,
                model="qwen3-omni",
                req=request,
                audio_format="wav",
            ),
            media_type="text/event-stream",
        )

        async def send(_message: dict[str, Any]) -> None:
            return

        async def receive() -> dict[str, Any]:
            await asyncio.Event().wait()
            return {"type": "http.disconnect"}

        response_task = asyncio.create_task(
            response(
                _http_scope(
                    path="/v1/chat/completions",
                    spec_version="2.4",
                ),
                receive,
                send,
            )
        )
        for _ in range(100):
            if request_id in coordinator._stream_queues:
                break
            await asyncio.sleep(0)

        response_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await response_task

        assert [msg.request_id for msg in control_plane.aborts] == [request_id]
        assert request_id not in coordinator._requests
        assert request_id not in coordinator._stream_queues
        assert request_id not in coordinator._completion_futures

    asyncio.run(_run())


def test_client_completion_stream_close_reaches_coordinator_owner() -> None:
    async def _run() -> None:
        client, coordinator, control_plane = _streaming_client()
        request_id = "req-client-close"
        stream = client.completion_stream(
            GenerateRequest(model="qwen3-omni", prompt="hello", stream=True),
            request_id=request_id,
        )
        first_chunk = asyncio.create_task(anext(stream))
        for _ in range(100):
            if request_id in coordinator._stream_queues:
                break
            await asyncio.sleep(0)
        await coordinator._handle_stream(
            StreamMessage(
                request_id=request_id,
                from_stage="decode",
                chunk={"text": "hello", "modality": "text"},
                modality="text",
            )
        )
        await first_chunk
        await stream.aclose()

        assert [msg.request_id for msg in control_plane.aborts] == [request_id]
        assert request_id not in coordinator._requests
        assert request_id not in coordinator._stream_queues
        assert request_id not in coordinator._completion_futures

    asyncio.run(_run())


def test_transcription_stream_close_reaches_coordinator_owner() -> None:
    async def _run() -> None:
        client, coordinator, control_plane = _streaming_client()
        request_id = "req-transcription-close"
        stream = _transcription_stream(
            client.generate(
                GenerateRequest(model="whisper", prompt="hello", stream=True),
                request_id=request_id,
            ),
            first_chunk=None,
            request_id=request_id,
            adapter=_IdentityTranscriptionAdapter(),
            duration_s=1.0,
        )
        first_event = asyncio.create_task(anext(stream))
        for _ in range(100):
            if request_id in coordinator._stream_queues:
                break
            await asyncio.sleep(0)
        await coordinator._handle_stream(
            StreamMessage(
                request_id=request_id,
                from_stage="decode",
                chunk={"text": "hello", "modality": "text"},
                modality="text",
            )
        )
        await first_event
        await stream.aclose()

        assert [msg.request_id for msg in control_plane.aborts] == [request_id]
        assert request_id not in coordinator._requests
        assert request_id not in coordinator._stream_queues
        assert request_id not in coordinator._completion_futures

    asyncio.run(_run())


def test_chat_request_omits_explicit_params_when_sampling_omitted() -> None:
    req = ChatCompletionRequest(
        model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        messages=[{"role": "user", "content": "hello"}],
    )

    gen_req = _build_chat_generate_request(req)

    assert gen_req.sampling.temperature == 1.0
    assert gen_req.sampling.top_p == 1.0
    assert gen_req.sampling.top_k == -1
    assert EXPLICIT_GENERATION_PARAMS_KEY not in gen_req.metadata


def test_chat_request_preserves_explicit_default_sampling_values() -> None:
    req = ChatCompletionRequest(
        model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        messages=[{"role": "user", "content": "hello"}],
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
    )

    gen_req = _build_chat_generate_request(req)

    assert gen_req.sampling.temperature == 1.0
    assert gen_req.sampling.top_p == 1.0
    assert gen_req.sampling.top_k == -1
    assert gen_req.metadata[EXPLICIT_GENERATION_PARAMS_KEY] == [
        "temperature",
        "top_k",
        "top_p",
    ]


def test_chat_request_does_not_mark_null_sampling_params_explicit() -> None:
    req = ChatCompletionRequest(
        model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        messages=[{"role": "user", "content": "hello"}],
        temperature=None,
        top_p=None,
        top_k=None,
    )

    gen_req = _build_chat_generate_request(req)

    assert gen_req.sampling.temperature == 1.0
    assert gen_req.sampling.top_p == 1.0
    assert gen_req.sampling.top_k == -1
    assert EXPLICIT_GENERATION_PARAMS_KEY not in gen_req.metadata


def test_speech_stream_defaults_to_raw_pcm() -> None:
    client = TestClient(
        create_app(SuccessfulSpeechClient(), model_name="higgs-audio-v2")
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "higgs-audio-v2",
            "input": "hello",
            "voice": "default",
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
            "model": "s2-pro",
            "input": "hello",
            "voice": "default",
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
            request=ConnectedRequest(),
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


def test_raw_pcm_response_disconnect_before_first_chunk_aborts_request() -> None:
    async def _drive() -> None:
        client = BlockingFirstAudioStreamingSpeechClient()
        request = DisconnectingRequest()
        task = asyncio.create_task(
            _speech_audio_response(
                request=request,
                client=client,
                gen_req=GenerateRequest(model="s2-pro", prompt="hello", stream=True),
                request_id="req-1",
                speed=1.0,
            )
        )
        await client.started.wait()
        request.disconnected.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.aborted == ["req-1"]

    asyncio.run(_drive())


def test_speech_stream_rejects_non_pcm_response_format() -> None:
    client = TestClient(
        create_app(SuccessfulSpeechClient(), model_name="higgs-audio-v2")
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "higgs-audio-v2",
            "input": "hello",
            "voice": "default",
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


def test_raw_pcm_speech_request_defers_initial_chunk_to_model() -> None:
    req = CreateSpeechRequest(
        input="hello",
        stream=True,
        response_format="pcm",
    )

    gen_req = SpeechRequestValidator(
        default_model="higgs-audio-v2"
    ).build_generate_request(req)

    assert "initial_codec_chunk_frames" not in gen_req.extra_params


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
    assert gen_req.extra_params == {
        "task": "transcribe",
        "language": "en",
    }
    assert gen_req.sampling.temperature == 0.0
    omni_req = Client._build_omni_request(gen_req)
    assert omni_req.params["temperature"] == 0.0
    assert gen_req.metadata == {"task": "asr"}
    assert gen_req.output_modalities == ["text"]
    assert gen_req.stream is False


def test_transcription_request_preserves_explicit_empty_language() -> None:
    gen_req = build_transcription_generate_request(
        audio_bytes=b"RIFF",
        filename="sample.wav",
        content_type="audio/wav",
        model="Qwen/Qwen3-ASR-1.7B",
        language="",
        prompt=None,
        temperature=None,
    )

    assert gen_req.extra_params["language"] == ""
    omni_req = Client._build_omni_request(gen_req)
    assert omni_req.params["language"] == ""


def test_transcription_request_passes_explicit_temperature() -> None:
    gen_req = build_transcription_generate_request(
        audio_bytes=b"RIFF",
        filename="sample.wav",
        content_type="audio/wav",
        model="openai/whisper-large-v3",
        language="en",
        prompt=None,
        temperature=0.7,
    )

    assert gen_req.sampling.temperature == 0.7
    assert gen_req.metadata[EXPLICIT_GENERATION_PARAMS_KEY] == ["temperature"]
    omni_req = Client._build_omni_request(gen_req)
    assert omni_req.params["temperature"] == 0.7


def test_transcription_request_passes_explicit_max_new_tokens() -> None:
    gen_req = build_transcription_generate_request(
        audio_bytes=b"RIFF",
        filename="sample.wav",
        content_type="audio/wav",
        model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        language="en",
        prompt=None,
        temperature=None,
        max_new_tokens=4096,
    )

    assert gen_req.model == "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    assert gen_req.sampling.max_new_tokens == 4096
    assert gen_req.metadata[EXPLICIT_GENERATION_PARAMS_KEY] == ["max_new_tokens"]
    omni_req = Client._build_omni_request(gen_req)
    assert omni_req.params["max_new_tokens"] == 4096


def _wav_upload(duration_s: float, sample_rate: int = 16000) -> bytes:
    """Loud float32 WAV of the given duration."""
    import numpy as np

    from sglang_omni.serve.transcription_chunking import encode_wav

    rng = np.random.default_rng(0)
    samples = int(duration_s * sample_rate)
    return encode_wav(rng.uniform(-0.5, 0.5, samples).astype(np.float32), sample_rate)


def _chunking_test_client(
    transcription_client: Any,
    *,
    max_total_audio_s: float | None = None,
    max_native_clip_s: float | None = None,
) -> TestClient:
    from sglang_omni.config import AudioChunkingConfig

    policy = AudioChunkingConfig(
        allow_audio_chunking=True,
        max_audio_clip_s=1.0,
        max_total_audio_s=max_total_audio_s,
        max_native_clip_s=max_native_clip_s,
    )
    return TestClient(
        create_app(transcription_client, model_name="asr", audio_chunking=policy)
    )


def test_long_audio_is_transcribed_chunk_by_chunk() -> None:
    transcription_client = ChunkRecordingTranscriptionClient()
    client = _chunking_test_client(transcription_client)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr"},
        files={"file": ("long.wav", _wav_upload(2.5), "audio/wav")},
    )

    assert response.status_code == 200
    # Chunk count depends on where the energy search cuts; the contract is
    # covered by the splitter tests. Here: more than one chunk, each sent as
    # its own WAV request with an indexed request id (arrival order is
    # scheduling-dependent now that chunks run concurrently).
    count = len(transcription_client.requests)
    assert count > 1
    seen_ids = {request_id for request_id, _ in transcription_client.requests}
    assert {int(rid.rsplit("-chunk-", 1)[-1]) for rid in seen_ids} == set(range(count))
    for _, request in transcription_client.requests:
        assert request.prompt["audio_bytes"][:4] == b"RIFF"
        assert request.prompt["content_type"] == "audio/wav"
    # Chunk texts are assembled in span order regardless of completion order.
    expected_text = " ".join(f"part{index}" for index in range(count))
    assert response.json()["text"] == expected_text
    # usage reports the whole upload, not one chunk.
    assert response.json()["usage"] == {"seconds": 3, "type": "duration"}


def test_noise_floor_chunks_are_skipped_not_transcribed() -> None:
    # A TTS-runaway-shaped upload: loud speech then a ~-70 dBFS noise floor.
    # Floor-only chunks hallucinate when decoded, so they must never be sent.
    import numpy as np

    from sglang_omni.serve.transcription_chunking import encode_wav

    rng = np.random.default_rng(0)
    loud = rng.uniform(-0.5, 0.5, 8000).astype(np.float32)
    floor = rng.uniform(-3e-4, 3e-4, 32000).astype(np.float32)
    upload = encode_wav(np.concatenate([loud, floor]), 16000)

    transcription_client = ChunkRecordingTranscriptionClient()
    client = _chunking_test_client(transcription_client)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr"},
        files={"file": ("runaway.wav", upload, "audio/wav")},
    )

    assert response.status_code == 200
    # Only the chunk holding speech reached the engine; the floor-only
    # chunks were answered locally with empty text.
    assert len(transcription_client.requests) == 1
    assert response.json()["text"] == "part0"
    # usage still reports the whole upload, skipped chunks included.
    assert response.json()["usage"] == {"seconds": 3, "type": "duration"}


def test_streamed_long_audio_is_rejected_explicitly() -> None:
    # Streaming cannot chunk; without this 400 a too-long upload would run
    # as a single request and truncate (observed live: 243s audio came back
    # 20% short with HTTP 200). No native limit declared here, so the guard
    # falls back to the chunk length.
    transcription_client = ChunkRecordingTranscriptionClient()
    client = _chunking_test_client(transcription_client)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr", "stream": "true"},
        files={"file": ("long.wav", _wav_upload(2.5), "audio/wav")},
    )

    assert response.status_code == 400
    assert "stream=false" in response.json()["detail"]
    assert transcription_client.requests == []


def _count_duration_probes(monkeypatch) -> list[int]:
    from sglang_omni.serve import speech_to_text, transcriptions

    calls: list[int] = []
    real_probe = speech_to_text.probe_audio_duration

    def counting_probe(audio_bytes: bytes) -> float:
        calls.append(1)
        return real_probe(audio_bytes)

    monkeypatch.setattr(speech_to_text, "probe_audio_duration", counting_probe)
    monkeypatch.setattr(transcriptions, "_probe_audio_duration", counting_probe)
    return calls


def test_non_streaming_transcription_probes_the_upload_once(monkeypatch) -> None:
    # The handler probes in a worker thread and hands the duration to the
    # response assembly. A second probe per request was measurable: SeedTTS
    # CI runs ~150 short requests/s, and each sync probe blocks the event
    # loop for the whole process.
    calls = _count_duration_probes(monkeypatch)
    client = _chunking_test_client(SuccessfulTranscriptionClient())

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr"},
        files={"file": ("short.wav", _wav_upload(0.5), "audio/wav")},
    )

    assert response.status_code == 200
    assert len(calls) == 1


def test_streaming_transcription_probes_the_upload_once(monkeypatch) -> None:
    calls = _count_duration_probes(monkeypatch)
    client = _chunking_test_client(SuccessfulTranscriptionClient())

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr", "stream": "true"},
        files={"file": ("short.wav", _wav_upload(0.5), "audio/wav")},
    )

    assert response.status_code == 200
    assert len(calls) == 1


def test_streamed_audio_within_the_native_limit_streams_whole() -> None:
    # The chunk length is a scheduling choice for the non-stream path; a
    # stream request only needs to fit the engine natively. 2.5s is past the
    # 1s chunk length but under the 3s native limit, so it streams as one
    # engine request instead of getting a 400.
    transcription_client = SuccessfulTranscriptionClient()
    client = _chunking_test_client(transcription_client, max_native_clip_s=3.0)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr", "stream": "true"},
        files={"file": ("long.wav", _wav_upload(2.5), "audio/wav")},
    )

    assert response.status_code == 200
    assert "transcript.text.done" in response.text


def test_streamed_audio_beyond_the_native_limit_is_rejected() -> None:
    transcription_client = ChunkRecordingTranscriptionClient()
    client = _chunking_test_client(transcription_client, max_native_clip_s=2.0)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr", "stream": "true"},
        files={"file": ("long.wav", _wav_upload(2.5), "audio/wav")},
    )

    assert response.status_code == 400
    assert "2 seconds" in response.json()["detail"]
    assert transcription_client.requests == []


def test_streamed_short_audio_streams_as_before() -> None:
    transcription_client = SuccessfulTranscriptionClient()
    client = _chunking_test_client(transcription_client)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr", "stream": "true"},
        files={"file": ("short.wav", _wav_upload(0.5), "audio/wav")},
    )

    assert response.status_code == 200
    assert "transcript.text.done" in response.text


def test_verbose_json_reports_per_chunk_segments() -> None:
    transcription_client = ChunkRecordingTranscriptionClient()
    client = _chunking_test_client(transcription_client)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr", "response_format": "verbose_json"},
        files={"file": ("long.wav", _wav_upload(2.5), "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    segments = body["segments"]
    # One segment per chunk, with the chunk's own timestamps -- not a single
    # segment spanning the whole upload.
    assert len(segments) == len(transcription_client.requests) > 1
    previous_end = 0.0
    for index, segment in enumerate(segments):
        assert segment["id"] == index
        assert segment["text"] == f"part{index}"
        assert segment["start"] == pytest.approx(previous_end)
        assert segment["end"] > segment["start"]
        previous_end = segment["end"]
    assert previous_end == pytest.approx(2.5, abs=0.01)
    assert body["duration"] == pytest.approx(2.5, abs=0.01)


def test_chunk_segments_skip_silent_chunks() -> None:
    from sglang_omni.serve.transcription_adapters.base import (
        DefaultTranscriptionAdapter,
    )

    response = DefaultTranscriptionAdapter().build_verbose_response_from_chunks(
        text="hello world",
        chunks=[(0.0, 1.0, "hello"), (1.0, 2.0, "   "), (2.0, 2.5, "world")],
        language="en",
        audio_duration_s=2.5,
    )

    # The silent chunk emits no segment and ids stay consecutive.
    assert [(s.id, s.text) for s in response.segments] == [(0, "hello"), (1, "world")]
    assert response.segments[1].start == 2.0


def test_chunk_failure_fails_the_whole_request() -> None:
    transcription_client = ChunkRecordingTranscriptionClient(fail_chunk=1)
    client = _chunking_test_client(transcription_client)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr"},
        files={"file": ("long.wav", _wav_upload(2.5), "audio/wav")},
    )

    # No partial 200: one failed chunk fails the request, naming the chunk.
    assert response.status_code == 500
    assert "chunk 1" in response.json()["detail"]
    assert "cuda out of memory" in response.json()["detail"]


def test_chunk_bad_request_failure_maps_to_400() -> None:
    transcription_client = ChunkRecordingTranscriptionClient(
        fail_chunk=0,
        fail_message="Requested audio is longer than the model's context length",
    )
    client = _chunking_test_client(transcription_client)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr"},
        files={"file": ("long.wav", _wav_upload(2.5), "audio/wav")},
    )

    assert response.status_code == 400
    assert "chunk 0" in response.json()["detail"]


def test_upload_over_the_total_duration_limit_is_rejected() -> None:
    transcription_client = ChunkRecordingTranscriptionClient()
    client = _chunking_test_client(transcription_client, max_total_audio_s=2.0)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr"},
        files={"file": ("long.wav", _wav_upload(2.5), "audio/wav")},
    )

    assert response.status_code == 400
    assert "accepts audio up to" in response.json()["detail"]
    # Rejected before any engine request or decode.
    assert transcription_client.requests == []


def test_total_duration_limit_is_re_enforced_on_the_decoded_audio(monkeypatch) -> None:
    # A metadata probe can under-measure estimated containers; once the
    # decode reveals the true duration the cap must still hold.
    from sglang_omni.serve import transcriptions

    monkeypatch.setattr(
        transcriptions, "_probe_audio_duration", lambda audio_bytes: 1.5
    )
    transcription_client = ChunkRecordingTranscriptionClient()
    client = _chunking_test_client(transcription_client, max_total_audio_s=2.0)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr"},
        files={"file": ("long.wav", _wav_upload(2.5), "audio/wav")},
    )

    assert response.status_code == 400
    assert "accepts audio up to" in response.json()["detail"]
    assert transcription_client.requests == []


def test_long_audio_without_chunking_policy_stays_one_request() -> None:
    # No audio_chunking passed to create_app: models that have not declared a
    # policy keep today's single-request behaviour, byte for byte.
    transcription_client = ChunkRecordingTranscriptionClient()
    client = TestClient(create_app(transcription_client, model_name="asr"))
    upload = _wav_upload(2.5)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr"},
        files={"file": ("long.wav", upload, "audio/wav")},
    )

    assert response.status_code == 200
    assert len(transcription_client.requests) == 1
    request_id, request = transcription_client.requests[0]
    assert "-chunk-" not in request_id
    assert request.prompt["audio_bytes"] == upload


def test_short_audio_with_chunking_enabled_stays_one_request() -> None:
    transcription_client = ChunkRecordingTranscriptionClient()
    client = _chunking_test_client(transcription_client)
    upload = _wav_upload(0.5)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr"},
        files={"file": ("short.wav", upload, "audio/wav")},
    )

    assert response.status_code == 200
    assert len(transcription_client.requests) == 1
    assert transcription_client.requests[0][1].prompt["audio_bytes"] == upload


def _tiny_plan(num_chunks: int):
    import numpy as np

    from sglang_omni.serve.transcription_chunking import ChunkPlan, ChunkSpan

    chunk_samples = 1600
    spans = [
        ChunkSpan(
            index=i,
            start_sample=i * chunk_samples,
            end_sample=(i + 1) * chunk_samples,
            start_s=i * 0.1,
            end_s=(i + 1) * 0.1,
        )
        for i in range(num_chunks)
    ]
    return ChunkPlan(
        sample_rate=16000,
        duration_s=num_chunks * 0.1,
        spans=spans,
        waveform=np.zeros(num_chunks * chunk_samples, dtype=np.float32),
    )


def _run_chunks(client: Any, plan: Any, *, max_concurrent: int):
    from sglang_omni.serve.transcriptions import _transcribe_audio_chunks

    return asyncio.wait_for(
        _transcribe_audio_chunks(
            client,
            plan,
            request_id="req",
            model="asr",
            filename=None,
            language=None,
            prompt=None,
            temperature=None,
            max_new_tokens=None,
            max_concurrent=max_concurrent,
        ),
        timeout=10.0,
    )


def test_chunks_run_concurrently() -> None:
    class BarrierClient:
        """completion() blocks until all expected chunks have arrived.

        A serial implementation deadlocks here (chunk 0 waits forever for
        peers that are never submitted); only a concurrent one passes.
        """

        def __init__(self, expected: int) -> None:
            self.expected = expected
            self.active = 0
            self.max_active = 0
            self.all_in = asyncio.Event()

        async def completion(self, request, *, request_id, **kwargs):
            from sglang_omni.client.types import CompletionResult

            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active >= self.expected:
                self.all_in.set()
            await self.all_in.wait()
            self.active -= 1
            index = request_id.rsplit("-chunk-", 1)[-1]
            return CompletionResult(request_id=request_id, text=f"part{index}")

    async def scenario() -> None:
        barrier_client = BarrierClient(expected=3)
        texts = await _run_chunks(barrier_client, _tiny_plan(3), max_concurrent=3)
        assert texts == ["part0", "part1", "part2"]
        assert barrier_client.max_active == 3

    asyncio.run(scenario())


def test_chunk_concurrency_respects_the_limit() -> None:
    class CountingClient:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def completion(self, request, *, request_id, **kwargs):
            from sglang_omni.client.types import CompletionResult

            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            index = request_id.rsplit("-chunk-", 1)[-1]
            return CompletionResult(request_id=request_id, text=f"part{index}")

    async def scenario() -> None:
        counting_client = CountingClient()
        texts = await _run_chunks(counting_client, _tiny_plan(6), max_concurrent=2)
        assert texts == [f"part{i}" for i in range(6)]
        assert counting_client.max_active <= 2

    asyncio.run(scenario())


def test_chunk_texts_join_in_span_order_not_completion_order() -> None:
    class ReversedLatencyClient:
        async def completion(self, request, *, request_id, **kwargs):
            from sglang_omni.client.types import CompletionResult

            index = int(request_id.rsplit("-chunk-", 1)[-1])
            # Later chunks finish first.
            await asyncio.sleep(0.03 * (3 - index))
            return CompletionResult(request_id=request_id, text=f"part{index}")

    async def scenario() -> None:
        texts = await _run_chunks(
            ReversedLatencyClient(), _tiny_plan(3), max_concurrent=3
        )
        assert texts == ["part0", "part1", "part2"]

    asyncio.run(scenario())


def test_chunk_failure_aborts_the_chunks_still_running() -> None:
    class OneFailsOthersHangClient:
        def __init__(self) -> None:
            self.aborted: list[str] = []
            self.hung = asyncio.Event()

        async def completion(self, request, *, request_id, **kwargs):
            from sglang_omni.client import ClientError

            index = int(request_id.rsplit("-chunk-", 1)[-1])
            if index == 1:
                # Let the others reach their await first.
                await asyncio.sleep(0.01)
                raise ClientError("cuda out of memory")
            self.hung.set()
            await asyncio.Future()  # hangs until cancelled

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    async def scenario() -> None:
        hang_client = OneFailsOthersHangClient()
        from sglang_omni.client import ClientError

        with pytest.raises(ClientError, match="chunk 1"):
            await _run_chunks(hang_client, _tiny_plan(3), max_concurrent=3)
        # The hanging engine requests were aborted by id.
        assert sorted(hang_client.aborted) == ["req-chunk-0", "req-chunk-2"]

    asyncio.run(scenario())


def test_client_disconnect_aborts_all_running_chunks() -> None:
    from sglang_omni.serve.transcriptions import (
        _await_transcription_with_disconnect_abort,
        _transcribe_audio_chunks,
    )

    class HangingClient:
        def __init__(self, expected: int) -> None:
            self.expected = expected
            self.arrived = 0
            self.all_started = asyncio.Event()
            self.aborted: list[str] = []

        async def completion(self, request, *, request_id, **kwargs):
            self.arrived += 1
            if self.arrived >= self.expected:
                self.all_started.set()
            await asyncio.Future()

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    class DisconnectingRequest:
        """Reports disconnected only once every chunk is in flight."""

        def __init__(self, all_started: asyncio.Event) -> None:
            self.all_started = all_started

        async def is_disconnected(self) -> bool:
            return self.all_started.is_set()

    async def scenario() -> None:
        hanging_client = HangingClient(expected=2)
        work = _transcribe_audio_chunks(
            hanging_client,
            _tiny_plan(2),
            request_id="req",
            model="asr",
            filename=None,
            language=None,
            prompt=None,
            temperature=None,
            max_new_tokens=None,
            max_concurrent=2,
        )
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(
                _await_transcription_with_disconnect_abort(
                    DisconnectingRequest(hanging_client.all_started), work
                ),
                timeout=10.0,
            )
        assert sorted(hanging_client.aborted) == ["req-chunk-0", "req-chunk-1"]

    asyncio.run(scenario())


def test_cancelling_the_wrapper_itself_aborts_running_chunks() -> None:
    # The wrapper's handler task can be cancelled from outside (server
    # shutdown, ASGI teardown) rather than via is_disconnected(). The
    # finally block must stop the work task too, or its engine requests
    # keep running with nobody left to abort them.
    from sglang_omni.serve.transcriptions import (
        _await_transcription_with_disconnect_abort,
        _transcribe_audio_chunks,
    )

    class HangingClient:
        def __init__(self, expected: int) -> None:
            self.expected = expected
            self.arrived = 0
            self.all_started = asyncio.Event()
            self.aborted: list[str] = []

        async def completion(self, request, *, request_id, **kwargs):
            self.arrived += 1
            if self.arrived >= self.expected:
                self.all_started.set()
            await asyncio.Future()

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    class NeverDisconnects:
        async def is_disconnected(self) -> bool:
            return False

    async def scenario() -> None:
        hanging_client = HangingClient(expected=2)
        work = _transcribe_audio_chunks(
            hanging_client,
            _tiny_plan(2),
            request_id="req",
            model="asr",
            filename=None,
            language=None,
            prompt=None,
            temperature=None,
            max_new_tokens=None,
            max_concurrent=2,
        )
        wrapper_task = asyncio.create_task(
            _await_transcription_with_disconnect_abort(NeverDisconnects(), work)
        )
        await asyncio.wait_for(hanging_client.all_started.wait(), timeout=10.0)
        wrapper_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wrapper_task
        # The work task's cleanup may finish just after the bounded wait; give
        # it a few ticks before asserting the engine-side aborts happened.
        for _ in range(50):
            if len(hanging_client.aborted) == 2:
                break
            await asyncio.sleep(0.01)
        assert sorted(hanging_client.aborted) == ["req-chunk-0", "req-chunk-1"]

    asyncio.run(scenario())


def _m4a_upload(duration_s: float, sample_rate: int = 16000) -> bytes:
    """Loud AAC/M4A clip -- a format libsndfile cannot inspect."""
    import io as io_module

    import av
    import numpy as np

    rng = np.random.default_rng(0)
    wave = rng.uniform(-0.5, 0.5, int(duration_s * sample_rate)).astype(np.float32)
    buffer = io_module.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    audio_stream = container.add_stream("aac", rate=sample_rate)
    audio_stream.layout = "mono"
    frame = av.AudioFrame.from_ndarray(
        wave.reshape(1, -1), format="fltp", layout="mono"
    )
    frame.sample_rate = sample_rate
    for packet in audio_stream.encode(frame):
        container.mux(packet)
    for packet in audio_stream.encode():
        container.mux(packet)
    container.close()
    return buffer.getvalue()


def test_long_m4a_is_probed_and_chunked() -> None:
    # libsndfile cannot inspect M4A, but load_audio (torchaudio/FFmpeg) can
    # decode it. The probe must cover everything the decoder covers --
    # otherwise a long M4A bypasses chunking and the total-duration guard,
    # runs as one request, and gets silently truncated at the output budget.
    transcription_client = ChunkRecordingTranscriptionClient()
    client = _chunking_test_client(transcription_client)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr"},
        files={"file": ("long.m4a", _m4a_upload(2.5), "audio/mp4")},
    )

    assert response.status_code == 200
    assert len(transcription_client.requests) > 1
    assert all(
        "-chunk-" in request_id for request_id, _ in transcription_client.requests
    )


def test_unprobeable_audio_with_chunking_enabled_stays_one_request() -> None:
    # soundfile cannot read these bytes, so the duration probe returns 0.0 and
    # the upload keeps today's behaviour instead of paying a decode.
    transcription_client = ChunkRecordingTranscriptionClient()
    client = _chunking_test_client(transcription_client)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr"},
        files={"file": ("mystery.bin", b"RIFF not really audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert len(transcription_client.requests) == 1
    assert (
        transcription_client.requests[0][1].prompt["audio_bytes"]
        == b"RIFF not really audio"
    )


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


def test_transcription_endpoint_maps_disallowed_special_token_to_400() -> None:
    transcription_client = FailingTranscriptionClient(
        "Encountered text in the prompt corresponding to disallowed "
        "special token: <|startoftranscript|>."
    )
    client = TestClient(
        create_app(
            transcription_client,
            model_name="openai/whisper-large-v3",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "openai/whisper-large-v3",
            "prompt": "<|startoftranscript|> sneaky",
        },
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 400
    assert "disallowed special token" in response.json()["detail"]


def test_transcription_endpoint_maps_bad_request_error_to_400() -> None:

    bad_request_error = (
        "Fun-ASR accepts audio up to 30.0 seconds because its official "
        "VAD segment limit is 30 seconds; split longer audio before inference."
    )
    client = TestClient(
        create_app(
            _fault_client("qwen3-omni", error=bad_request_error),
            model_name="qwen3-omni",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "qwen3-omni"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 400
    assert "accepts audio up to" in response.json()["detail"]


def test_transcription_endpoint_maps_invalid_audio_error_to_400() -> None:
    client = TestClient(
        create_app(
            _fault_client(
                "qwen3-omni",
                error=(
                    "Qwen3-ASR could not decode the uploaded audio; "
                    "provide a valid audio file."
                ),
            ),
            model_name="qwen3-omni",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "qwen3-omni"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 400
    assert "could not decode the uploaded audio" in response.json()["detail"]


def test_transcription_endpoint_preserves_audio_backend_error_as_500() -> None:
    client = TestClient(
        create_app(
            _fault_client("qwen3-omni", error="audio decoder backend unavailable"),
            model_name="qwen3-omni",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "qwen3-omni"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 500
    assert "audio decoder backend unavailable" in response.json()["detail"]


def test_transcription_endpoint_maps_kv_capacity_error_to_400() -> None:
    bad_request_error = (
        "Request requires more tokens than the thinker KV cache can hold "
        "(input_tokens=1500, max_new_tokens=128, required_tokens=1628, "
        "kv_capacity=1600)."
    )
    client = TestClient(
        create_app(
            _fault_client("qwen3-omni", error=bad_request_error),
            model_name="qwen3-omni",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "qwen3-omni"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 400
    assert "thinker KV cache" in response.json()["detail"]


def test_transcription_endpoint_maps_max_new_tokens_error_to_400() -> None:
    bad_request_error = "max_new_tokens must be between 1 and 200, got 65536"
    client = TestClient(
        create_app(
            _fault_client("qwen3-omni", error=bad_request_error),
            model_name="qwen3-omni",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "qwen3-omni"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 400
    assert "max_new_tokens must be" in response.json()["detail"]


def test_transcription_endpoint_maps_unsupported_language_error_to_400() -> None:
    bad_request_error = (
        "Unsupported language: 'Klingon'. Use a supported language code "
        "(ar, en, zh) or canonical name."
    )
    client = TestClient(
        create_app(
            _fault_client("qwen3-omni", error=bad_request_error),
            model_name="qwen3-omni",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "qwen3-omni", "language": "Klingon"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 400
    assert "Unsupported language:" in response.json()["detail"]
    assert "Use a supported language code" in response.json()["detail"]


def test_transcription_endpoint_passes_explicit_max_new_tokens() -> None:
    transcription_client = SuccessfulTranscriptionClient()
    client = TestClient(
        create_app(
            transcription_client,
            model_name="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            "max_new_tokens": "4096",
        },
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 200
    assert transcription_client.requests
    request = transcription_client.requests[0]
    assert request.model == "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    assert request.sampling.max_new_tokens == 4096
    assert request.metadata[EXPLICIT_GENERATION_PARAMS_KEY] == ["max_new_tokens"]


def test_transcription_endpoint_maps_input_length_error_to_400() -> None:
    transcription_client = FailingTranscriptionClient(
        "Input length (140000 tokens) exceeds the maximum allowed length "
        "(131071 tokens). Use a shorter input or enable --allow-auto-truncate."
    )
    client = TestClient(
        create_app(
            transcription_client,
            model_name="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "OpenMOSS-Team/MOSS-Transcribe-Diarize"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 400
    assert "exceeds the maximum allowed length" in response.json()["detail"]


def test_transcription_endpoint_maps_runtime_input_length_error_to_400() -> None:
    transcription_client = FailingTranscriptionClient(
        "Input length (140000 tokens) exceeds the maximum allowed length "
        "(131071 tokens).",
        exc_type=RuntimeError,
    )
    client = TestClient(
        create_app(
            transcription_client,
            model_name="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "OpenMOSS-Team/MOSS-Transcribe-Diarize"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 400


def test_transcription_endpoint_maps_processor_max_length_error_to_400() -> None:
    transcription_client = FailingTranscriptionClient(
        "Prompt/audio sequence exceeds max_length=132096"
    )
    client = TestClient(
        create_app(
            transcription_client,
            model_name="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "OpenMOSS-Team/MOSS-Transcribe-Diarize"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 400
    assert "exceeds max_length" in response.json()["detail"]


def test_transcription_endpoint_keeps_500_for_server_errors() -> None:
    transcription_client = FailingTranscriptionClient("scheduler worker crashed")
    client = TestClient(
        create_app(
            transcription_client,
            model_name="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "OpenMOSS-Team/MOSS-Transcribe-Diarize"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 500


def test_transcription_stream_emits_delta_done_and_sentinel() -> None:
    transcription_client = SuccessfulTranscriptionClient()
    client = TestClient(
        create_app(
            transcription_client,
            model_name="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            "stream": "true",
        },
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    first_delta = body.index('"transcript.text.delta"')
    done_index = body.index('"transcript.text.done"')
    sentinel_index = body.index("data: [DONE]")
    assert first_delta < done_index < sentinel_index
    assert '"delta":"hello "' in body
    assert '"delta":"world"' in body
    assert '"text":"hello world"' in body


def test_transcription_stream_maps_input_length_error_to_400() -> None:
    transcription_client = FailingTranscriptionClient(
        "Input length (140000 tokens) exceeds the maximum allowed length "
        "(131071 tokens).",
        exc_type=RuntimeError,
    )
    client = TestClient(
        create_app(
            transcription_client,
            model_name="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            "stream": "true",
        },
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 400
    assert "exceeds the maximum allowed length" in response.json()["detail"]


def test_transcription_first_chunk_disconnect_aborts_backend() -> None:
    async def _drive() -> None:
        aborts: list[str] = []
        stream_closed = asyncio.Event()

        class _AbortRecordingClient:
            async def abort(self, request_id: str) -> None:
                aborts.append(request_id)

        async def _never_first_chunk():
            try:
                await asyncio.Event().wait()
            finally:
                stream_closed.set()
            yield  # unreachable, makes this an async generator

        request = DisconnectingRequest()
        request.disconnected.set()
        with pytest.raises(asyncio.CancelledError):
            await _first_transcription_chunk(
                request,
                _AbortRecordingClient(),
                _never_first_chunk(),
                "transcription-1",
            )
        assert aborts == ["transcription-1"]
        assert stream_closed.is_set()

    asyncio.run(_drive())


def test_transcription_stream_keeps_500_for_server_errors() -> None:
    transcription_client = FailingTranscriptionClient(
        "scheduler worker crashed",
        exc_type=RuntimeError,
    )
    client = TestClient(
        create_app(
            transcription_client,
            model_name="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            "stream": "true",
        },
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 500


def test_transcription_endpoint_uses_openai_temperature_default() -> None:
    transcription_client = SuccessfulTranscriptionClient()
    client = TestClient(
        create_app(transcription_client, model_name="openai/whisper-large-v3")
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "openai/whisper-large-v3"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 200
    assert transcription_client.requests
    request = transcription_client.requests[0]
    assert request.sampling.temperature == 0.0
    assert EXPLICIT_GENERATION_PARAMS_KEY not in request.metadata


def test_transcription_endpoint_marks_mtd_request_for_model_sampling_defaults() -> None:
    transcription_client = SuccessfulTranscriptionClient()
    client = TestClient(
        create_app(
            transcription_client,
            model_name="OpenMOSS-Team/MOSS-Transcribe-Diarize",
            architectures=["MossTranscribeDiarizeForConditionalGeneration"],
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "OpenMOSS-Team/MOSS-Transcribe-Diarize"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 200
    assert transcription_client.requests
    request = transcription_client.requests[0]
    assert request.sampling.temperature == 0.0
    assert EXPLICIT_GENERATION_PARAMS_KEY not in request.metadata


class DiarizationTranscriptionClient:
    """Stub returning MOSS-style diarized markup for verbose_json tests."""

    async def completion(self, request, *, request_id, audio_format="wav"):
        from sglang_omni.client.types import CompletionResult

        del request, request_id, audio_format
        return CompletionResult(
            request_id="transcription-1",
            text="[0.00][S01] hello there.[1.20][1.30][S02] bye.[3.00]",
        )


def test_transcription_verbose_json_returns_diarized_segments() -> None:
    client = TestClient(
        create_app(
            DiarizationTranscriptionClient(),
            model_name="moss-transcribe-diarize",
            architectures=["MossTranscribeDiarizeForConditionalGeneration"],
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "moss-transcribe-diarize", "response_format": "verbose_json"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "transcribe"
    assert [(s["id"], s["start"], s["end"], s["text"]) for s in body["segments"]] == [
        (0, 0.0, 1.2, "[S01]hello there."),
        (1, 1.3, 3.0, "[S02]bye."),
    ]


def test_transcription_verbose_json_falls_back_for_plain_text() -> None:
    client = TestClient(
        create_app(
            SuccessfulTranscriptionClient(),
            model_name="moss-transcribe-diarize",
            architectures=["MossTranscribeDiarizeForConditionalGeneration"],
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "moss-transcribe-diarize", "response_format": "verbose_json"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["segments"]) == 1
    assert body["segments"][0]["text"] == "[S01]hello world"


def _wav_bytes(duration_s: float, sample_rate: int = 16000) -> bytes:
    import io

    import numpy as np
    import soundfile as sf

    samples = np.zeros(int(duration_s * sample_rate), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


def test_transcription_probes_duration_from_real_wav() -> None:
    client = TestClient(
        create_app(
            DiarizationTranscriptionClient(),
            model_name="moss-transcribe-diarize",
            architectures=["MossTranscribeDiarizeForConditionalGeneration"],
        )
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "moss-transcribe-diarize", "response_format": "verbose_json"},
        files={"file": ("sample.wav", _wav_bytes(3.5), "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["usage"] == {"type": "duration", "seconds": 4}


def test_speech_request_passes_moss_token_count() -> None:
    req = CreateSpeechRequest(input="hello", token_count=180)

    gen_req = SpeechRequestValidator(default_model="moss-tts").build_generate_request(
        req
    )

    assert gen_req.metadata["tts_params"]["token_count"] == 180


# ---------------------------------------------------------------------------
# Admin auth tests
# ---------------------------------------------------------------------------

_ADMIN_PATHS_THAT_NEED_AUTH = [
    ("GET", "/model_info"),
    ("POST", "/model_info"),
    ("POST", "/pause_generation"),
    ("POST", "/continue_generation"),
    ("POST", "/update_weights_from_disk"),
    ("POST", "/update_weights_from_tensor"),
    ("POST", "/update_weights_from_distributed"),
    ("POST", "/init_weights_update_group"),
    ("POST", "/destroy_weights_update_group"),
    ("GET", "/weights_checker"),
    ("POST", "/weights_checker"),
]

_ADMIN_API_KEY = "secret-key"


def _admin_headers(
    key: str = _ADMIN_API_KEY,
    *,
    scheme: str = "Bearer",
) -> dict[str, str]:
    return {"Authorization": f"{scheme} {key}"}


def test_admin_routes_open_when_no_key_configured() -> None:
    """Without a key, all admin routes are accessible with no auth header."""
    admin = AdminClient()
    client = TestClient(create_app(admin, model_name="qwen3-omni"))

    resp = client.get("/model_info")
    assert resp.status_code == 200

    resp = client.post("/pause_generation", json={})
    assert resp.status_code == 200


def test_admin_routes_require_bearer_token_when_key_configured() -> None:
    """When admin_api_key is set, requests without the header are rejected."""
    admin = AdminClient()
    client = TestClient(
        create_app(admin, model_name="qwen3-omni", admin_api_key=_ADMIN_API_KEY)
    )

    for method, path in _ADMIN_PATHS_THAT_NEED_AUTH:
        resp = client.request(method, path, json={})
        assert (
            resp.status_code == 401
        ), f"{method} {path} should be 401, got {resp.status_code}"
        assert "WWW-Authenticate" in resp.headers


def test_admin_routes_reject_wrong_bearer_token() -> None:
    admin = AdminClient()
    client = TestClient(
        create_app(admin, model_name="qwen3-omni", admin_api_key=_ADMIN_API_KEY)
    )

    for method, path in _ADMIN_PATHS_THAT_NEED_AUTH:
        resp = client.request(
            method, path, json={}, headers=_admin_headers("wrong-key")
        )
        assert (
            resp.status_code == 403
        ), f"{method} {path} should be 403, got {resp.status_code}"


def test_admin_routes_accept_correct_bearer_token() -> None:
    admin = AdminClient()
    client = TestClient(
        create_app(admin, model_name="qwen3-omni", admin_api_key=_ADMIN_API_KEY)
    )

    resp = client.get("/model_info", headers=_admin_headers(scheme="bearer"))
    assert resp.status_code == 200

    resp = client.post(
        "/pause_generation",
        json={},
        headers=_admin_headers(),
    )
    assert resp.status_code == 200


def test_admin_routes_env_key_is_used_when_no_explicit_key(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_ADMIN_KEY", "env-key")
    admin = AdminClient()
    client = TestClient(create_app(admin, model_name="qwen3-omni"))

    resp = client.get("/model_info")
    assert resp.status_code == 401

    resp = client.get("/model_info", headers=_admin_headers("env-key"))
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Stub endpoint 501 tests
# ---------------------------------------------------------------------------


def test_unimplemented_tensor_weight_update_returns_501() -> None:
    admin = AdminClient()
    client = TestClient(create_app(admin, model_name="qwen3-omni"))

    resp = client.post("/update_weights_from_tensor", json={})
    assert resp.status_code == 501
    assert resp.json()["error"]["code"] == "not_implemented"
    assert "update_weights_from_disk" in resp.json()["error"]["message"]


def test_distributed_weight_update_routes_forward_to_client() -> None:
    admin = AdminClient()
    client = TestClient(create_app(admin, model_name="qwen3-omni"))

    init = client.post(
        "/init_weights_update_group",
        json={
            "master_address": "10.0.0.1",
            "master_port": 12355,
            "world_size": 2,
            "rank_offset": 1,
            "stages": ["talker"],
            "timeout_s": 0,
        },
    )
    update = client.post(
        "/update_weights_from_distributed",
        json={
            "names": ["w.0"],
            "dtypes": ["bfloat16"],
            "shapes": [[2, 2]],
            "group_name": "weight_update_group",
            "weight_version": "v2",
            "timeout_s": 0,
        },
    )
    destroy = client.post(
        "/destroy_weights_update_group",
        json={
            "group_name": "weight_update_group",
            "stages": ["talker"],
            "timeout_s": 0,
        },
    )

    assert init.status_code == 200
    assert update.status_code == 200
    assert destroy.status_code == 200
    assert admin.calls == [
        (
            "init_weights_update_group",
            {
                "master_address": "10.0.0.1",
                "master_port": 12355,
                "world_size": 2,
                "rank_offset": 1,
                "group_name": "weight_update_group",
                "backend": "nccl",
            },
            ["talker"],
            0,
        ),
        (
            "update_weights_from_distributed",
            {
                "names": ["w.0"],
                "dtypes": ["bfloat16"],
                "shapes": [[2, 2]],
                "group_name": "weight_update_group",
                "flush_cache": True,
                "abort_all_requests": False,
                "weight_version": "v2",
                "torch_empty_cache": False,
            },
            None,
            0,
        ),
        (
            "destroy_weights_update_group",
            {"group_name": "weight_update_group"},
            ["talker"],
            0,
        ),
    ]


def test_stub_endpoint_checks_auth_before_501() -> None:
    """Auth check fires before the tensor stub 501 body."""
    admin = AdminClient()
    client = TestClient(
        create_app(admin, model_name="qwen3-omni", admin_api_key=_ADMIN_API_KEY)
    )

    resp = client.post("/update_weights_from_tensor", json={})
    assert resp.status_code == 401
