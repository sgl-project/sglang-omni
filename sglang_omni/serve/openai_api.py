# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible API server for sglang-omni.

Provides the following endpoints:
- POST /v1/chat/completions  — Text (+ audio) chat completions
- POST /v1/audio/speech      — Text-to-speech synthesis
- POST /v1/audio/speech/batch — Batch text-to-speech synthesis
- WS   /v1/audio/speech/stream — Stateful TTS WebSocket streaming
- GET  /v1/audio/voices      — List preset and uploaded TTS voices
- POST /v1/audio/voices      — Upload a persistent TTS reference voice
- DELETE /v1/audio/voices/{name} — Delete an uploaded TTS voice
- GET  /v1/models            — List available models
- GET  /v1/fs/list           — Browse filesystem directories
- GET  /v1/fs/file           — Download a file
- GET  /health               — Health check
- WS   /v1/realtime          — OpenAI-compatible Realtime API (when enabled)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, AsyncIterator

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)

from sglang_omni.client import (
    Client,
    ClientError,
    CompletionResult,
    GenerateRequest,
    Message,
    SamplingParams,
)
from sglang_omni.client.audio import (
    DEFAULT_SAMPLE_RATE,
    apply_speed,
    encode_pcm,
    select_audio_delta,
)
from sglang_omni.http.admin_auth import (
    make_admin_auth_dependency,
    resolve_admin_api_key,
)
from sglang_omni.http.favicon import register_favicon
from sglang_omni.serve.protocol import (
    DEFAULT_TTS_BATCH_MAX_ITEMS,
    AdminRequestBase,
    ChatCompletionAudio,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatCompletionStreamResponse,
    ContinueGenerationRequest,
    CreateSpeechBatchRequest,
    DestroyWeightsUpdateGroupRequest,
    GenerateAudio,
    GenerateFinishReason,
    GenerateMetaInfo,
    GenerateResponse,
    InitWeightsUpdateGroupRequest,
    ModelCard,
    ModelList,
    PauseGenerationRequest,
    RolloutGenerateRequest,
    RolloutSamplingParams,
    SpeechBatchResponse,
    TranscriptionResponse,
    TranscriptionUsage,
    UpdateWeightFromDiskRequest,
    UpdateWeightsFromDistributedRequest,
    UsageResponse,
    VoiceListResponse,
    WeightsCheckerRequest,
)
from sglang_omni.serve.speech_errors import (
    SpeechAPIError,
    bad_request,
    internal_error,
    openai_error_payload,
    speech_error_response,
)
from sglang_omni.serve.speech_service import SpeechRequestValidator
from sglang_omni.serve.speech_voices import MAX_VOICE_UPLOAD_BYTES, SpeakerSampleStore
from sglang_omni.serve.speech_ws import SpeechWebSocketSession
from sglang_omni.serve.transcription_adapters import resolve_adapter

logger = logging.getLogger(__name__)
STREAM_DONE_SENTINEL = "[DONE]"
HTTP_DISCONNECT_POLL_INTERVAL_S = 0.05
HTTP_DISCONNECT_CANCEL_TIMEOUT_S = 0.1
VOICE_UPLOAD_MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_VOICE_UPLOAD_BODY_BYTES = (
    MAX_VOICE_UPLOAD_BYTES + VOICE_UPLOAD_MULTIPART_OVERHEAD_BYTES
)

_BAD_REQUEST_MARKERS = (
    "longer than the model's context length",
    "Requested token count exceeds the model's maximum context length",
)


def _is_bad_request_error(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in _BAD_REQUEST_MARKERS)


class _RequestBodyTooLarge(Exception):
    pass


class VoiceUploadBodyLimitMiddleware:
    """Reject oversized voice uploads before Starlette parses multipart bodies."""

    def __init__(self, app: Callable[..., Awaitable[None]], max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if not _is_voice_upload_scope(scope):
            await self.app(scope, receive, send)
            return

        content_length = _content_length(scope)
        if content_length is not None and content_length > self.max_bytes:
            await _send_voice_upload_too_large(send, self.max_bytes)
            return

        received_bytes = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await _send_voice_upload_too_large(send, self.max_bytes)


def create_app(
    client: Client,
    *,
    model_name: str | None = None,
    requires_uploaded_voice_for_named_voice: bool = False,
    supports_uploaded_voice_references: bool = True,
    enable_realtime: bool = False,
    allowed_local_media_path: str | None = None,
    allowed_media_domains: list[str] | None = None,
    admin_api_key: str | None = None,
    tts_batch_max_items: int = DEFAULT_TTS_BATCH_MAX_ITEMS,
    architectures: list[str] | None = None,
) -> FastAPI:
    """Create a FastAPI application with OpenAI-compatible endpoints.

    Args:
        client: Client instance connected to the pipeline coordinator.
        model_name: Default model name to report in responses and /v1/models.
        requires_uploaded_voice_for_named_voice: Whether non-default TTS voice
            names must resolve to uploaded voices before reaching the model.
        supports_uploaded_voice_references: Whether uploaded voice names can be
            lowered into backend reference-audio requests.
        enable_realtime: If True, mount the WebSocket ``/v1/realtime``
            endpoint (OpenAI Realtime API).
        allowed_local_media_path: Directory allowed for ``file://`` TTS
            reference audio.
        allowed_media_domains: Domains allowed for remote TTS reference audio.
        admin_api_key: Optional API key for admin-control endpoints.
        tts_batch_max_items: Maximum items accepted by
            ``/v1/audio/speech/batch``.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="sglang-omni", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        VoiceUploadBodyLimitMiddleware,
        max_bytes=MAX_VOICE_UPLOAD_BODY_BYTES,
    )

    # Store references in app state for access from route handlers
    app.state.client = client
    app.state.model_name = model_name or "sglang-omni"
    app.state.architectures = [a for a in (architectures or []) if a]
    app.state.realtime_enabled = enable_realtime
    app.state.speaker_sample_store = SpeakerSampleStore()
    app.state.speech_service = SpeechRequestValidator(
        default_model=app.state.model_name,
        requires_uploaded_voice_for_named_voice=(
            requires_uploaded_voice_for_named_voice
        ),
        supports_uploaded_voice_references=supports_uploaded_voice_references,
        allowed_local_media_path=allowed_local_media_path,
        allowed_media_domains=allowed_media_domains,
        voice_store=app.state.speaker_sample_store,
        tts_batch_max_items=tts_batch_max_items,
    )

    resolved_key = resolve_admin_api_key(admin_api_key)

    # Register all routes
    register_favicon(app)
    _register_health(app)
    _register_models(app)
    _register_admin(app, resolved_key)
    _register_chat_completions(app)
    _register_voices(app)
    _register_generate(app)
    _register_speech(app)
    _register_speech_batch(app)
    _register_speech_ws(app)
    _register_transcriptions(app)
    if enable_realtime:
        _register_realtime(app)

    return app


def _register_voices(app: FastAPI) -> None:
    @app.get("/v1/audio/voices")
    async def list_voices() -> JSONResponse:
        voice_store: SpeakerSampleStore = app.state.speaker_sample_store
        response = VoiceListResponse.model_validate(voice_store.list_response())
        return JSONResponse(content=response.model_dump(exclude_none=True))

    @app.post("/v1/audio/voices")
    async def upload_voice(
        audio_sample: UploadFile = File(...),
        consent: str = Form(...),
        name: str = Form(...),
        ref_text: str | None = Form(default=None),
        speaker_description: str | None = Form(default=None),
    ) -> JSONResponse:
        voice_store: SpeakerSampleStore = app.state.speaker_sample_store
        try:
            response = voice_store.upload(
                name=name,
                consent=consent,
                audio_bytes=await _read_voice_upload(audio_sample),
                filename=audio_sample.filename,
                content_type=audio_sample.content_type,
                ref_text=ref_text,
                speaker_description=speaker_description,
            )
        except SpeechAPIError as exc:
            return speech_error_response(exc)
        return JSONResponse(content=response)

    @app.delete("/v1/audio/voices/{name}")
    async def delete_voice(name: str) -> JSONResponse:
        voice_store: SpeakerSampleStore = app.state.speaker_sample_store
        try:
            deleted = voice_store.delete(name)
        except SpeechAPIError as exc:
            return speech_error_response(exc)
        if not deleted:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"Voice '{name}' not found"},
            )
        return JSONResponse(
            content={
                "success": True,
                "message": f"Voice '{name}' deleted successfully",
            }
        )


async def _read_voice_upload(audio_sample: UploadFile) -> bytes:
    audio_bytes = await audio_sample.read(MAX_VOICE_UPLOAD_BYTES + 1)
    if len(audio_bytes) > MAX_VOICE_UPLOAD_BYTES:
        raise bad_request(
            f"audio_sample must be at most {MAX_VOICE_UPLOAD_BYTES} bytes",
            param="audio_sample",
        )
    return audio_bytes


def _is_voice_upload_scope(scope: dict[str, Any]) -> bool:
    return (
        scope.get("type") == "http"
        and scope.get("method") == "POST"
        and scope.get("path") == "/v1/audio/voices"
    )


def _content_length(scope: dict[str, Any]) -> int | None:
    for name, value in scope.get("headers", ()):
        if name.lower() != b"content-length":
            continue
        try:
            return int(value.decode("ascii"))
        except ValueError:
            return None
    return None


async def _send_voice_upload_too_large(
    send: Callable[[dict[str, Any]], Awaitable[None]],
    max_bytes: int,
) -> None:
    body = json.dumps(
        openai_error_payload(
            f"request body must be at most {max_bytes} bytes",
            error_type="RequestTooLargeError",
            param="audio_sample",
            code=413,
        )
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _register_health(app: FastAPI) -> None:
    @app.get("/health")
    async def health() -> JSONResponse:
        """Health check endpoint (includes filesystem browse info)."""
        client: Client = app.state.client
        info = client.health()
        is_running = info.get("running", False)
        status_code = 200 if is_running else 503
        return JSONResponse(
            content={
                "status": "healthy" if is_running else "unhealthy",
                **info,
            },
            status_code=status_code,
        )


def _register_models(app: FastAPI) -> None:
    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        """List available models."""
        model_name: str = app.state.model_name
        model_list = ModelList(
            data=[
                ModelCard(
                    id=model_name,
                    root=model_name,
                    created=0,
                )
            ]
        )
        return JSONResponse(content=model_list.model_dump())


def _register_admin(app: FastAPI, admin_api_key: str | None = None) -> None:
    _auth = make_admin_auth_dependency(admin_api_key)

    @app.get("/model_info", dependencies=[Depends(_auth)])
    async def model_info_get() -> JSONResponse:
        client: Client = app.state.client
        return _model_info_response(await client.model_info())

    @app.post("/model_info", dependencies=[Depends(_auth)])
    async def model_info_post(req: AdminRequestBase) -> JSONResponse:
        client: Client = app.state.client
        return _model_info_response(
            await client.model_info(
                stages=req.stages,
                timeout_s=_timeout_or_default(req.timeout_s, 30.0),
            )
        )

    @app.post("/pause_generation", dependencies=[Depends(_auth)])
    async def pause_generation(req: PauseGenerationRequest) -> JSONResponse:
        client: Client = app.state.client
        payload = _request_payload(req)
        return _admin_response(
            await client.pause_generation(
                payload,
                stages=req.stages,
                timeout_s=_timeout_or_default(req.timeout_s, 60.0),
            )
        )

    @app.post("/continue_generation", dependencies=[Depends(_auth)])
    async def continue_generation(req: ContinueGenerationRequest) -> JSONResponse:
        client: Client = app.state.client
        payload = _request_payload(req)
        return _admin_response(
            await client.continue_generation(
                payload,
                stages=req.stages,
                timeout_s=_timeout_or_default(req.timeout_s, 60.0),
            )
        )

    @app.post("/update_weights_from_disk", dependencies=[Depends(_auth)])
    async def update_weights_from_disk(
        req: UpdateWeightFromDiskRequest,
    ) -> JSONResponse:
        client: Client = app.state.client
        payload = _request_payload(req)
        return _admin_response(
            await client.update_weights_from_disk(
                payload,
                stages=req.stages,
                timeout_s=_timeout_or_default(req.timeout_s, 120.0),
            )
        )

    @app.post("/update_weights_from_tensor", dependencies=[Depends(_auth)])
    async def update_weights_from_tensor(
        request: Request,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": (
                        "update_weights_from_tensor is not yet implemented. "
                        "Use update_weights_from_disk for the disk-based weight update path."
                    ),
                    "code": "not_implemented",
                }
            },
        )

    @app.post("/init_weights_update_group", dependencies=[Depends(_auth)])
    async def init_weights_update_group(
        req: InitWeightsUpdateGroupRequest,
    ) -> JSONResponse:
        client: Client = app.state.client
        payload = _request_payload(req)
        return _admin_response(
            await client.init_weights_update_group(
                payload,
                stages=req.stages,
                timeout_s=_timeout_or_default(req.timeout_s, 300.0),
            )
        )

    @app.post("/destroy_weights_update_group", dependencies=[Depends(_auth)])
    async def destroy_weights_update_group(
        req: DestroyWeightsUpdateGroupRequest,
    ) -> JSONResponse:
        client: Client = app.state.client
        payload = _request_payload(req)
        return _admin_response(
            await client.destroy_weights_update_group(
                payload,
                stages=req.stages,
                timeout_s=_timeout_or_default(req.timeout_s, 300.0),
            )
        )

    @app.post("/update_weights_from_distributed", dependencies=[Depends(_auth)])
    async def update_weights_from_distributed(
        req: UpdateWeightsFromDistributedRequest,
    ) -> JSONResponse:
        client: Client = app.state.client
        payload = _request_payload(req)
        return _admin_response(
            await client.update_weights_from_distributed(
                payload,
                stages=req.stages,
                timeout_s=_timeout_or_default(req.timeout_s, 300.0),
            )
        )

    @app.get("/weights_checker", dependencies=[Depends(_auth)])
    async def weights_checker_get(action: str = "checksum") -> JSONResponse:
        client: Client = app.state.client
        return _admin_response(await client.weights_checker({"action": action}))

    @app.post("/weights_checker", dependencies=[Depends(_auth)])
    async def weights_checker_post(req: WeightsCheckerRequest) -> JSONResponse:
        client: Client = app.state.client
        payload = _request_payload(req)
        return _admin_response(
            await client.weights_checker(
                payload,
                stages=req.stages,
                timeout_s=_timeout_or_default(req.timeout_s, 120.0),
            )
        )


def _timeout_or_default(timeout_s: float | None, default: float) -> float:
    return default if timeout_s is None else timeout_s


def _request_payload(req: AdminRequestBase) -> dict[str, Any]:
    return req.model_dump(exclude={"stages", "timeout_s"}, exclude_none=True)


def _admin_response(result: dict[str, Any]) -> JSONResponse:
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result)
    return JSONResponse(content=result)


def _model_info_response(result: dict[str, Any]) -> JSONResponse:
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result)

    stage_infos = _extract_model_info_stage_data(result)
    weight_version = _common_model_info_value(
        result,
        stage_infos,
        "weight_version",
        mixed_status_code=409,
    )
    payload = dict(result)
    payload.update(
        {
            "weight_version": weight_version,
            "model_path": _common_model_info_value(result, stage_infos, "model_path"),
            "load_format": _common_model_info_value(result, stage_infos, "load_format"),
            "stages": result.get("results", []),
        }
    )
    return JSONResponse(content=payload)


def _extract_model_info_stage_data(result: dict[str, Any]) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for item in result.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        data = item.get("data")
        if not isinstance(data, dict):
            continue
        if data.get("skipped") or data.get("unsupported"):
            continue
        stage_info = dict(data)
        stage_info.setdefault("stage", item.get("stage"))
        stage_info.setdefault("success", item.get("success"))
        infos.append(stage_info)
    return infos


def _common_model_info_value(
    result: dict[str, Any],
    stage_infos: list[dict[str, Any]],
    key: str,
    *,
    mixed_status_code: int | None = None,
) -> Any:
    values = [info[key] for info in stage_infos if info.get(key) is not None]
    if not values:
        return None

    unique: dict[str, Any] = {}
    for value in values:
        unique.setdefault(json.dumps(value, sort_keys=True, default=str), value)
    if len(unique) == 1:
        return next(iter(unique.values()))
    if mixed_status_code is not None:
        raise HTTPException(
            status_code=mixed_status_code,
            detail={
                "success": False,
                "message": f"mixed stage {key}",
                "mixed_state": {key: list(unique.values())},
                "stages": stage_infos,
                "admin": result,
            },
        )
    return None


def _register_chat_completions(app: FastAPI) -> None:
    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Response:
        client: Client = app.state.client
        default_model: str = app.state.model_name

        request_id = req.request_id or str(uuid.uuid4())
        response_id = f"chatcmpl-{request_id}"
        created = int(time.time())
        model = req.model or default_model

        gen_req = _build_chat_generate_request(req)

        # Determine audio format from request
        audio_format = "wav"
        if req.audio and isinstance(req.audio, dict):
            audio_format = req.audio.get("format", "wav")

        if req.stream:
            return StreamingResponse(
                _chat_stream(
                    client,
                    gen_req,
                    request_id,
                    response_id,
                    created,
                    model,
                    req,
                    audio_format,
                ),
                media_type="text/event-stream",
            )

        return await _chat_non_stream(
            client,
            gen_req,
            request_id,
            response_id,
            created,
            model,
            req,
            audio_format,
        )


async def _chat_non_stream(
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    response_id: str,
    created: int,
    model: str,
    req: ChatCompletionRequest,
    audio_format: str,
) -> JSONResponse:
    """Handle non-streaming chat completions."""
    try:
        result = await client.completion(
            gen_req,
            request_id=request_id,
            audio_format=audio_format,
        )
    except ClientError as exc:
        if _is_bad_request_error(exc):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error generating response for request %s", request_id)
        if _is_bad_request_error(exc):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    requested_modalities = req.modalities or ["text"]

    # Build message content
    message: dict[str, Any] = {"role": "assistant"}

    if "text" in requested_modalities and result.text:
        message["content"] = result.text

    if "audio" in requested_modalities and result.audio is not None:
        message["audio"] = {
            "id": result.audio.id,
            "data": result.audio.data,
            "transcript": result.audio.transcript,
        }

    if "content" not in message and "audio" not in message:
        message["content"] = result.text

    # Build usage
    usage = None
    if result.usage is not None:
        usage = UsageResponse(
            prompt_tokens=result.usage.prompt_tokens or 0,
            completion_tokens=result.usage.completion_tokens or 0,
            total_tokens=result.usage.total_tokens or 0,
        )

    response = ChatCompletionResponse(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason=result.finish_reason,
            )
        ],
        usage=usage,
    )

    return JSONResponse(content=response.model_dump())


async def _chat_stream(
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    response_id: str,
    created: int,
    model: str,
    req: ChatCompletionRequest,
    audio_format: str,
):
    """Streaming chat completion generator (yields SSE events)."""
    role_sent = False
    requested_modalities = req.modalities or ["text"]
    finish_reason: str | None = None
    final_usage: UsageResponse | None = None

    async for chunk in client.completion_stream(
        gen_req,
        request_id=request_id,
        audio_format=audio_format,
    ):
        # Capture finish info for the dedicated finish chunk after the loop.
        # Some pipelines only emit a final aggregate chunk; do not drop its
        # text/audio just because it already carries a finish reason.
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason
            if chunk.usage is not None:
                final_usage = UsageResponse(
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
                )
            has_payload = (
                chunk.modality == "text"
                and bool(chunk.text)
                and "text" in requested_modalities
            ) or (
                chunk.modality == "audio"
                and chunk.audio_b64 is not None
                and "audio" in requested_modalities
            )
            if not has_payload:
                continue

        delta = ChatCompletionStreamDelta()
        emit = False

        # Send role on first chunk
        if not role_sent:
            delta.role = "assistant"
            role_sent = True
            emit = True

        # Text chunk
        if chunk.modality == "text" and chunk.text and "text" in requested_modalities:
            delta.content = chunk.text
            emit = True

        # Audio chunk
        if (
            chunk.modality == "audio"
            and chunk.audio_b64 is not None
            and "audio" in requested_modalities
        ):
            delta.audio = ChatCompletionAudio(
                id=f"audio-{request_id}",
                data=chunk.audio_b64,
            )
            emit = True

        if not emit:
            continue

        stream_resp = ChatCompletionStreamResponse(
            id=response_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=delta,
                    finish_reason=None,
                )
            ],
        )

        data = stream_resp.model_dump(exclude_none=True)
        for choice in data.get("choices", []):
            choice.setdefault("finish_reason", None)
        yield f"data: {json.dumps(data)}\n\n"

    # Finish chunk: empty delta + finish_reason.
    finish_resp = ChatCompletionStreamResponse(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=ChatCompletionStreamDelta(),
                finish_reason=finish_reason or "stop",
            )
        ],
        usage=final_usage,
    )
    data = finish_resp.model_dump(exclude_none=True)
    for choice in data.get("choices", []):
        choice.setdefault("finish_reason", None)
    yield f"data: {json.dumps(data)}\n\n"

    yield f"data: {STREAM_DONE_SENTINEL}\n\n"


def _build_chat_generate_request(req: ChatCompletionRequest) -> GenerateRequest:
    """Convert a ChatCompletionRequest into a client GenerateRequest."""
    # Parse stop sequences
    stop: list[str] = []
    if isinstance(req.stop, str):
        stop = [req.stop]
    elif isinstance(req.stop, list):
        stop = list(req.stop)

    # Build sampling params
    sampling = SamplingParams(
        temperature=req.temperature if req.temperature is not None else 1.0,
        top_p=req.top_p if req.top_p is not None else 1.0,
        top_k=req.top_k if req.top_k is not None else -1,
        min_p=req.min_p if req.min_p is not None else 0.0,
        repetition_penalty=(
            req.repetition_penalty if req.repetition_penalty is not None else 1.0
        ),
        stop=stop,
        seed=req.seed,
        max_new_tokens=req.effective_max_tokens,
    )

    # Convert messages
    messages = [Message(role=m.role, content=m.content) for m in req.messages]

    # Determine output modalities
    output_modalities = req.modalities or ["text"]  # e.g. ["text", "audio"]

    # Build per-stage sampling overrides
    stage_sampling: dict[str, SamplingParams] | None = None
    if req.stage_sampling:
        stage_sampling = {}
        for stage_name, params_dict in req.stage_sampling.items():
            stage_sampling[stage_name] = SamplingParams(**params_dict)

    # Extract audios, images, and videos from request
    audios: list[str] | None = None
    if req.audios:
        audios = req.audios

    images: list[str] | None = None
    if req.images:
        images = req.images

    videos: list[str] | None = None
    if req.videos:
        videos = req.videos

    # Merge audio config, audios, images, and videos into metadata
    metadata: dict[str, Any] = {}
    if req.audio:
        metadata["audio_config"] = req.audio
    if audios:
        metadata["audios"] = audios
    if images:
        metadata["images"] = images
    if videos:
        metadata["videos"] = videos
    if req.video_fps is not None:
        metadata["video_fps"] = req.video_fps
    if req.video_max_frames is not None:
        metadata["video_max_frames"] = req.video_max_frames
    if req.video_min_pixels is not None:
        metadata["video_min_pixels"] = req.video_min_pixels
    if req.video_max_pixels is not None:
        metadata["video_max_pixels"] = req.video_max_pixels
    if req.video_total_pixels is not None:
        metadata["video_total_pixels"] = req.video_total_pixels

    extra_params: dict[str, Any] = {}
    for field_name, value in (
        ("talker_temperature", req.talker_temperature),
        ("talker_top_p", req.talker_top_p),
        ("talker_top_k", req.talker_top_k),
        ("talker_repetition_penalty", req.talker_repetition_penalty),
        ("talker_max_new_tokens", req.talker_max_new_tokens),
    ):
        if value is not None:
            extra_params[field_name] = value

    return GenerateRequest(
        model=req.model,
        messages=messages,
        sampling=sampling,
        stage_sampling=stage_sampling,
        stage_params=req.stage_params,
        extra_params=extra_params,
        stream=req.stream,
        max_tokens=req.effective_max_tokens,
        output_modalities=output_modalities,
        metadata=metadata,
    )


def _register_generate(app: FastAPI) -> None:
    @app.post("/generate")
    async def generate(req: RolloutGenerateRequest) -> Response:
        client: Client = app.state.client

        provided = [
            value is not None for value in (req.input_ids, req.prompt, req.messages)
        ]
        if sum(provided) != 1:
            raise HTTPException(
                status_code=400,
                detail="exactly one of input_ids, prompt, or messages is required",
            )
        if req.stream:
            raise HTTPException(
                status_code=400,
                detail="stream=true is not supported by /generate yet",
            )

        request_id = str(uuid.uuid4())
        audio_format = "wav"

        try:
            gen_req = _build_rollout_generate_request(req)
            result = await client.completion(
                gen_req,
                request_id=request_id,
                audio_format=audio_format,
            )
        except ClientError as exc:
            if _is_bad_request_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Error generating rollout for request %s", request_id)
            if _is_bad_request_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        response = _build_generate_response(req, result, audio_format)
        return JSONResponse(content=response.model_dump())


def _rollout_sampling_to_client(params: RolloutSamplingParams) -> SamplingParams:
    kwargs: dict[str, Any] = {
        key: value
        for key, value in (
            ("temperature", params.temperature),
            ("top_p", params.top_p),
            ("top_k", params.top_k),
            ("min_p", params.min_p),
            ("repetition_penalty", params.repetition_penalty),
            ("stop_token_ids", params.stop_token_ids),
            ("seed", params.seed),
            ("max_new_tokens", params.max_new_tokens),
        )
        if value is not None
    }
    if params.stop is not None:
        kwargs["stop"] = (
            [params.stop] if isinstance(params.stop, str) else list(params.stop)
        )
    if "max_new_tokens" not in kwargs and params.max_tokens is not None:
        kwargs["max_new_tokens"] = params.max_tokens
    return SamplingParams(**kwargs)


def _build_rollout_generate_request(req: RolloutGenerateRequest) -> GenerateRequest:
    """Convert a rollout GenerateRequest into a client GenerateRequest."""
    sampling = _rollout_sampling_to_client(req.sampling_params)

    messages: list[Message] | None = None
    if req.messages is not None:
        messages = [Message(role=m.role, content=m.content) for m in req.messages]

    stage_sampling: dict[str, SamplingParams] | None = None
    if req.stage_sampling:
        stage_sampling = {
            name: _rollout_sampling_to_client(params)
            for name, params in req.stage_sampling.items()
        }

    extra_params: dict[str, Any] = {
        "return_logprob": req.return_logprob,
        "return_omni_rollout": req.return_omni_rollout,
        "return_routed_experts": req.return_routed_experts,
        "return_indexer_topk": req.return_indexer_topk,
    }

    return GenerateRequest(
        model=req.model,
        prompt=req.prompt,
        prompt_token_ids=req.input_ids,
        messages=messages,
        sampling=sampling,
        stage_sampling=stage_sampling,
        stage_params=req.stage_params,
        extra_params=extra_params,
        stream=req.stream,
        max_tokens=sampling.max_new_tokens,
        output_modalities=(
            req.output_modalities if req.output_modalities is not None else ["text"]
        ),
        metadata=dict(req.metadata) if req.metadata else {},
    )


def _build_generate_response(
    req: RolloutGenerateRequest,
    result: CompletionResult,
    audio_format: str,
) -> GenerateResponse:
    usage = result.usage
    completion_tokens = (
        usage.completion_tokens
        if usage is not None and usage.completion_tokens is not None
        else 0
    )
    prompt_tokens = (
        usage.prompt_tokens
        if usage is not None and usage.prompt_tokens is not None
        else 0
    )

    finish_type = result.finish_reason or "stop"
    finish_reason = GenerateFinishReason(
        type=finish_type,
        length=completion_tokens if finish_type == "length" else None,
    )
    # (Jingwen): Validate that return_logprob has a backend carrier: text token
    # logprobs or omni_rollout action logprobs, depending on modality.
    if result.omni_rollout is None and (
        req.return_omni_rollout
        or (req.return_logprob and result.output_token_logprobs is None)
    ):
        raise HTTPException(
            status_code=501,
            detail=(
                "backend did not return requested logprobs; expected "
                "output_token_logprobs for text or meta_info.omni_rollout for "
                "audio (set return_omni_rollout=true for audio logprobs)"
            ),
        )
    if (
        req.return_logprob
        and result.output_token_logprobs is not None
        and len(result.output_token_logprobs) != completion_tokens
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "backend returned output_token_logprobs length "
                f"{len(result.output_token_logprobs)} for "
                f"completion_tokens={completion_tokens}"
            ),
        )
    audio: GenerateAudio | None = None
    if result.audio is not None:
        audio = GenerateAudio(data=result.audio.data, format=audio_format)

    meta_info = GenerateMetaInfo(
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        weight_version=result.weight_version,
        request_metadata=req.metadata,
        output_token_logprobs=(
            result.output_token_logprobs if req.return_logprob else None
        ),
        omni_rollout=result.omni_rollout if req.return_omni_rollout else None,
    )
    return GenerateResponse(text=result.text, audio=audio, meta_info=meta_info)


def _register_realtime(app: FastAPI) -> None:
    """Mount the OpenAI-compatible WebSocket Realtime endpoint."""
    from sglang_omni.serve.realtime import RealtimeSessionManager

    client: Client = app.state.client
    model_name: str = app.state.model_name
    manager = RealtimeSessionManager(client=client, model_name=model_name)
    app.state.realtime_manager = manager

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        session = manager.open(websocket)
        try:
            await session.run()
        finally:
            await manager.close(session.session_id)


def _register_speech(app: FastAPI) -> None:
    @app.post("/v1/audio/speech")
    async def create_speech(request: Request) -> Response:
        client: Client = app.state.client
        speech_service: SpeechRequestValidator = app.state.speech_service

        request_id = f"speech-{uuid.uuid4()}"
        try:
            payload = await request.json()
            prepared = await asyncio.to_thread(
                speech_service.parse_generation_request, payload
            )
            req = prepared.request
            gen_req = speech_service.build_generate_request(
                req,
                validate=False,
                reference_descriptors=prepared.reference_descriptors,
                uploaded_voice=prepared.uploaded_voice,
            )
        except json.JSONDecodeError:
            return speech_error_response(
                bad_request("speech request body must be valid JSON")
            )
        except SpeechAPIError as exc:
            return speech_error_response(exc)

        if req.stream:
            try:
                return await _speech_audio_response(
                    request=request,
                    client=client,
                    gen_req=gen_req,
                    request_id=request_id,
                    speed=req.speed,
                )
            except ClientError as exc:
                return speech_error_response(internal_error(str(exc)))
            except Exception as exc:
                logger.exception(
                    "Error preparing raw PCM speech stream for request %s",
                    request_id,
                )
                return speech_error_response(internal_error(str(exc)))

        try:
            result = await _await_speech_response(
                request=request,
                client=client,
                gen_req=gen_req,
                request_id=request_id,
                response_format=req.response_format,
                speed=req.speed,
            )
        except ClientError as exc:
            return speech_error_response(internal_error(str(exc)))
        except Exception as exc:
            logger.exception("Error generating speech for request %s", request_id)
            return speech_error_response(internal_error(str(exc)))

        headers = {
            "Content-Disposition": f'attachment; filename="speech.{result.format}"',
        }
        if result.usage is not None:
            if result.usage.prompt_tokens is not None:
                headers["X-Prompt-Tokens"] = str(result.usage.prompt_tokens)
            if result.usage.completion_tokens is not None:
                headers["X-Completion-Tokens"] = str(result.usage.completion_tokens)
            if result.usage.engine_time_s is not None:
                headers["X-Engine-Time"] = str(result.usage.engine_time_s)

        return Response(
            content=result.audio_bytes,
            media_type=result.mime_type,
            headers=headers,
        )


def _register_speech_batch(app: FastAPI) -> None:
    @app.post("/v1/audio/speech/batch")
    async def create_speech_batch(request: Request) -> JSONResponse:
        client: Client = app.state.client
        speech_service: SpeechRequestValidator = app.state.speech_service
        request_id = f"speech-batch-{uuid.uuid4()}"
        try:
            payload = await request.json()
            batch = await asyncio.to_thread(speech_service.parse_batch_request, payload)
            response = await _create_speech_batch_with_disconnect_watch(
                request,
                client=client,
                speech_service=speech_service,
                batch=batch,
                request_id=request_id,
            )
        except json.JSONDecodeError:
            return speech_error_response(
                bad_request("speech batch request body must be valid JSON")
            )
        except SpeechAPIError as exc:
            return speech_error_response(exc)
        except Exception as exc:
            logger.exception("Error generating speech batch for request %s", request_id)
            return speech_error_response(internal_error(str(exc)))

        response = SpeechBatchResponse.model_validate(response)
        return JSONResponse(content=response.model_dump(exclude_none=True))


async def _create_speech_batch_with_disconnect_watch(
    request: Request,
    *,
    client: Client,
    speech_service: SpeechRequestValidator,
    batch: CreateSpeechBatchRequest,
    request_id: str,
) -> Any:
    batch_task = asyncio.create_task(
        speech_service.create_speech_batch(
            client,
            batch,
            request_id=request_id,
        )
    )
    disconnect_task = asyncio.create_task(_wait_for_request_disconnect(request))
    try:
        done, _ = await asyncio.wait(
            {batch_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if batch_task in done:
            return await batch_task

        batch_task.cancel()
        with suppress(asyncio.CancelledError):
            await batch_task
        raise asyncio.CancelledError
    finally:
        if not disconnect_task.done():
            await _cancel_task_bounded(disconnect_task)


def _register_speech_ws(app: FastAPI) -> None:
    @app.websocket("/v1/audio/speech/stream")
    async def speech_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        session = SpeechWebSocketSession(
            websocket,
            client=app.state.client,
            speech_service=app.state.speech_service,
        )
        await session.run()


def _speech_pcm_chunk_bytes(
    chunk: Any,
    *,
    emitted_samples: int,
    speed: float,
) -> tuple[bytes | None, int, int]:
    sample_rate = chunk.sample_rate or DEFAULT_SAMPLE_RATE
    audio_data, emitted_samples = select_audio_delta(
        chunk.audio_data,
        emitted_samples=emitted_samples,
        is_terminal=chunk.finish_reason is not None,
    )
    if audio_data is None:
        return None, emitted_samples, sample_rate

    if speed != 1.0:
        audio_data, sample_rate = apply_speed(audio_data, speed, sample_rate)
    audio_bytes = encode_pcm(audio_data, sample_rate)
    if not audio_bytes:
        return None, emitted_samples, sample_rate
    return audio_bytes, emitted_samples, sample_rate


async def _speech_audio_response(
    request: Request,
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    speed: float,
) -> StreamingResponse:
    """Build a raw PCM stream after deriving headers from the first audio chunk."""
    emitted_samples = 0
    chunk_stream = client.generate(gen_req, request_id=request_id)
    first_audio_bytes: bytes | None = None
    stream_sample_rate: int | None = None
    stream_completed = False
    stream_closed = False
    disconnect_task = asyncio.create_task(_wait_for_request_disconnect(request))
    next_chunk_task: asyncio.Task[Any] | None = None

    try:
        while True:
            next_chunk_task = asyncio.create_task(anext(chunk_stream))
            done, _ = await asyncio.wait(
                {next_chunk_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                if not next_chunk_task.done():
                    await _cancel_task_bounded(next_chunk_task)
                await _abort_and_close_speech_stream(client, request_id, chunk_stream)
                stream_closed = True
                raise asyncio.CancelledError

            try:
                chunk = next_chunk_task.result()
            except StopAsyncIteration:
                stream_completed = True
                break
            if chunk.audio_data is None:
                continue

            first_audio_bytes, emitted_samples, stream_sample_rate = (
                _speech_pcm_chunk_bytes(
                    chunk,
                    emitted_samples=emitted_samples,
                    speed=speed,
                )
            )
            if first_audio_bytes is not None:
                break

        if first_audio_bytes is None or stream_sample_rate is None:
            raise RuntimeError("No audio output generated from the pipeline.")
    except asyncio.CancelledError:
        if not stream_closed:
            await _abort_and_close_speech_stream(client, request_id, chunk_stream)
        raise
    except Exception:
        if not stream_completed:
            await _abort_and_close_speech_stream(client, request_id, chunk_stream)
        else:
            await _close_async_iterator_if_supported(chunk_stream)
        raise
    finally:
        if next_chunk_task is not None and not next_chunk_task.done():
            await _cancel_task_bounded(next_chunk_task)
        if not disconnect_task.done():
            await _cancel_task_bounded(disconnect_task)

    async def _body():
        nonlocal emitted_samples
        active_request = True
        try:
            yield first_audio_bytes

            async for chunk in chunk_stream:
                if chunk.audio_data is None:
                    continue

                audio_bytes, emitted_samples, sample_rate = _speech_pcm_chunk_bytes(
                    chunk,
                    emitted_samples=emitted_samples,
                    speed=speed,
                )
                if audio_bytes is None:
                    continue
                if sample_rate != stream_sample_rate:
                    raise RuntimeError(
                        "Raw PCM speech stream sample rate changed from "
                        f"{stream_sample_rate} to {sample_rate}"
                    )
                yield audio_bytes
            active_request = False
        finally:
            if active_request:
                await _abort_and_close_speech_stream(client, request_id, chunk_stream)
            else:
                await _close_async_iterator_if_supported(chunk_stream)

    return StreamingResponse(
        _body(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(stream_sample_rate),
            "X-Channels": "1",
            "X-Bit-Depth": "16",
        },
    )


async def _await_speech_response(
    request: Request,
    client: Client,
    gen_req: GenerateRequest,
    *,
    request_id: str,
    response_format: str,
    speed: float,
):
    speech_task = asyncio.create_task(
        client.speech(
            gen_req,
            request_id=request_id,
            response_format=response_format,
            speed=speed,
            allow_format_fallback=False,
        )
    )
    disconnect_task = asyncio.create_task(_wait_for_request_disconnect(request))
    aborted = False
    try:
        done, _ = await asyncio.wait(
            {speech_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if speech_task in done:
            return speech_task.result()

        await client.abort(request_id)
        aborted = True
        speech_task.cancel()
        raise asyncio.CancelledError
    except asyncio.CancelledError:
        if not aborted:
            await client.abort(request_id)
        raise
    finally:
        if not speech_task.done():
            await _cancel_task_bounded(speech_task)
        if not disconnect_task.done():
            await _cancel_task_bounded(disconnect_task)


async def _cancel_task_bounded(task: asyncio.Task[Any]) -> None:
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=HTTP_DISCONNECT_CANCEL_TIMEOUT_S)
    if done:
        await asyncio.gather(*done, return_exceptions=True)
    else:
        task.add_done_callback(_discard_cancelled_task_result)


def _discard_cancelled_task_result(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("Cancelled request task finished with an error", exc_info=True)


async def _wait_for_request_disconnect(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(HTTP_DISCONNECT_POLL_INTERVAL_S)


async def _close_async_iterator_if_supported(stream: AsyncIterator[Any]) -> None:
    try:
        close = stream.aclose
    except AttributeError:
        return
    await close()


async def _abort_and_close_speech_stream(
    client: Client,
    request_id: str,
    stream: AsyncIterator[Any],
) -> None:
    try:
        await client.abort(request_id)
    finally:
        await _close_async_iterator_if_supported(stream)


def _register_transcriptions(app: FastAPI) -> None:
    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str | None = Form(default=None),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        response_format: str = Form(default="json"),
        temperature: float | None = Form(default=None),
    ) -> Response:
        client: Client = app.state.client
        default_model: str = app.state.model_name
        request_id = f"transcription-{uuid.uuid4()}"

        # TODO(Ratish): add the same pre-parser body limit used by voice uploads
        # once transcription upload limits are defined.
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

        gen_req = build_transcription_generate_request(
            audio_bytes=audio_bytes,
            filename=file.filename,
            content_type=file.content_type,
            model=model or default_model,
            language=language,
            prompt=prompt,
            temperature=temperature,
        )

        try:
            result = await client.completion(gen_req, request_id=request_id)
        except ClientError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Error transcribing audio for request %s", request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        text = result.text
        normalized_response_format = response_format.strip().lower()
        if normalized_response_format == "text":
            return PlainTextResponse(text)
        if normalized_response_format not in {"json", "verbose_json"}:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported response_format for /v1/audio/transcriptions: "
                    f"{response_format!r}"
                ),
            )

        adapter = resolve_adapter(getattr(app.state, "architectures", None))
        text = adapter.postprocess_text(text)
        duration_s = _probe_audio_duration(audio_bytes)
        usage = (
            TranscriptionUsage(seconds=math.ceil(duration_s))
            if duration_s > 0
            else None
        )
        if normalized_response_format == "verbose_json":
            response = adapter.build_verbose_response(
                text=text,
                language=language,
                audio_duration_s=duration_s,
            )
            response.usage = usage
            return JSONResponse(content=response.model_dump(exclude_none=True))
        return JSONResponse(
            content=TranscriptionResponse(text=text, usage=usage).model_dump(
                exclude_none=True
            )
        )


def _probe_audio_duration(audio_bytes: bytes) -> float:
    """Best-effort audio duration (seconds) from raw upload bytes.

    Uses ``soundfile.info`` (metadata only, no full decode; torchaudio removed
    its ``info`` API in 2.x). Returns 0.0 if the duration cannot be
    determined; callers treat 0.0 as "unknown".
    """
    try:
        import soundfile as sf

        info = sf.info(io.BytesIO(audio_bytes))
        if info.samplerate:
            return max(info.frames / float(info.samplerate), 0.0)
    except (RuntimeError, ValueError):
        logger.debug("Could not probe audio duration", exc_info=True)
    return 0.0


def build_transcription_generate_request(
    *,
    audio_bytes: bytes,
    filename: str | None,
    content_type: str | None,
    model: str,
    language: str | None,
    prompt: str | None,
    temperature: float | None,
) -> GenerateRequest:
    params: dict[str, Any] = {"task": "transcribe"}
    if language is not None:
        params["language"] = language
    if prompt is not None:
        params["prompt"] = prompt
    # note (aaron): the client layer fills in the SamplingParams default of 1.0
    # when this key is absent. 0.0 (greedy) matches the OpenAI API, see issue #959.
    params["temperature"] = temperature if temperature is not None else 0.0

    return GenerateRequest(
        model=model,
        prompt={
            "audio_bytes": audio_bytes,
            "filename": filename,
            "content_type": content_type,
        },
        extra_params=params,
        stream=False,
        output_modalities=["text"],
        metadata={"task": "asr"},
    )
