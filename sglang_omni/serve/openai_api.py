# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible API server for sglang-omni.

Provides the following endpoints:
- POST /v1/chat/completions  — Text (+ audio) chat completions
- POST /v1/audio/speech      — Text-to-speech synthesis
- GET  /v1/models            — List available models
- GET  /health               — Health check
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang_omni.gateway import Gateway, GenerateRequest, Message, SamplingParams
from sglang_omni.serve.audio_utils import audio_to_base64, encode_audio
from sglang_omni.serve.protocol import (
    ChatCompletionAudio,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatCompletionStreamResponse,
    CreateSpeechRequest,
    ModelCard,
    ModelList,
    UsageResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    gateway: Gateway,
    *,
    model_name: str | None = None,
) -> FastAPI:
    """Create a FastAPI application with OpenAI-compatible endpoints.

    Args:
        gateway: Gateway instance connected to the pipeline coordinator.
        model_name: Default model name to report in responses and /v1/models.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="sglang-omni", version="0.1.0")

    # Store references in app state for access from route handlers
    app.state.gateway = gateway
    app.state.model_name = model_name or "sglang-omni"

    # Register all routes
    _register_health(app)
    _register_models(app)
    _register_chat_completions(app)
    _register_speech(app)

    return app


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def _register_health(app: FastAPI) -> None:
    @app.get("/health")
    async def health() -> JSONResponse:
        """Health check endpoint."""
        gateway: Gateway = app.state.gateway
        info = gateway.health()
        is_running = info.get("running", False)
        status_code = 200 if is_running else 503
        return JSONResponse(
            content={"status": "healthy" if is_running else "unhealthy", **info},
            status_code=status_code,
        )


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------


def _register_chat_completions(app: FastAPI) -> None:
    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Response:
        gateway: Gateway = app.state.gateway
        default_model: str = app.state.model_name

        request_id = req.request_id or str(uuid.uuid4())
        response_id = f"chatcmpl-{request_id}"
        created = int(time.time())
        model = req.model or default_model

        gen_req = _build_chat_generate_request(req)

        if req.stream:
            return StreamingResponse(
                _chat_stream(
                    gateway, gen_req, request_id, response_id, created, model, req
                ),
                media_type="text/event-stream",
            )

        return await _chat_non_stream(
            gateway, gen_req, request_id, response_id, created, model, req
        )


async def _chat_non_stream(
    gateway: Gateway,
    gen_req: GenerateRequest,
    request_id: str,
    response_id: str,
    created: int,
    model: str,
    req: ChatCompletionRequest,
) -> JSONResponse:
    """Handle non-streaming chat completions."""
    text_parts: list[str] = []
    audio_chunks: list[Any] = []
    last_chunk = None
    finish_reason: str | None = None

    async for chunk in gateway.generate(gen_req, request_id=request_id):
        last_chunk = chunk
        if chunk.modality == "text" and chunk.text:
            text_parts.append(chunk.text)
        if chunk.modality == "audio" and chunk.audio_data is not None:
            audio_chunks.append(chunk.audio_data)
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason

    if last_chunk is None:
        raise HTTPException(status_code=500, detail="No response from pipeline")

    # Build message content
    message: dict[str, Any] = {"role": "assistant"}
    full_text = "".join(text_parts)

    requested_modalities = req.modalities or ["text"]

    if "text" in requested_modalities and full_text:
        message["content"] = full_text

    if "audio" in requested_modalities and audio_chunks:
        # Determine audio format from request
        audio_format = "wav"
        if req.audio and isinstance(req.audio, dict):
            audio_format = req.audio.get("format", "wav")

        # Combine all audio data and encode to base64
        combined_audio = audio_chunks[-1]  # Use last (most complete) audio chunk
        audio_b64 = audio_to_base64(
            combined_audio,
            output_format=audio_format,
        )

        message["audio"] = {
            "id": f"audio-{request_id}",
            "data": audio_b64,
            "expires_at": created + 3600,
            "transcript": full_text if full_text else None,
        }

    if "content" not in message and "audio" not in message:
        # Fallback: always include content
        message["content"] = full_text

    # Build usage
    usage = None
    if last_chunk.usage is not None:
        usage = UsageResponse(
            prompt_tokens=last_chunk.usage.prompt_tokens or 0,
            completion_tokens=last_chunk.usage.completion_tokens or 0,
            total_tokens=last_chunk.usage.total_tokens or 0,
        )

    response = ChatCompletionResponse(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason=finish_reason or "stop",
            )
        ],
        usage=usage,
    )

    return JSONResponse(content=response.model_dump())


async def _chat_stream(
    gateway: Gateway,
    gen_req: GenerateRequest,
    request_id: str,
    response_id: str,
    created: int,
    model: str,
    req: ChatCompletionRequest,
):
    """Streaming chat completion generator (yields SSE events)."""
    role_sent = False
    requested_modalities = req.modalities or ["text"]

    # Determine audio format
    audio_format = "wav"
    if req.audio and isinstance(req.audio, dict):
        audio_format = req.audio.get("format", "wav")

    async for chunk in gateway.generate(gen_req, request_id=request_id):
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
            and chunk.audio_data is not None
            and "audio" in requested_modalities
        ):
            audio_b64 = audio_to_base64(
                chunk.audio_data,
                output_format=audio_format,
            )
            delta.audio = ChatCompletionAudio(
                id=f"audio-{request_id}",
                data=audio_b64,
            )
            emit = True

        # Finish reason
        finish_reason = chunk.finish_reason

        if not emit and finish_reason is None:
            continue

        # Build usage for final chunk
        usage = None
        if finish_reason is not None and chunk.usage is not None:
            usage = UsageResponse(
                prompt_tokens=chunk.usage.prompt_tokens or 0,
                completion_tokens=chunk.usage.completion_tokens or 0,
                total_tokens=chunk.usage.total_tokens or 0,
            )

        stream_resp = ChatCompletionStreamResponse(
            id=response_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

        yield f"data: {json.dumps(stream_resp.model_dump(exclude_none=True))}\n\n"

    yield "data: [DONE]\n\n"


def _build_chat_generate_request(req: ChatCompletionRequest) -> GenerateRequest:
    """Convert a ChatCompletionRequest into a gateway GenerateRequest."""
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
    output_modalities = req.modalities  # e.g. ["text", "audio"]

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

    return GenerateRequest(
        model=req.model,
        messages=messages,
        sampling=sampling,
        stage_sampling=stage_sampling,
        stage_params=req.stage_params,
        stream=req.stream,
        max_tokens=req.effective_max_tokens,
        output_modalities=output_modalities,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# POST /v1/audio/speech
# ---------------------------------------------------------------------------


def _register_speech(app: FastAPI) -> None:
    @app.post("/v1/audio/speech")
    async def create_speech(req: CreateSpeechRequest) -> Response:
        gateway: Gateway = app.state.gateway
        default_model: str = app.state.model_name

        request_id = f"speech-{uuid.uuid4()}"

        gen_req = _build_speech_generate_request(req, default_model)

        # Collect the final audio output
        audio_data: Any = None
        last_chunk = None

        async for chunk in gateway.generate(gen_req, request_id=request_id):
            last_chunk = chunk
            # Collect audio data from any modality
            if chunk.audio_data is not None:
                audio_data = chunk.audio_data
            elif chunk.modality == "audio" and chunk.text:
                # Some pipelines might return audio as encoded text
                audio_data = chunk.text

        if audio_data is None:
            raise HTTPException(
                status_code=500,
                detail="No audio output generated from the pipeline.",
            )

        # Encode to requested format
        audio_bytes, mime_type = encode_audio(
            audio_data,
            response_format=req.response_format,
            speed=req.speed,
        )

        return Response(
            content=audio_bytes,
            media_type=mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{req.response_format}"',
            },
        )


def _build_speech_generate_request(
    req: CreateSpeechRequest,
    default_model: str,
) -> GenerateRequest:
    """Convert a CreateSpeechRequest into a gateway GenerateRequest."""

    # Build TTS-specific parameters to pass through the pipeline
    tts_params: dict[str, Any] = {
        "voice": req.voice,
        "response_format": req.response_format,
        "speed": req.speed,
    }
    if req.task_type is not None:
        tts_params["task_type"] = req.task_type
    if req.language is not None:
        tts_params["language"] = req.language
    if req.instructions is not None:
        tts_params["instructions"] = req.instructions
    if req.ref_audio is not None:
        tts_params["ref_audio"] = req.ref_audio
    if req.ref_text is not None:
        tts_params["ref_text"] = req.ref_text
    if req.seed is not None:
        tts_params["seed"] = req.seed

    # Sampling params
    sampling = SamplingParams()
    if req.max_new_tokens is not None:
        sampling.max_new_tokens = req.max_new_tokens

    return GenerateRequest(
        model=req.model or default_model,
        prompt=req.input,
        sampling=sampling,
        stage_params=req.stage_params,
        stream=False,  # TTS returns complete audio, no streaming
        output_modalities=["audio"],
        metadata={
            "task": "tts",
            "tts_params": tts_params,
        },
    )
