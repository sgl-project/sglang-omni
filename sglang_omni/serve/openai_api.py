# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible API server for sglang-omni.

Provides the following endpoints:
- POST /v1/chat/completions  — Text (+ audio) chat completions
- POST /v1/audio/speech      — Text-to-speech synthesis
- GET  /v1/models            — List available models
- GET  /v1/fs/list           — Browse filesystem directories
- GET  /v1/fs/file           — Download a file
- GET  /health               — Health check
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang_omni.client import (
    Client,
    ClientError,
    GenerateRequest,
    Message,
    SamplingParams,
)
from sglang_omni.client.audio import (
    DEFAULT_SAMPLE_RATE,
    FORMAT_MIME_TYPES,
    encode_audio,
    to_numpy,
)
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
MIME_TO_FORMAT = {mime: fmt for fmt, mime in FORMAT_MIME_TYPES.items()}
_BAD_REQUEST_MARKERS = (
    "longer than the model's context length",
    "Requested token count exceeds the model's maximum context length",
)


def _is_bad_request_error(exc: Exception) -> bool:
    # TODO (Qiujiang): replace with structured error code.
    # Worker → coordinator currently serializes exceptions to str, so
    # 400 vs 500 must be recovered via phrase match. See Ccyest's proposal
    # on #330 for the end-to-end design (CompleteMessage.error_code).
    # These markers must stay in sync with SGLang's ValueError wording:
    #   - managers/tokenizer_manager.py:761, 791
    message = str(exc)
    return any(marker in message for marker in _BAD_REQUEST_MARKERS)


def create_app(
    client: Client,
    *,
    model_name: str | None = None,
) -> FastAPI:
    """Create a FastAPI application with OpenAI-compatible endpoints.

    Args:
        client: Client instance connected to the pipeline coordinator.
        model_name: Default model name to report in responses and /v1/models.
        serve_playground: Path to the playground directory to serve as static
            files.  When set, the filesystem browser API and static file
            serving are enabled so the entire playground runs on a single port.

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

    # Store references in app state for access from route handlers
    app.state.client = client
    app.state.model_name = model_name or "sglang-omni"

    # Register all routes
    _register_health(app)
    _register_models(app)
    _register_chat_completions(app)
    _register_speech(app)

    return app


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
            # TODO (Qiujiang): Align streaming bad-request behavior with upstream SGLang.
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
    except RuntimeError as exc:
        if _is_bad_request_error(exc):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ClientError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error generating response for request %s", request_id)
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
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason
            if chunk.usage is not None:
                final_usage = UsageResponse(
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
                )
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

    yield "data: [DONE]\n\n"


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


def _register_speech(app: FastAPI) -> None:
    @app.post("/v1/audio/speech")
    async def create_speech(req: CreateSpeechRequest) -> Response:
        client: Client = app.state.client
        default_model: str = app.state.model_name

        request_id = f"speech-{uuid.uuid4()}"

        gen_req = _build_speech_generate_request(req, default_model)
        if req.stream:
            return StreamingResponse(
                _speech_stream(
                    client=client,
                    gen_req=gen_req,
                    request_id=request_id,
                    response_format=req.response_format,
                    speed=req.speed,
                ),
                media_type="text/event-stream",
            )

        try:
            result = await client.speech(
                gen_req,
                request_id=request_id,
                response_format=req.response_format,
                speed=req.speed,
            )
        except ClientError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Error generating speech for request %s", request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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


async def _speech_stream(
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    response_format: str,
    speed: float,
):
    """Streaming speech generator (yields SSE events with audio chunks)."""
    chunk_index = 0
    emitted_samples = 0
    finish_reason: str | None = None
    usage: dict | None = None

    try:
        async for chunk in client.generate(gen_req, request_id=request_id):
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
                if chunk.usage is not None:
                    usage = chunk.usage.to_dict()

            if chunk.audio_data is None:
                continue

            sample_rate = chunk.sample_rate or DEFAULT_SAMPLE_RATE
            audio_data, emitted_samples = _select_speech_audio_delta(
                chunk.audio_data,
                emitted_samples=emitted_samples,
                is_terminal=chunk.finish_reason is not None,
            )
            if audio_data is None:
                continue

            audio_bytes, mime_type = encode_audio(
                audio_data,
                response_format=response_format,
                sample_rate=sample_rate,
                speed=speed,
            )
            actual_format = MIME_TO_FORMAT.get(mime_type, response_format)
            payload = {
                "id": f"speech-{request_id}",
                "object": "audio.speech.chunk",
                "index": chunk_index,
                "audio": {
                    "data": base64.b64encode(audio_bytes).decode("ascii"),
                    "format": actual_format,
                    "mime_type": mime_type,
                    "sample_rate": sample_rate,
                },
                "finish_reason": None,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            chunk_index += 1
    except ClientError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error streaming speech for request %s", request_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    final_payload = {
        "id": f"speech-{request_id}",
        "object": "audio.speech.chunk",
        "index": chunk_index,
        "audio": None,
        "finish_reason": finish_reason or "stop",
        "usage": usage,
    }
    yield f"data: {json.dumps(final_payload)}\n\n"
    yield "data: [DONE]\n\n"


def _select_speech_audio_delta(
    audio_data: Any,
    *,
    emitted_samples: int,
    is_terminal: bool,
) -> tuple[Any | None, int]:
    audio = to_numpy(audio_data)
    if audio.ndim > 1:
        audio = audio.squeeze()
    if audio.ndim > 1:
        if audio.shape[0] < audio.shape[-1]:
            audio = audio[0]
        else:
            audio = audio[:, 0]

    total_samples = int(audio.shape[-1]) if audio.ndim else 0
    if total_samples == 0:
        return None, emitted_samples
    if not is_terminal:
        return audio, emitted_samples + total_samples
    if total_samples <= emitted_samples:
        return None, emitted_samples
    return audio[emitted_samples:], total_samples


def _build_speech_generate_request(
    req: CreateSpeechRequest,
    default_model: str,
) -> GenerateRequest:
    """Convert a CreateSpeechRequest into a client GenerateRequest."""

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

    # Sampling params — use S2-Pro-tuned defaults
    sampling = SamplingParams(
        temperature=0.8, top_p=0.8, top_k=30, repetition_penalty=1.1
    )
    if req.max_new_tokens is not None:
        sampling.max_new_tokens = req.max_new_tokens
    if req.temperature is not None:
        sampling.temperature = req.temperature
    if req.top_p is not None:
        sampling.top_p = req.top_p
    if req.top_k is not None:
        sampling.top_k = req.top_k
    if req.repetition_penalty is not None:
        sampling.repetition_penalty = req.repetition_penalty

    # Build prompt: plain string if no references, dict otherwise
    prompt: Any = req.input
    references: list[dict[str, Any]] = []
    if req.references:
        references.extend(
            [reference.model_dump(exclude_none=True) for reference in req.references]
        )

    # Backward compatibility with ref_audio/ref_text form.
    if req.ref_audio is not None:
        ref: dict[str, Any] = {"audio_path": req.ref_audio}
        if req.ref_text is not None:
            ref["text"] = req.ref_text
        references.append(ref)

    if references:
        prompt = {"text": req.input, "references": references}

    return GenerateRequest(
        model=req.model or default_model,
        prompt=prompt,
        sampling=sampling,
        stage_params=req.stage_params,
        stream=req.stream,
        output_modalities=["audio"],
        metadata={
            "task": "tts",
            "tts_params": tts_params,
        },
    )
