# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible API server for sglang-omni.

Provides the following endpoints:
- POST /v1/chat/completions  — Text (+ audio) chat completions
- POST /v1/audio/speech      — Text-to-speech synthesis
- GET  /v1/models            — List available models
- GET  /health               — Health check
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang_omni.client import (
    Client,
    ClientError,
    GenerateRequest,
    Message,
    SamplingParams,
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


_UPLOAD_DIR = "/tmp/sglang_omni_uploads"

_MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    client: Client,
    *,
    model_name: str | None = None,
) -> FastAPI:
    """Create a FastAPI application with OpenAI-compatible endpoints.

    Args:
        client: Client instance connected to the pipeline coordinator.
        model_name: Default model name to report in responses and /v1/models.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="sglang-omni", version="0.1.0")

    # Store references in app state for access from route handlers
    app.state.client = client
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
        client: Client = app.state.client
        info = client.health()
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

    async for chunk in client.completion_stream(
        gen_req,
        request_id=request_id,
        audio_format=audio_format,
    ):
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


def _dedupe_keep_order(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _save_data_url_image(data_url: str) -> str:
    """
    Accept:
      data:image/png;base64,....
    Save to /tmp/sglang_omni_uploads and return local path.
    """
    os.makedirs(_UPLOAD_DIR, exist_ok=True)

    if not data_url.startswith("data:"):
        raise ValueError("Not a data URL")

    header, b64data = data_url.split(",", 1)
    # header: data:image/png;base64
    mime = "application/octet-stream"
    if ";" in header:
        mime = header[5:].split(";", 1)[0]  # strip 'data:'

    ext = _MIME_EXT.get(mime.lower(), ".img")

    try:
        raw = base64.b64decode(b64data)
    except Exception as e:
        raise ValueError(f"Invalid base64 data URL: {e}") from e

    path = os.path.join(_UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(raw)
    return path


def _extract_text_and_images_from_content(content: Any) -> tuple[Any, list[str]]:
    """
    OpenAI Chat Completions:
      content can be:
        - string
        - array of content parts [{type:"text", text:"..."}, {type:"image_url", image_url:{url:"..."}} ...]
    We normalize content -> plain text (string) and separately collect images.
    """
    if content is None or isinstance(content, str):
        return content, []

    if not isinstance(content, list):
        # Fallback: keep original, but don't treat it as image parts
        return content, []

    texts: list[str] = []
    images: list[str] = []

    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")

        if ptype == "text":
            t = part.get("text")
            if isinstance(t, str) and t:
                texts.append(t)

        elif ptype == "image_url":
            img = part.get("image_url")
            # Handle both object form {url: "..."} and legacy string form
            url: str | None = None
            if isinstance(img, dict):
                url = img.get("url")
            elif isinstance(img, str):
                url = img

            if isinstance(url, str) and url:
                # Support data URL (OpenAI allows base64-encoded images as data URL in many examples):contentReference[oaicite:1]{index=1}
                if url.startswith("data:image/"):
                    try:
                        url = _save_data_url_image(url)
                    except Exception as e:
                        raise ValueError(f"Invalid data URL image: {e}") from e
                images.append(url)

        elif ptype == "image_file":
            # OpenAI allows image_file reference to uploaded File（file_id）:contentReference[oaicite:2]{index=2}
            # Currently report 400
            imgf = part.get("image_file")
            file_id = imgf.get("file_id") if isinstance(imgf, dict) else None
            raise ValueError(
                f"image_file is not supported yet (file_id={file_id}). "
                "Implement File upload/resolve or use image_url."
            )

        else:
            # Unknown part types are ignored for now
            continue

    # Preserve text order; join by newline to avoid accidental concatenation
    normalized_text = "\n".join(texts) if texts else ""
    return normalized_text, images


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

    # Parse OpenAI content-parts, normalize messages, extract images ----
    extracted_images: list[str] = []
    messages: list[Message] = []

    for m in req.messages:
        try:
            normalized_content, imgs = _extract_text_and_images_from_content(m.content)
        except ValueError as e:
            # Make it a 400, not a 500
            raise HTTPException(status_code=400, detail=str(e))

        extracted_images.extend(imgs)
        messages.append(Message(role=m.role, content=normalized_content))

    # ---- Merge with extensions (top-level images/audios/videos) ----
    top_images = req.images or []
    images = _dedupe_keep_order([*_dedupe_keep_order(top_images), *extracted_images])

    audios = req.audios if req.audios else None
    videos = req.videos if req.videos else None

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
        client: Client = app.state.client
        default_model: str = app.state.model_name

        request_id = f"speech-{uuid.uuid4()}"

        gen_req = _build_speech_generate_request(req, default_model)

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

        return Response(
            content=result.audio_bytes,
            media_type=result.mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{result.format}"',
            },
        )


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
