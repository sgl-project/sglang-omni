# SPDX-License-Identifier: Apache-2.0
"""Shared mechanics for OpenAI-compatible speech-to-text endpoints."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
from collections.abc import AsyncIterator, Collection
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from sglang_omni.client import (
    Client,
    ClientError,
    CompletionResult,
    GenerateChunk,
    GenerateRequest,
    SamplingParams,
)
from sglang_omni.serve.generation_params import record_explicit_generation_params
from sglang_omni.serve.openai_errors import is_bad_request_error
from sglang_omni.serve.protocol import (
    TranscriptionResponse,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
    TranscriptionUsage,
)
from sglang_omni.serve.streaming import (
    STREAM_DONE_SENTINEL,
    ClosableStreamingResponse,
    close_async_iterator_if_supported,
)
from sglang_omni.serve.transcription_adapters import resolve_adapter
from sglang_omni.serve.transcription_adapters.base import TranscriptionAdapter

logger = logging.getLogger(__name__)
HTTP_DISCONNECT_POLL_INTERVAL_S = 0.05
HTTP_DISCONNECT_CANCEL_TIMEOUT_S = 0.1
DEFAULT_RESPONSE_FORMATS = frozenset({"json", "text", "verbose_json"})
DEFAULT_STREAMING_RESPONSE_FORMATS = frozenset({"json", "text"})


@dataclass(frozen=True, slots=True)
class SpeechToTextForm:
    """Separate shared form parsing from endpoint-specific policy."""

    file: UploadFile
    model: str | None
    language: str | None
    prompt: str | None
    response_format: str
    temperature: float | None
    max_new_tokens: int | None
    stream: bool


async def parse_speech_to_text_form(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float | None = Form(default=None),
    max_new_tokens: int | None = Form(default=None, ge=1),
    stream: bool = Form(default=False),
) -> SpeechToTextForm:
    return SpeechToTextForm(
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        stream=stream,
    )


async def read_and_validate_speech_to_text_audio(file: UploadFile) -> bytes:
    """Reject empty uploads before dispatch can consume backend resources."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
    return audio_bytes


def validate_speech_to_text_response_format(
    response_format: str,
    *,
    stream: bool,
    endpoint_path: str,
    response_formats: Collection[str] = DEFAULT_RESPONSE_FORMATS,
    streaming_response_formats: Collection[str] = DEFAULT_STREAMING_RESPONSE_FORMATS,
) -> str:
    """Keep format errors endpoint-specific without duplicating validation."""
    normalized_response_format = response_format.strip().lower()
    if stream and normalized_response_format not in streaming_response_formats:
        raise HTTPException(
            status_code=400,
            detail=(
                "stream=true supports only response_format 'json' or "
                f"'text', got {response_format!r}"
            ),
        )
    if not stream and normalized_response_format not in response_formats:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported response_format for {endpoint_path}: {response_format!r}"
            ),
        )
    return normalized_response_format


def build_speech_to_text_generate_request(
    *,
    audio_bytes: bytes,
    filename: str | None,
    content_type: str | None,
    model: str,
    language: str | None,
    prompt: str | None,
    temperature: float | None,
    max_new_tokens: int | None = None,
    stream: bool = False,
    task: str = "transcribe",
) -> GenerateRequest:
    """Keep endpoint policy out of model-neutral request construction."""
    params: dict[str, Any] = {"task": task}
    metadata: dict[str, Any] = {"task": "asr"}
    explicit_fields: list[str] = []
    if language is not None:
        params["language"] = language
    if prompt is not None:
        params["prompt"] = prompt
    if temperature is not None:
        explicit_fields.append("temperature")
    if max_new_tokens is not None:
        explicit_fields.append("max_new_tokens")
    record_explicit_generation_params(metadata, sorted(explicit_fields))
    sampling = SamplingParams(
        temperature=temperature if temperature is not None else 0.0,
        max_new_tokens=max_new_tokens,
    )

    return GenerateRequest(
        model=model,
        prompt={
            "audio_bytes": audio_bytes,
            "filename": filename,
            "content_type": content_type,
        },
        sampling=sampling,
        extra_params=params,
        stream=stream,
        output_modalities=["text"],
        metadata=metadata,
    )


# note (Junnan Li): Keep the old name while stacked callers migrate to the
# endpoint-neutral shared API.
build_transcription_generate_request = build_speech_to_text_generate_request


async def complete_speech_to_text_request(
    client: Client,
    gen_req: GenerateRequest,
    *,
    request_id: str,
    error_log_message: str,
) -> CompletionResult:
    """Keep sibling endpoints on the same backend-to-HTTP error mapping."""
    try:
        return await client.completion(gen_req, request_id=request_id)
    except ClientError as exc:
        if is_bad_request_error(exc):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        if is_bad_request_error(exc):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.exception(error_log_message, request_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def resolve_speech_to_text_adapter(
    architectures: list[str] | None,
) -> TranscriptionAdapter:
    return resolve_adapter(architectures)


def probe_audio_duration(audio_bytes: bytes) -> float:
    """Avoid a full decode; unknown duration is valid for response formatting."""
    try:
        import soundfile as sf

        info = sf.info(io.BytesIO(audio_bytes))
        if info.samplerate:
            return max(info.frames / float(info.samplerate), 0.0)
    except (RuntimeError, ValueError):
        logger.debug("Could not probe audio duration", exc_info=True)
    return 0.0


def assemble_speech_to_text_response(
    *,
    text: str,
    response_format: str,
    endpoint_path: str,
    task: str,
    language: str | None,
    audio_bytes: bytes,
    architectures: list[str] | None,
) -> Response:
    """Keep response schemas consistent across sibling endpoints."""
    normalized_response_format = validate_speech_to_text_response_format(
        response_format,
        stream=False,
        endpoint_path=endpoint_path,
    )
    if normalized_response_format == "text":
        return PlainTextResponse(text)

    adapter = resolve_speech_to_text_adapter(architectures)
    text = adapter.postprocess_text(text)
    duration_s = probe_audio_duration(audio_bytes)
    usage = (
        TranscriptionUsage(seconds=math.ceil(duration_s)) if duration_s > 0 else None
    )
    if normalized_response_format == "verbose_json":
        response = adapter.build_verbose_response(
            text=text,
            language=language,
            audio_duration_s=duration_s,
        )
        response.task = task
        response.usage = usage
        return JSONResponse(content=response.model_dump(exclude_none=True))
    return JSONResponse(
        content=TranscriptionResponse(text=text, usage=usage).model_dump(
            exclude_none=True
        )
    )


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


async def _abort_and_close_speech_to_text_stream(
    client: Client,
    request_id: str,
    stream: AsyncIterator[Any],
) -> None:
    try:
        await client.abort(request_id)
    finally:
        await close_async_iterator_if_supported(stream)


async def _first_speech_to_text_chunk(
    request: Request,
    client: Client,
    chunk_stream: AsyncIterator[GenerateChunk],
    request_id: str,
) -> GenerateChunk | None:
    # note (Junnan Li): Admit before headers so model validation remains an
    # HTTP error instead of becoming an SSE error event.
    disconnect_task = asyncio.create_task(_wait_for_request_disconnect(request))
    first_chunk_task = asyncio.create_task(anext(chunk_stream))
    try:
        done, _ = await asyncio.wait(
            {first_chunk_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnect_task in done:
            await _cancel_task_bounded(first_chunk_task)
            await _abort_and_close_speech_to_text_stream(
                client, request_id, chunk_stream
            )
            raise asyncio.CancelledError
        try:
            return first_chunk_task.result()
        except StopAsyncIteration:
            return None
    finally:
        if not disconnect_task.done():
            await _cancel_task_bounded(disconnect_task)


async def speech_to_text_stream(
    chunk_stream: AsyncIterator[GenerateChunk],
    *,
    first_chunk: GenerateChunk | None,
    request_id: str,
    adapter: TranscriptionAdapter,
    duration_s: float,
    operation_name: str = "transcription",
) -> AsyncIterator[str]:
    """Keep terminal event ordering stable for OpenAI-compatible clients."""
    final_text: str | None = None

    def _event_for(chunk: GenerateChunk) -> str | None:
        nonlocal final_text
        if chunk.finish_reason is not None:
            if isinstance(chunk.text, str) and chunk.text:
                final_text = chunk.text
            return None
        if chunk.modality == "text" and chunk.text:
            event = TranscriptionTextDeltaEvent(delta=chunk.text)
            return f"data: {event.model_dump_json(exclude_none=True)}\n\n"
        return None

    try:
        async with aclosing(chunk_stream):
            if first_chunk is not None:
                line = _event_for(first_chunk)
                if line is not None:
                    yield line
            async for chunk in chunk_stream:
                line = _event_for(chunk)
                if line is not None:
                    yield line
    except Exception as exc:
        logger.exception(
            "Error streaming %s for request %s", operation_name, request_id
        )
        payload = {"type": "error", "error": {"message": str(exc)}}
        yield f"data: {json.dumps(payload)}\n\n"
        return

    text = adapter.postprocess_text(final_text or "")
    usage = (
        TranscriptionUsage(seconds=math.ceil(duration_s)) if duration_s > 0 else None
    )
    done_event = TranscriptionTextDoneEvent(text=text, usage=usage)
    yield f"data: {done_event.model_dump_json(exclude_none=True)}\n\n"
    yield f"data: {STREAM_DONE_SENTINEL}\n\n"


async def create_speech_to_text_streaming_response(
    *,
    request: Request,
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    audio_bytes: bytes,
    architectures: list[str] | None,
    operation_name: str = "transcription",
) -> Response:
    """Delay headers until backend admission can still return an HTTP error."""
    adapter = resolve_speech_to_text_adapter(architectures)
    duration_s = probe_audio_duration(audio_bytes)
    chunk_stream = client.generate(gen_req, request_id=request_id)
    try:
        first_chunk = await _first_speech_to_text_chunk(
            request, client, chunk_stream, request_id
        )
    except ClientError as exc:
        await close_async_iterator_if_supported(chunk_stream)
        if is_bad_request_error(exc):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        await close_async_iterator_if_supported(chunk_stream)
        if is_bad_request_error(exc):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.exception(
            "Error starting %s stream for request %s",
            operation_name,
            request_id,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ClosableStreamingResponse(
        speech_to_text_stream(
            chunk_stream,
            first_chunk=first_chunk,
            request_id=request_id,
            adapter=adapter,
            duration_s=duration_s,
            operation_name=operation_name,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Request-Id": request_id},
    )
