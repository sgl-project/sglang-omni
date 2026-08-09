# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible audio transcription route and request helpers."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from sglang_omni.client import (
    Client,
    ClientError,
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

logger = logging.getLogger(__name__)
HTTP_DISCONNECT_POLL_INTERVAL_S = 0.05
HTTP_DISCONNECT_CANCEL_TIMEOUT_S = 0.1


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


async def _abort_and_close_transcription_stream(
    client: Client,
    request_id: str,
    stream: AsyncIterator[Any],
) -> None:
    try:
        await client.abort(request_id)
    finally:
        await close_async_iterator_if_supported(stream)


async def _first_transcription_chunk(
    request: Request,
    client: Client,
    chunk_stream: AsyncIterator[GenerateChunk],
    request_id: str,
) -> GenerateChunk | None:
    """Wait for the first stream chunk while watching for client disconnect."""
    disconnect_task = asyncio.create_task(_wait_for_request_disconnect(request))
    first_chunk_task = asyncio.create_task(anext(chunk_stream))
    try:
        done, _ = await asyncio.wait(
            {first_chunk_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnect_task in done:
            await _cancel_task_bounded(first_chunk_task)
            await _abort_and_close_transcription_stream(
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


def register_transcriptions(app: FastAPI) -> None:
    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        request: Request,
        file: UploadFile = File(...),
        model: str | None = Form(default=None),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        response_format: str = Form(default="json"),
        temperature: float | None = Form(default=None),
        max_new_tokens: int | None = Form(default=None, ge=1),
        stream: bool = Form(default=False),
    ) -> Response:
        client: Client = app.state.client
        default_model: str = app.state.model_name
        request_id = f"transcription-{uuid.uuid4()}"

        # TODO(Ratish): add the same pre-parser body limit used by voice uploads
        # once transcription upload limits are defined.
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

        normalized_response_format = response_format.strip().lower()
        if stream:
            if normalized_response_format not in {"json", "text"}:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "stream=true supports only response_format 'json' or "
                        f"'text', got {response_format!r}"
                    ),
                )
            gen_req = build_transcription_generate_request(
                audio_bytes=audio_bytes,
                filename=file.filename,
                content_type=file.content_type,
                model=model or default_model,
                language=language,
                prompt=prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                stream=True,
            )
            adapter = resolve_adapter(getattr(app.state, "architectures", None))
            duration_s = _probe_audio_duration(audio_bytes)
            chunk_stream = client.generate(gen_req, request_id=request_id)
            # Pull the first chunk before sending response headers so admission
            # failures map to HTTP statuses rather than SSE error payloads.
            try:
                first_chunk = await _first_transcription_chunk(
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
                    "Error starting transcription stream for request %s",
                    request_id,
                )
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return ClosableStreamingResponse(
                _transcription_stream(
                    chunk_stream,
                    first_chunk=first_chunk,
                    request_id=request_id,
                    adapter=adapter,
                    duration_s=duration_s,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Request-Id": request_id},
            )

        gen_req = build_transcription_generate_request(
            audio_bytes=audio_bytes,
            filename=file.filename,
            content_type=file.content_type,
            model=model or default_model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        try:
            result = await client.completion(gen_req, request_id=request_id)
        except ClientError as exc:
            if is_bad_request_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            if is_bad_request_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            logger.exception("Error transcribing audio for request %s", request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        text = result.text
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


async def _transcription_stream(
    chunk_stream: AsyncIterator[GenerateChunk],
    *,
    first_chunk: GenerateChunk | None,
    request_id: str,
    adapter: Any,
    duration_s: float,
) -> AsyncIterator[str]:
    """SSE generator for streaming transcriptions.

    Emits OpenAI-style transcript.text.delta events for each partial text
    chunk, then a terminal transcript.text.done event carrying the full
    post-processed transcript. The caller already pulled first_chunk from
    chunk_stream so admission failures map to HTTP statuses before response
    headers go out.
    """
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
        logger.exception("Error streaming transcription for request %s", request_id)
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
    max_new_tokens: int | None = None,
    stream: bool = False,
) -> GenerateRequest:
    params: dict[str, Any] = {"task": "transcribe"}
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
