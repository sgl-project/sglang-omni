# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible audio transcription route and request helpers."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import uuid
from collections.abc import AsyncIterator, Awaitable
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
from sglang_omni.config import AudioChunkingConfig
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
from sglang_omni.serve.transcription_chunking import (
    ChunkPlan,
    ChunkSpan,
    check_total_duration,
    join_transcript_parts,
    needs_chunking,
    plan_audio_chunks,
)

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
            # Chunking does not cover the streaming path yet. Without this
            # guard a long upload would run as one request and the output
            # budget would silently cut the transcript tail.
            stream_chunking: AudioChunkingConfig = app.state.audio_chunking
            if needs_chunking(_probe_audio_duration(audio_bytes), stream_chunking):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "stream=true does not support audio longer than "
                        f"{stream_chunking.max_audio_clip_s:g} seconds yet; "
                        "use stream=false, which transcribes long audio in "
                        "chunks"
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

        duration_s = _probe_audio_duration(audio_bytes)
        chunking: AudioChunkingConfig = app.state.audio_chunking
        plan: ChunkPlan | None = None
        if needs_chunking(duration_s, chunking):
            try:
                # Check duration before decoding: a small compressed file can
                # hold hours of audio, and decoding that eats gigabytes.
                check_total_duration(duration_s, chunking)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            # Decode + split in a worker thread: decoding a long file is pure
            # CPU and would stall the event loop.
            plan = await asyncio.to_thread(plan_audio_chunks, audio_bytes, chunking)

        chunk_texts: list[str] | None = None
        try:
            if plan is not None:
                duration_s = plan.duration_s
                chunk_texts = await _await_transcription_with_disconnect_abort(
                    request,
                    _transcribe_audio_chunks(
                        client,
                        plan,
                        request_id=request_id,
                        model=model or default_model,
                        filename=file.filename,
                        language=language,
                        prompt=prompt,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        max_concurrent=chunking.max_concurrent_chunks,
                    ),
                )
                text = join_transcript_parts(chunk_texts)
            else:
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
                result = await client.completion(gen_req, request_id=request_id)
                text = result.text
        except ClientError as exc:
            if is_bad_request_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            if is_bad_request_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            logger.exception("Error transcribing audio for request %s", request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
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
        usage = (
            TranscriptionUsage(seconds=math.ceil(duration_s))
            if duration_s > 0
            else None
        )
        if normalized_response_format == "verbose_json":
            if plan is not None and chunk_texts is not None:
                # Chunked path: the split already knows where each chunk sits
                # in the upload, so segments get real (chunk-level) timestamps
                # instead of one segment spanning the whole file.
                response = adapter.build_verbose_response_from_chunks(
                    text=text,
                    chunks=[
                        (span.start_s, span.end_s, chunk_text)
                        for span, chunk_text in zip(plan.spans, chunk_texts)
                    ],
                    language=language,
                    audio_duration_s=duration_s,
                )
            else:
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


def _build_chunk_generate_request(
    chunk_bytes: bytes,
    *,
    model: str,
    filename: str | None,
    language: str | None,
    prompt: str | None,
    temperature: float | None,
    max_new_tokens: int | None,
    stream: bool = False,
) -> GenerateRequest:
    return build_transcription_generate_request(
        audio_bytes=chunk_bytes,
        filename=filename,
        # Chunks are re-encoded WAV no matter what the upload was.
        # (Nothing downstream reads this field; just kept it anyway.)
        content_type="audio/wav",
        model=model,
        language=language,
        prompt=prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        stream=stream,
    )


async def _transcribe_audio_chunks(
    client: Client,
    plan: ChunkPlan,
    *,
    request_id: str,
    model: str,
    filename: str | None,
    language: str | None,
    prompt: str | None,
    temperature: float | None,
    max_new_tokens: int | None,
    max_concurrent: int,
) -> list[str]:
    """Transcribe the chunks of a plan, returning one text per chunk.

    Texts come back in span order no matter which chunk finishes first
    (``asyncio.gather`` preserves input order); the caller joins them and,
    for verbose_json, pairs them with the spans' timestamps. Up to
    ``max_concurrent`` chunks run in the engine at once.

    Any chunk failing fails the whole request. The error names the chunk and
    its time range to make sure the failure is diagnosable.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    in_flight: set[str] = set()

    async def run_chunk(span: ChunkSpan) -> str:
        async with semaphore:
            # Encode inside the semaphore so at most max_concurrent chunk
            # WAVs exist at a time.
            chunk_bytes = await asyncio.to_thread(plan.encode, span)
            gen_req = _build_chunk_generate_request(
                chunk_bytes,
                model=model,
                filename=filename,
                language=language,
                prompt=prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            chunk_request_id = f"{request_id}-chunk-{span.index}"
            in_flight.add(chunk_request_id)
            # in_flight lists engine requests that may still be running.
            # Cancelling our local task does NOT stop the engine -- the
            # request keeps computing over there. So an id is removed only
            # when the engine side truly ended (result returned, or the
            # request itself errored); a cancelled chunk stays listed, and
            # the cleanup below aborts it by id.
            try:
                result = await client.completion(gen_req, request_id=chunk_request_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                in_flight.discard(chunk_request_id)
                raise ClientError(
                    f"transcription failed for chunk {span.index} "
                    f"({span.start_s:.1f}s-{span.end_s:.1f}s): {exc}"
                ) from exc
            in_flight.discard(chunk_request_id)
            return result.text

    tasks = [asyncio.create_task(run_chunk(span)) for span in plan.spans]
    try:
        texts = await asyncio.gather(*tasks)
    except BaseException:
        # One chunk failed (or we were cancelled by a client disconnect).
        # in_flight holds every chunk whose engine request has not finished,
        # including chunks whose local task got cancelled above us.
        pending_engine_requests = sorted(in_flight)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Cancelling the local task does not stop the engine; abort by id.
        for chunk_request_id in pending_engine_requests:
            try:
                await client.abort(chunk_request_id)
            except Exception:
                logger.warning("Failed to abort chunk request %s", chunk_request_id)
        raise
    return texts


async def _await_transcription_with_disconnect_abort(
    request: Request,
    work: Awaitable[list[str]],
) -> list[str]:
    """Run chunked transcription while watching for client disconnect.

    The non-stream handler has no response-owned disconnect watcher, so
    without this a disconnected client would leave every chunk running to
    completion. Cancelling the work task triggers its own cleanup path,
    which aborts the in-flight engine requests.
    """
    work_task = asyncio.create_task(work)
    disconnect_task = asyncio.create_task(_wait_for_request_disconnect(request))
    try:
        done, _ = await asyncio.wait(
            {work_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if work_task in done:  # transcription completed first
            return work_task.result()
        # Bounded: the work task's cleanup (child cancels + engine aborts)
        # keeps running inside the task even after we stop waiting; the bound
        # only protects this handler from a cleanup that itself hangs.
        await _cancel_task_bounded(work_task)
        raise asyncio.CancelledError
    finally:
        if not disconnect_task.done():
            await _cancel_task_bounded(disconnect_task)


def _probe_audio_duration(audio_bytes: bytes) -> float:
    """Best-effort audio duration (seconds) from raw upload bytes."""
    try:
        import av

        with av.open(io.BytesIO(audio_bytes), metadata_errors="ignore") as container:
            if container.duration:  # in av.time_base units (microseconds)
                return max(container.duration / 1_000_000, 0.0)
            for audio_stream in container.streams.audio:
                if audio_stream.duration and audio_stream.time_base:
                    return max(
                        float(audio_stream.duration * audio_stream.time_base), 0.0
                    )
    except Exception:
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
