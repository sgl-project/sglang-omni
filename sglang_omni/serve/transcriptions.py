# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible audio transcription endpoint."""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from collections.abc import Awaitable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from sglang_omni.client import Client, ClientError, GenerateRequest
from sglang_omni.config import AudioChunkingConfig
from sglang_omni.serve import speech_to_text
from sglang_omni.serve.openai_errors import is_bad_request_error
from sglang_omni.serve.protocol import TranscriptionResponse, TranscriptionUsage
from sglang_omni.serve.transcription_chunking import (
    ChunkPlan,
    ChunkSpan,
    check_total_duration,
    join_transcript_parts,
    needs_chunking,
    plan_audio_chunks,
)

logger = logging.getLogger(__name__)

TRANSCRIPTIONS_ENDPOINT = "/v1/audio/transcriptions"

_first_transcription_chunk = speech_to_text._first_speech_to_text_chunk
_transcription_stream = speech_to_text.speech_to_text_stream
_cancel_task_bounded = speech_to_text._cancel_task_bounded
_wait_for_request_disconnect = speech_to_text._wait_for_request_disconnect
_probe_audio_duration = speech_to_text.probe_audio_duration
build_transcription_generate_request = (
    speech_to_text.build_speech_to_text_generate_request
)

__all__ = [
    "_first_transcription_chunk",
    "_transcription_stream",
    "build_transcription_generate_request",
    "register_transcriptions",
]


def register_transcriptions(app: FastAPI) -> None:
    @app.post(TRANSCRIPTIONS_ENDPOINT)
    async def create_transcription(
        request: Request,
        form: speech_to_text.SpeechToTextForm = Depends(
            speech_to_text.parse_speech_to_text_form
        ),
    ) -> Response:
        client: Client = app.state.client
        default_model: str = app.state.model_name
        request_id = f"transcription-{uuid.uuid4()}"

        # TODO(Ratish): add the same pre-parser body limit used by voice uploads
        # once transcription upload limits are defined.
        audio_bytes = await speech_to_text.read_and_validate_speech_to_text_audio(
            form.file
        )

        chunking: AudioChunkingConfig = app.state.audio_chunking

        if form.stream:
            speech_to_text.validate_speech_to_text_response_format(
                form.response_format,
                stream=True,
                endpoint_path=TRANSCRIPTIONS_ENDPOINT,
            )
            duration_s = await asyncio.to_thread(_probe_audio_duration, audio_bytes)
            if (
                chunking.allow_audio_chunking
                and duration_s > chunking.stream_clip_limit_s
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "stream=true does not support audio longer than "
                        f"{chunking.stream_clip_limit_s:g} seconds; "
                        "use stream=false, which transcribes long audio in "
                        "chunks"
                    ),
                )
            gen_req = speech_to_text.build_speech_to_text_generate_request(
                audio_bytes=audio_bytes,
                filename=form.file.filename,
                content_type=form.file.content_type,
                model=form.model or default_model,
                language=form.language,
                prompt=form.prompt,
                temperature=form.temperature,
                max_new_tokens=form.max_new_tokens,
                stream=True,
            )
            return await speech_to_text.create_speech_to_text_streaming_response(
                request=request,
                client=client,
                gen_req=gen_req,
                request_id=request_id,
                audio_bytes=audio_bytes,
                architectures=getattr(app.state, "architectures", None),
                duration_s=duration_s,
            )

        duration_s = await asyncio.to_thread(_probe_audio_duration, audio_bytes)
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

        if plan is None:
            gen_req = speech_to_text.build_speech_to_text_generate_request(
                audio_bytes=audio_bytes,
                filename=form.file.filename,
                content_type=form.file.content_type,
                model=form.model or default_model,
                language=form.language,
                prompt=form.prompt,
                temperature=form.temperature,
                max_new_tokens=form.max_new_tokens,
            )
            result = await speech_to_text.complete_speech_to_text_request(
                client,
                gen_req,
                request_id=request_id,
                error_log_message="Error transcribing audio for request %s",
            )
            return speech_to_text.assemble_speech_to_text_response(
                text=result.text,
                response_format=form.response_format,
                endpoint_path=TRANSCRIPTIONS_ENDPOINT,
                task="transcribe",
                language=form.language,
                audio_bytes=audio_bytes,
                architectures=getattr(app.state, "architectures", None),
                duration_s=duration_s,
            )

        try:
            chunk_texts = await _await_transcription_with_disconnect_abort(
                request,
                _transcribe_audio_chunks(
                    client,
                    plan,
                    request_id=request_id,
                    model=form.model or default_model,
                    filename=form.file.filename,
                    language=form.language,
                    prompt=form.prompt,
                    temperature=form.temperature,
                    max_new_tokens=form.max_new_tokens,
                    max_concurrent=chunking.max_concurrent_chunks,
                ),
            )
        except ClientError as exc:
            if is_bad_request_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except (HTTPException, asyncio.CancelledError):
            raise
        except Exception as exc:
            if is_bad_request_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            logger.exception("Error transcribing audio for request %s", request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        text = join_transcript_parts(chunk_texts)
        return _assemble_chunked_response(
            text=text,
            response_format=form.response_format,
            language=form.language,
            plan=plan,
            chunk_texts=chunk_texts,
            architectures=getattr(app.state, "architectures", None),
        )


def _assemble_chunked_response(
    *,
    text: str,
    response_format: str,
    language: str | None,
    plan: ChunkPlan,
    chunk_texts: list[str],
    architectures: list[str] | None,
) -> Response:
    """Chunk-aware sibling of speech_to_text.assemble_speech_to_text_response.

    The shared assembler probes the upload for a duration and builds one
    verbose segment for the whole file; here the plan already knows the exact
    duration and where each chunk sits, so verbose_json gets one segment per
    chunk with real timestamps.
    """
    normalized_response_format = speech_to_text.validate_speech_to_text_response_format(
        response_format,
        stream=False,
        endpoint_path=TRANSCRIPTIONS_ENDPOINT,
    )
    if normalized_response_format == "text":
        return PlainTextResponse(text)

    adapter = speech_to_text.resolve_speech_to_text_adapter(architectures)
    text = adapter.postprocess_text(text)
    duration_s = plan.duration_s
    usage = (
        TranscriptionUsage(seconds=math.ceil(duration_s)) if duration_s > 0 else None
    )
    if normalized_response_format == "verbose_json":
        response = adapter.build_verbose_response_from_chunks(
            text=text,
            chunks=[
                (span.start_s, span.end_s, chunk_text)
                for span, chunk_text in zip(plan.spans, chunk_texts)
            ],
            language=language,
            audio_duration_s=duration_s,
        )
        response.task = "transcribe"
        response.usage = usage
        return JSONResponse(content=response.model_dump(exclude_none=True))
    return JSONResponse(
        content=TranscriptionResponse(text=text, usage=usage).model_dump(
            exclude_none=True
        )
    )


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
    (asyncio.gather preserves input order); the caller joins them and,
    for verbose_json, pairs them with the spans' timestamps. Up to
    max_concurrent chunks run in the engine at once.

    Any chunk failing fails the whole request. The error names the chunk and
    its time range to make sure the failure is diagnosable.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    in_flight: set[str] = set()

    async def run_chunk(span: ChunkSpan) -> str:
        if not span.has_speech:
            return ""
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
        raise asyncio.CancelledError
    finally:
        # Clean up both tasks on every exit, otherwise the work task keeps
        # running and nobody aborts its engine requests.
        if not work_task.done():
            await _cancel_task_bounded(work_task)
        if not disconnect_task.done():
            await _cancel_task_bounded(disconnect_task)
