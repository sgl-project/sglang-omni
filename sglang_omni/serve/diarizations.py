# SPDX-License-Identifier: Apache-2.0
"""Standalone speaker diarization endpoint (an Omni API extension)."""

from __future__ import annotations

import asyncio
import logging
import uuid

import msgspec
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse

from sglang_omni.admission import QueueFullError
from sglang_omni.client import Client, GenerateRequest
from sglang_omni.client.types import CompletionResult
from sglang_omni.serve.openai_errors import is_bad_request_error
from sglang_omni.serve.speech_to_text import read_and_validate_speech_to_text_audio

logger = logging.getLogger(__name__)
_SUPPORTED_ARCHITECTURES = frozenset({"SortformerEncLabelModel"})
_FORM_FIELDS = frozenset({"file", "model", "response_format", "stream"})


async def complete_diarization(
    request: Request, client: Client, generation: GenerateRequest, request_id: str
) -> CompletionResult:
    """Abort coordinator work when the upload's HTTP client disconnects."""
    task = asyncio.create_task(client.completion(generation, request_id=request_id))
    completed = False
    try:
        while not task.done():
            if await request.is_disconnected():
                raise HTTPException(status_code=499, detail="Client disconnected")
            await asyncio.wait({task}, timeout=0.05)
        completed = True
        return task.result()
    finally:
        if not completed:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            try:
                await client.abort(request_id)
            except Exception:
                logger.exception("Failed to abort diarization request %s", request_id)


def register_diarizations(app: FastAPI) -> None:
    @app.websocket("/v1/audio/diarizations/stream")
    async def live_diarization(websocket: WebSocket) -> None:
        from sglang_omni.serve.diarization_ws import DiarizationSession

        await websocket.accept()
        if not _SUPPORTED_ARCHITECTURES.intersection(app.state.architectures or []):
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "This model does not support standalone diarization",
                }
            )
            await websocket.close(code=1008)
            return
        health = app.state.client.health()
        if health["stages"] != [health["entry_stage"]]:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Live diarization requires a single-stage pipeline without process replicas",
                }
            )
            await websocket.close(code=1008)
            return
        await DiarizationSession(
            websocket, client=app.state.client, model_name=app.state.model_name
        ).run()

    @app.post("/v1/audio/diarizations")
    async def create_diarization(
        request: Request,
        file: UploadFile = File(...),
        model: str | None = Form(default=None),
        response_format: str = Form(default="json"),
        stream: bool = Form(default=False),
    ) -> JSONResponse:
        if not _SUPPORTED_ARCHITECTURES.intersection(app.state.architectures or []):
            raise HTTPException(
                status_code=400,
                detail="This model does not support standalone diarization",
            )
        if model is not None and model != app.state.model_name:
            raise HTTPException(
                status_code=404, detail=f"Model {model!r} is not served"
            )
        if stream or response_format != "json":
            raise HTTPException(
                status_code=400,
                detail="Diarization supports stream=false and response_format=json",
            )
        unknown = set(await request.form()) - _FORM_FIELDS
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported diarization fields: {sorted(unknown)}",
            )
        audio_bytes = await read_and_validate_speech_to_text_audio(file)
        generation = GenerateRequest(
            model=model or app.state.model_name,
            prompt={"audio_bytes": audio_bytes},
            stream=False,
            metadata={"task": "diarization"},
        )
        request_id = f"diarization-{uuid.uuid4()}"
        try:
            result = await complete_diarization(
                request, app.state.client, generation, request_id
            )
        except HTTPException:
            raise
        except Exception as exc:
            if QueueFullError.matches(exc):
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            if is_bad_request_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            logger.exception("Diarization failed for %s", request_id)
            raise HTTPException(status_code=500, detail="Diarization failed") from exc
        if result.diarization is None:
            raise HTTPException(
                status_code=500, detail="Model returned no diarization result"
            )
        return JSONResponse(
            content=msgspec.to_builtins(result.diarization),
            headers={"X-Request-Id": request_id},
        )
