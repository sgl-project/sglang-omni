# SPDX-License-Identifier: Apache-2.0
"""Standalone speaker diarization endpoints (an Omni API extension)."""

from __future__ import annotations

import uuid

import msgspec
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse

from sglang_omni.client.types import GenerateRequest
from sglang_omni.serve.diarization_ws import DiarizationSession
from sglang_omni.serve.speech_to_text import (
    complete_speech_to_text_request,
    read_and_validate_speech_to_text_audio,
)

DIARIZATION_ARCHITECTURES = frozenset(
    {"Nemotron3DiarizationForAudioFrameClassification"}
)
FORM_FIELDS = frozenset({"file", "model", "response_format", "stream"})


def register_diarizations(app: FastAPI) -> None:
    @app.websocket("/v1/audio/diarizations/stream")
    async def live_diarization(websocket: WebSocket) -> None:
        await websocket.accept()
        health = app.state.client.health()
        if not DIARIZATION_ARCHITECTURES.intersection(app.state.architectures):
            message = "This model does not support standalone diarization"
        elif health["stages"] != [health["entry_stage"]]:
            message = "Live diarization requires a single-stage pipeline without process replicas"
        else:
            await DiarizationSession(
                websocket, client=app.state.client, model_name=app.state.model_name
            ).run()
            return
        await websocket.send_json({"type": "error", "message": message})
        await websocket.close(code=1008)

    @app.post("/v1/audio/diarizations")
    async def create_diarization(
        request: Request,
        file: UploadFile = File(...),
        model: str | None = Form(default=None),
        response_format: str = Form(default="json"),
        stream: bool = Form(default=False),
    ) -> JSONResponse:
        unknown_fields = sorted(set(await request.form()) - FORM_FIELDS)
        if not DIARIZATION_ARCHITECTURES.intersection(app.state.architectures):
            raise HTTPException(
                status_code=400,
                detail="This model does not support standalone diarization",
            )
        elif model is not None and model != app.state.model_name:
            raise HTTPException(
                status_code=404, detail=f"Model {model!r} is not served"
            )
        elif stream or response_format != "json":
            raise HTTPException(
                status_code=400,
                detail="Diarization supports stream=false and response_format=json",
            )
        elif unknown_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported diarization fields: {unknown_fields}",
            )
        else:
            pass
        request_id = f"diarization-{uuid.uuid4()}"
        generation = GenerateRequest(
            model=model or app.state.model_name,
            prompt={"audio_bytes": await read_and_validate_speech_to_text_audio(file)},
            stream=False,
            metadata={"task": "diarization"},
        )
        result = await complete_speech_to_text_request(
            request,
            app.state.client,
            generation,
            request_id=request_id,
            error_log_message="Diarization failed for %s",
        )
        if result.diarization is None:
            raise HTTPException(
                status_code=500, detail="Model returned no diarization result"
            )
        else:
            pass
        return JSONResponse(
            content=msgspec.to_builtins(result.diarization),
            headers={"X-Request-Id": request_id},
        )
