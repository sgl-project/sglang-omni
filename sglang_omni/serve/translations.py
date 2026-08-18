# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible audio translation endpoint."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response

from sglang_omni.client import Client
from sglang_omni.serve import speech_to_text
from sglang_omni.serve.speech_errors import openai_error_response

TRANSLATIONS_ENDPOINT = "/v1/audio/translations"
# note (Junnan Li): rejected until a real segment-timestamp channel exists
_SEGMENT_FORMATS = frozenset({"srt", "vtt"})


def _invalid_request(
    message: str,
    *,
    param: str | None,
    status_code: int = 400,
    code: str | None = None,
) -> Response:
    return openai_error_response(
        message,
        status_code=status_code,
        error_type="invalid_request_error",
        param=param,
        code=code,
    )


def _http_exception_response(exc: HTTPException, *, param: str | None) -> Response:
    error_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
    return openai_error_response(
        str(exc.detail),
        status_code=exc.status_code,
        error_type=error_type,
        param=param,
        code=None,
    )


def register_translations(app: FastAPI) -> None:
    @app.post(TRANSLATIONS_ENDPOINT)
    async def create_translation(
        request: Request,
        form: speech_to_text.SpeechToTextForm = Depends(
            speech_to_text.parse_speech_to_text_form
        ),
    ) -> Response:
        client: Client = app.state.client
        default_model: str = app.state.model_name
        model = form.model or default_model
        request_id = f"translation-{uuid.uuid4()}"

        if model != default_model:
            return _invalid_request(
                f"The model {model!r} does not exist.",
                param="model",
                status_code=404,
                code="model_not_found",
            )

        if not app.state.supports_audio_translation:
            return _invalid_request(
                f"Model {model!r} does not support {TRANSLATIONS_ENDPOINT}; "
                "use /v1/audio/transcriptions instead.",
                param="model",
            )

        normalized_response_format = form.response_format.strip().lower()
        if normalized_response_format in _SEGMENT_FORMATS:
            return _invalid_request(
                f"response_format {normalized_response_format!r} requires a "
                "segment-timestamp capability that the translation pipeline "
                "does not provide.",
                param="response_format",
            )
        try:
            speech_to_text.validate_speech_to_text_response_format(
                form.response_format,
                stream=form.stream,
                endpoint_path=TRANSLATIONS_ENDPOINT,
            )
        except HTTPException as exc:
            return _http_exception_response(exc, param="response_format")

        try:
            audio_bytes = await speech_to_text.read_and_validate_speech_to_text_audio(
                form.file
            )
        except HTTPException as exc:
            return _http_exception_response(exc, param="file")

        # note (Junnan Li): probe once off the event loop and pass the
        # duration through, matching transcriptions.
        duration_s = await asyncio.to_thread(
            speech_to_text.probe_audio_duration, audio_bytes
        )

        language = form.language
        if language is None:
            language = await speech_to_text.detect_speech_language(
                client,
                audio_bytes=audio_bytes,
                filename=form.file.filename,
                content_type=form.file.content_type,
                model=model,
                request_id=request_id,
            )

        gen_req = speech_to_text.build_speech_to_text_generate_request(
            audio_bytes=audio_bytes,
            filename=form.file.filename,
            content_type=form.file.content_type,
            model=model,
            language=language,
            prompt=form.prompt,
            temperature=form.temperature,
            max_new_tokens=form.max_new_tokens,
            stream=form.stream,
            task="translate",
        )
        if form.stream:
            try:
                return await speech_to_text.create_speech_to_text_streaming_response(
                    request=request,
                    client=client,
                    gen_req=gen_req,
                    request_id=request_id,
                    audio_bytes=audio_bytes,
                    architectures=getattr(app.state, "architectures", None),
                    duration_s=duration_s,
                    operation_name="translation",
                )
            except HTTPException as exc:
                return _http_exception_response(exc, param=None)

        try:
            result = await speech_to_text.complete_speech_to_text_request(
                client,
                gen_req,
                request_id=request_id,
                error_log_message="Error translating audio for request %s",
            )
        except HTTPException as exc:
            return _http_exception_response(exc, param=None)

        # note (Junnan Li): verbose_json keeps transcription parity: with no
        # segment-timestamp channel, the shared adapter emits one placeholder
        # segment.
        return speech_to_text.assemble_speech_to_text_response(
            text=result.text,
            response_format=form.response_format,
            endpoint_path=TRANSLATIONS_ENDPOINT,
            task="translate",
            language=language,
            audio_bytes=audio_bytes,
            architectures=getattr(app.state, "architectures", None),
            duration_s=duration_s,
        )


__all__ = ["register_translations"]
