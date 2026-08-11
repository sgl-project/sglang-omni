# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible audio transcription endpoint."""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, Request
from fastapi.responses import Response

from sglang_omni.client import Client
from sglang_omni.serve import speech_to_text

TRANSCRIPTIONS_ENDPOINT = "/v1/audio/transcriptions"

_first_transcription_chunk = speech_to_text._first_speech_to_text_chunk
_transcription_stream = speech_to_text.speech_to_text_stream
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

        if form.stream:
            speech_to_text.validate_speech_to_text_response_format(
                form.response_format,
                stream=True,
                endpoint_path=TRANSCRIPTIONS_ENDPOINT,
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
        )
