# SPDX-License-Identifier: Apache-2.0
"""Standalone MOSS-Transcribe-Diarize server on MLX.

    python -m sglang_omni_mlx.moss_transcribe_diarize.server --model-path DIR --model-name NAME --port 8000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlx.core as mx
import uvicorn
from starlette.applications import Starlette
from starlette.datastructures import FormData
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from sglang_omni_mlx.moss_transcribe_diarize.audio import decode_audio_wav
from sglang_omni_mlx.moss_transcribe_diarize.transcriber import TranscriptionOptions
from sglang_omni_mlx.moss_transcribe_diarize.worker import TranscriptionWorker
from sglang_omni_mlx.serving import (
    bad_request,
    form_flag,
    info_routes,
    text_done_event,
    transcription_response,
    uploaded_samples,
)

logger = logging.getLogger(__name__)


def form_text(form: FormData, name: str) -> str | None:
    value = form.get(name)
    return value if isinstance(value, str) else None


def transcription_options(form: FormData) -> TranscriptionOptions:
    max_new_tokens = form_text(form, "max_new_tokens")
    requested_tokens = int(max_new_tokens) if max_new_tokens else None
    if requested_tokens is not None and requested_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    else:
        pass
    temperature = float(form_text(form, "temperature") or 0.0)
    repetition_penalty = float(form_text(form, "repetition_penalty") or 1.0)
    if temperature != 0.0 or repetition_penalty != 1.0:
        raise ValueError("MOSS-TD MLX supports temperature=0 and repetition_penalty=1")
    else:
        pass
    return TranscriptionOptions(
        prompt=form_text(form, "prompt"), max_new_tokens=requested_tokens
    )


def build_app(worker: TranscriptionWorker, model_name: str) -> Starlette:
    async def transcriptions(request: Request) -> Response:
        form = await request.form()
        try:
            samples = await uploaded_samples(form, decode_audio_wav)
            options = transcription_options(form)
        except ValueError as error:
            return bad_request(str(error))
        return await transcription_response(
            worker,
            samples,
            options,
            form_flag(form.get("stream")),
            text_done_event,
        )

    return Starlette(
        routes=[
            *info_routes(worker, model_name),
            Route("/v1/audio/transcriptions", transcriptions, methods=["POST"]),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    mx.set_cache_limit(0)
    worker = TranscriptionWorker(Path(arguments.model_path))
    logger.info(f"Loaded MOSS-Transcribe-Diarize from {arguments.model_path}")
    uvicorn.run(
        build_app(worker, arguments.model_name),
        host=arguments.host,
        port=arguments.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
