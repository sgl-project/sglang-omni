# SPDX-License-Identifier: Apache-2.0
"""Single-process Fun-ASR-Nano server on MLX: transcriptions over HTTP.

    python -m sglang_omni_mlx.fun_asr.server --model-path DIR --model-name NAME --port 8000
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

from sglang_omni_mlx.fun_asr.transcriber import (
    MAX_OUTPUT_TOKENS,
    TranscriptionOptions,
    normalize_language,
    require_supported_duration,
)
from sglang_omni_mlx.fun_asr.worker import TranscriptionWorker
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
    """Request fields to options; the prompt carries comma-separated hotwords."""
    prompt = form_text(form, "prompt") or ""
    max_new_tokens = form_text(form, "max_new_tokens")
    if max_new_tokens:
        requested = int(max_new_tokens)
        if not 1 <= requested <= MAX_OUTPUT_TOKENS:
            raise ValueError(
                f"max_new_tokens must be between 1 and {MAX_OUTPUT_TOKENS}"
            )
        else:
            pass
    else:
        requested = None
    return TranscriptionOptions(
        language=normalize_language(form_text(form, "language") or ""),
        itn=form_flag(form.get("itn"), default=True),
        hotwords=tuple(term.strip() for term in prompt.split(",") if term.strip()),
        max_new_tokens=requested,
    )


def build_app(worker: TranscriptionWorker, model_name: str) -> Starlette:
    async def transcriptions(request: Request) -> Response:
        form = await request.form()
        try:
            samples = await uploaded_samples(form)
            require_supported_duration(samples)
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
    # Freed MLX buffers go back to the system: an idle server holds only the model.
    mx.set_cache_limit(0)
    worker = TranscriptionWorker(Path(arguments.model_path))
    logger.info(f"Loaded Fun-ASR from {arguments.model_path}")
    uvicorn.run(
        build_app(worker, arguments.model_name),
        host=arguments.host,
        port=arguments.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
