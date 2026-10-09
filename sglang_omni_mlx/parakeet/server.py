# SPDX-License-Identifier: Apache-2.0
"""Single-process Parakeet server on MLX: transcriptions over HTTP.

    python -m sglang_omni_mlx.parakeet.server --model-path DIR --model-name NAME --port 8000
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

from sglang_omni_mlx.parakeet.transcriber import (
    DTYPES,
    TranscriptionOptions,
    require_supported_duration,
)
from sglang_omni_mlx.parakeet.worker import TranscriptionWorker
from sglang_omni_mlx.serving import (
    bad_request,
    form_flag,
    info_routes,
    text_done_event,
    transcription_response,
    uploaded_samples,
)

logger = logging.getLogger(__name__)


def transcription_options(form: FormData) -> TranscriptionOptions:
    """Reject the fields greedy Parakeet decoding cannot honor.

    A language field is accepted and ignored: multilingual checkpoints
    detect the language themselves.
    """
    temperature = form.get("temperature")
    if isinstance(temperature, str) and temperature.strip():
        try:
            value = float(temperature)
        except ValueError as error:
            raise ValueError("temperature must be a number") from error
        if value != 0.0:
            raise ValueError("Parakeet decodes greedily; temperature must be 0")
        else:
            pass
    else:
        pass
    prompt = form.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        raise ValueError("Parakeet does not take a text prompt")
    else:
        pass
    if form.get("max_new_tokens"):
        raise ValueError("Parakeet does not take max_new_tokens")
    else:
        pass
    return TranscriptionOptions()


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
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Freed MLX buffers go back to the system: an idle server holds only the model.
    mx.set_cache_limit(0)
    worker = TranscriptionWorker(Path(arguments.model_path), arguments.dtype)
    logger.info(f"Loaded Parakeet from {arguments.model_path} ({arguments.dtype})")
    uvicorn.run(
        build_app(worker, arguments.model_name),
        host=arguments.host,
        port=arguments.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
