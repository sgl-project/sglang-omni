# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS HTTP server backed by the standalone MLX implementation.

    python -m sglang_omni_mlx.voxtral_tts.server --model-path DIR --model-name NAME --port 8000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

import uvicorn
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from sglang_omni_mlx.voxtral_tts.worker import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_VOICE,
    MAX_NEW_TOKENS_LIMIT,
    SpeechGenerationTimeoutError,
    SpeechQueueFullError,
    VoxtralTTSWorker,
)

logger = logging.getLogger(__name__)


def bad_request(message: str) -> JSONResponse:
    return JSONResponse({"detail": message}, status_code=400)


class SpeechBody(BaseModel):
    input: str = Field(min_length=1)
    voice: str = DEFAULT_VOICE
    model: str | None = None
    response_format: Literal["wav"] = "wav"
    max_new_tokens: int = Field(
        default=DEFAULT_MAX_NEW_TOKENS,
        ge=1,
        le=MAX_NEW_TOKENS_LIMIT,
    )

    @field_validator("input")
    @classmethod
    def validate_non_whitespace_input(cls, speech_input: str) -> str:
        if not speech_input.strip():
            raise ValueError("input must contain non-whitespace characters")
        else:
            return speech_input


def build_app(worker: VoxtralTTSWorker, model_name: str) -> Starlette:
    async def health(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "healthy",
                "running": True,
                "request_states": worker.request_states(),
            }
        )

    async def models(request: Request) -> JSONResponse:
        return JSONResponse(
            {"object": "list", "data": [{"id": model_name, "object": "model"}]}
        )

    async def speech(request: Request) -> Response:
        try:
            body = SpeechBody.model_validate(await request.json())
        except ValidationError as error:
            return JSONResponse(
                {
                    "detail": error.errors(
                        include_input=False,
                        include_context=False,
                    )
                },
                status_code=400,
            )
        except ValueError as error:
            return bad_request(str(error))
        if body.model is not None and body.model != model_name:
            return JSONResponse(
                {"detail": f"Unknown model: {body.model}"}, status_code=404
            )
        elif body.voice not in worker.voice_embeddings:
            return bad_request(f"Unknown Voxtral voice: {body.voice}")
        else:
            pass
        try:
            result = await worker.synthesize(
                body.input,
                body.voice,
                body.max_new_tokens,
            )
        except SpeechQueueFullError as error:
            return JSONResponse(
                {"detail": str(error)},
                status_code=429,
                headers={"Retry-After": "5"},
            )
        except SpeechGenerationTimeoutError as error:
            logger.warning(f"Voxtral request timed out: {error}")
            return JSONResponse(
                {"detail": "Voxtral speech generation timed out"},
                status_code=504,
            )
        except Exception:
            logger.exception("Voxtral speech generation failed")
            return JSONResponse(
                {"detail": "Voxtral speech generation failed"}, status_code=500
            )
        return Response(
            result.audio,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="speech.wav"',
                "X-Speech-Finish-Reason": result.finish_reason,
            },
        )

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/models", models),
            Route("/v1/audio/speech", speech, methods=["POST"]),
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
    worker = VoxtralTTSWorker(Path(arguments.model_path))
    logger.info(f"Loaded Voxtral TTS from {arguments.model_path}")
    uvicorn.run(
        build_app(worker, arguments.model_name),
        host=arguments.host,
        port=arguments.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
