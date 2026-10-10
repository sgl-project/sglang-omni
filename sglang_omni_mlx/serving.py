# SPDX-License-Identifier: Apache-2.0
"""The HTTP surface every model's server shares: health, models, uploads, JSON or SSE replies."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import AsyncIterator, Callable
from typing import Protocol, TypeVar

import numpy as np
from starlette.datastructures import FormData
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from sglang_omni_mlx.transcription import TranscriptionCancelled
from sglang_omni_mlx.wav import decode_wav

logger = logging.getLogger(__name__)

TRUE_FORM_VALUES = frozenset({"1", "true", "yes", "on"})

OptionsT = TypeVar("OptionsT", contravariant=True)
ResultT = TypeVar("ResultT")


class Worker(Protocol[OptionsT, ResultT]):
    def request_states(self) -> dict[str, int]: ...

    async def transcribe(
        self, samples: np.ndarray, options: OptionsT, cancel: threading.Event
    ) -> ResultT: ...


class TranscriptText(Protocol):
    @property
    def text(self) -> str: ...


def form_flag(value: object, default: bool = False) -> bool:
    """A true form value; a missing or blank field is the default, anything else false."""
    if not isinstance(value, str) or not value.strip():
        return default
    else:
        return value.strip().casefold() in TRUE_FORM_VALUES


def bad_request(message: str) -> JSONResponse:
    return JSONResponse({"detail": message}, status_code=400)


async def uploaded_samples(
    form: FormData, decoder: Callable[[bytes], np.ndarray] = decode_wav
) -> np.ndarray:
    upload = form.get("file")
    if upload is None or isinstance(upload, str):
        raise ValueError("file is required")
    else:
        pass
    return decoder(await upload.read())


def info_routes(worker: Worker[OptionsT, ResultT], model_name: str) -> list[Route]:
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

    return [Route("/health", health), Route("/v1/models", models)]


def text_done_event(result: TranscriptText) -> dict[str, object]:
    return {"type": "transcript.text.done", "text": result.text}


async def transcription_response(
    worker: Worker[OptionsT, ResultT],
    samples: np.ndarray,
    options: OptionsT,
    stream: bool,
    done_event: Callable[[ResultT], dict[str, object]],
) -> Response:
    """The transcript as JSON, or as one SSE done event followed by [DONE]."""
    cancel = threading.Event()
    if not stream:
        result = await worker.transcribe(samples, options, cancel)
        return JSONResponse({"text": result.text})
    else:
        pass

    async def events() -> AsyncIterator[str]:
        try:
            result = await worker.transcribe(samples, options, cancel)
            yield f"data: {json.dumps(done_event(result), ensure_ascii=False)}\n\n"
        except TranscriptionCancelled:
            return
        except (ValueError, RuntimeError):
            logger.exception("transcription failed")
            failure = {
                "type": "error",
                "error": {
                    "type": "server_error",
                    "code": "transcription_failed",
                    "message": "Transcription failed.",
                },
            }
            yield f"data: {json.dumps(failure)}\n\n"
        finally:
            # A client that disconnects stops the decode it was waiting for.
            cancel.set()
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
