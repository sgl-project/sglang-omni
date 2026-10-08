# SPDX-License-Identifier: Apache-2.0
"""Single-process Qwen3-ASR server on MLX: transcriptions over HTTP, realtime over a socket.

    python -m sglang_omni_mlx.qwen3_asr.server --model-path DIR --model-name NAME --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import mlx.core as mx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from sglang_omni_mlx.qwen3_asr.audio import AudioLayout, decode_wav
from sglang_omni_mlx.qwen3_asr.realtime import (
    RealtimeSession,
    RealtimeSettings,
    realtime_settings,
)
from sglang_omni_mlx.qwen3_asr.transcriber import (
    TranscriptionCancelled,
    TranscriptionOptions,
    TranscriptionResult,
    normalize_language,
)
from sglang_omni_mlx.qwen3_asr.worker import TranscriptionWorker

logger = logging.getLogger(__name__)

TRUE_FORM_VALUES = frozenset({"1", "true", "yes", "on"})


def form_flag(value: object) -> bool:
    return isinstance(value, str) and value.strip().casefold() in TRUE_FORM_VALUES


def bad_request(message: str) -> JSONResponse:
    return JSONResponse({"detail": message}, status_code=400)


def done_event(
    result: TranscriptionResult, include_generation_metadata: bool
) -> dict[str, object]:
    event: dict[str, object] = {"type": "transcript.text.done", "text": result.text}
    if include_generation_metadata:
        event["generation_metadata"] = {
            "generated_token_count": result.generated_token_count,
            "language": result.language,
            "finish_reason": result.finish_reason.value,
        }
    else:
        pass
    return event


def build_app(
    worker: TranscriptionWorker, model_name: str, settings: RealtimeSettings
) -> Starlette:
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

    async def transcriptions(request: Request) -> Response:
        form = await request.form()
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            return bad_request("file is required")
        else:
            pass
        try:
            samples = decode_wav(await upload.read())
            language = form.get("language")
            options = TranscriptionOptions(
                language=(
                    normalize_language(language) if isinstance(language, str) else None
                ),
                context=(
                    form.get("prompt") if isinstance(form.get("prompt"), str) else None
                ),
                max_new_tokens=(
                    int(form["max_new_tokens"]) if form.get("max_new_tokens") else None
                ),
                stop_at_end_of_text=form_flag(form.get("stop_at_end_of_text")),
                stop_on_token_loop=form_flag(form.get("stop_on_token_loop")),
                layout=AudioLayout(
                    form.get("audio_layout") or AudioLayout.REFERENCE.value
                ),
            )
        except ValueError as error:
            return bad_request(str(error))
        stream = form_flag(form.get("stream"))
        include_generation_metadata = form_flag(form.get("include_generation_metadata"))
        if include_generation_metadata and not stream:
            return bad_request("include_generation_metadata requires stream=true")
        else:
            pass
        cancel = threading.Event()
        if not stream:
            result = await worker.transcribe(samples, options, cancel)
            return JSONResponse({"text": result.text})
        else:
            pass

        async def events() -> AsyncIterator[str]:
            try:
                result = await worker.transcribe(samples, options, cancel)
                yield f"data: {json.dumps(done_event(result, include_generation_metadata), ensure_ascii=False)}\n\n"
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

    async def realtime(websocket: WebSocket) -> None:
        await websocket.accept()

        async def send(event: dict[str, object]) -> None:
            await websocket.send_text(json.dumps(event, ensure_ascii=False))

        session = RealtimeSession(worker=worker, settings=settings, send=send)
        try:
            while True:
                try:
                    message = json.loads(await websocket.receive_text())
                except json.JSONDecodeError:
                    await session.send_error(
                        "invalid_request_error", "invalid_json", "Events must be JSON."
                    )
                    continue
                if not isinstance(message, dict) or not await session.handle(message):
                    break
                else:
                    pass
            await websocket.close()
        except WebSocketDisconnect:
            pass
        except Exception as error:
            # The socket closes; the type alone is logged, never audio or text.
            logger.error(f"realtime session failed: {type(error).__name__}")
        finally:
            session.close()

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/models", models),
            Route("/v1/audio/transcriptions", transcriptions, methods=["POST"]),
            WebSocketRoute("/v1/realtime", realtime),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--decode-interval-ms", type=int, default=1000)
    parser.add_argument("--first-decode-ms", type=int, default=100)
    parser.add_argument("--max-segment-seconds", type=float, default=30.0)
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Freed MLX buffers go back to the system: an idle server holds only the model.
    mx.set_cache_limit(0)
    worker = TranscriptionWorker(Path(arguments.model_path))
    logger.info(f"Loaded Qwen3-ASR from {arguments.model_path}")
    settings = realtime_settings(
        arguments.decode_interval_ms,
        arguments.first_decode_ms,
        arguments.max_segment_seconds,
    )
    uvicorn.run(
        build_app(worker, arguments.model_name, settings),
        host=arguments.host,
        port=arguments.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
