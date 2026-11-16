# SPDX-License-Identifier: Apache-2.0
"""The HTTP service a speech-avatar worker runs in its model's own environment: standard library and numpy only, Python 3.9 and later.

The speech_avatar stage of sglang-omni (sglang_omni/models/speech_avatar)
drives a worker one chunk at a time:

- GET /health: 200 once the model is loaded and warmed up, 503 before.
- GET /info: the chunk geometry as JSON: kind ("video" or "motion"), fps,
  chunk_frames, context_samples, lookahead_samples, and width and height
  (video) or features (motion).
- GET /idle: the picture a video worker shows while the agent is silent, as a
  JPEG.
- POST /render: the body is a 4-byte little-endian state length, the state
  (empty for a Session's first chunk) and the agent's speech as 16 kHz int16
  PCM, context_samples before the chunk, the chunk and lookahead_samples
  after it. The answer's body is the next state followed by the chunk's
  payloads, one JPEG per frame (video) or one float32 [chunk_frames,
  features] array (motion); X-State-Length-Bytes and X-Payload-Lengths (comma
  separated) split it, and X-Render-Ms tells the time the model took.

The state carries everything a Session's next chunk needs, so any worker of
the same model renders any Session's chunk. A worker renders one chunk at a
time.
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

SAMPLE_RATE_HZ = 16000
PCM16_SCALE = 32768
STATE_LENGTH = struct.Struct("<I")

logger = logging.getLogger("speech_avatar_worker")


class BadRequest(Exception):
    """The request, not the worker, is at fault: a 400."""


class ChunkModel:
    """What a worker serves; a model sets `info` and implements `render`."""

    info: dict = {}

    def idle_jpeg(self) -> Optional[bytes]:
        return None

    def render(self, state: bytes, speech: np.ndarray) -> Tuple[bytes, List[bytes]]:
        """The next state and the chunk's payloads from the previous state and the float32 speech window."""
        raise NotImplementedError

    def window_samples(self) -> int:
        info = self.info
        return (
            info["context_samples"]
            + info["chunk_frames"] * SAMPLE_RATE_HZ // info["fps"]
            + info["lookahead_samples"]
        )


def parse_render_body(body: bytes, window: int) -> Tuple[bytes, np.ndarray]:
    if len(body) < STATE_LENGTH.size:
        raise BadRequest("the body starts with a 4-byte state length")
    else:
        (state_length,) = STATE_LENGTH.unpack_from(body)
    state = body[STATE_LENGTH.size : STATE_LENGTH.size + state_length]
    pcm = body[STATE_LENGTH.size + state_length :]
    if len(state) != state_length or len(pcm) != 2 * window:
        raise BadRequest(
            f"expected a {state_length}-byte state and {window} int16 samples, got {len(state)} bytes and {len(pcm)} bytes of audio"
        )
    else:
        return state, np.frombuffer(pcm, dtype="<i2").astype(np.float32) / PCM16_SCALE


def serve(load: Callable[[], ChunkModel], host: str, port: int) -> None:
    """Answer /health with 503 while `load` builds and warms the model, then serve it."""
    holder: dict = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            pass

        def answer(
            self,
            code: int,
            body: bytes,
            content_type: str = "application/octet-stream",
            headers: Optional[dict] = None,
        ) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            model = holder.get("model")
            if path == "/health":
                self.answer(200 if model else 503, b"ok" if model else b"warming up")
            elif model is None:
                self.answer(503, b"warming up")
            elif path == "/info":
                self.answer(200, json.dumps(model.info).encode(), "application/json")
            elif path == "/idle" and model.idle_jpeg() is not None:
                self.answer(200, model.idle_jpeg(), "image/jpeg")
            else:
                self.answer(404, b"not found")

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            model = holder.get("model")
            if urlparse(self.path).path != "/render":
                self.answer(404, b"not found")
                return
            elif model is None:
                self.answer(503, b"warming up")
                return
            else:
                pass
            try:
                state, speech = parse_render_body(body, model.window_samples())
                with lock:
                    start = time.perf_counter()
                    next_state, payloads = model.render(state, speech)
                    render_ms = (time.perf_counter() - start) * 1000
                logger.info("render %.1f ms", render_ms)
            except BadRequest as error:
                self.answer(400, str(error).encode())
                return
            except Exception:
                logger.error("render failed:\n%s", traceback.format_exc())
                self.answer(500, traceback.format_exc().encode())
                return
            self.answer(
                200,
                next_state + b"".join(payloads),
                headers={
                    "X-State-Length-Bytes": str(len(next_state)),
                    "X-Payload-Lengths": ",".join(str(len(p)) for p in payloads),
                    "X-Render-Ms": f"{render_ms:.1f}",
                },
            )

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info("listening on %s:%d, loading the model", host, port)
    start = time.perf_counter()
    holder["model"] = load()
    logger.info(
        "ready after %.1f s: %s", time.perf_counter() - start, holder["model"].info
    )
    threading.Event().wait()


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--warmup-chunks",
        type=int,
        default=4,
        help="Chunks of silence and of noise rendered before /health answers 200, so compilation and kernel selection happen before the first Session.",
    )
    return parser


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )


def warm_up(model: ChunkModel, chunks: int) -> List[float]:
    """Render `chunks` chunks from a fresh state, alternating silence and noise, and return their times in ms."""
    rng = np.random.default_rng(0)
    window = model.window_samples()
    state = b""
    times = []
    for index in range(chunks):
        speech = (
            np.zeros(window, dtype=np.float32)
            if index % 2 == 0
            else (0.1 * rng.standard_normal(window)).astype(np.float32)
        )
        start = time.perf_counter()
        state, _ = model.render(state, speech)
        times.append((time.perf_counter() - start) * 1000)
        logger.info("warm-up chunk %d: %.1f ms", index, times[-1])
    return times
