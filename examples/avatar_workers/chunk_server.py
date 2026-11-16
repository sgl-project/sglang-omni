# SPDX-License-Identifier: Apache-2.0
"""The HTTP service an avatar_worker chunk worker runs in its model's own environment: standard library and numpy only, Python 3.9 and later.

The avatar_worker stage (sglang_omni/models/avatar_worker/renderer.py) drives
it one chunk at a time:

- GET /health: 200 once the model is loaded and warmed up, 503 before.
- GET /info: the geometry as JSON: fps, chunk_frames, lookahead_samples,
  listens, media, and width and height (jpeg_frame) or features (bvh_motion).
- POST /render: a 4-byte little-endian state length, the state (empty for a
  Session's first chunk), the agent's speech as 16 kHz int16 PCM of the chunk
  and the look-ahead, and, when the worker listens, the user's audio of as
  many samples. The answer's body is the next state followed by the chunk's
  frames, split by X-State-Length-Bytes and X-Payload-Lengths; X-Render-Ms
  tells the model's time.
- POST /release: the body is a Session's last state.

Models keep a Session's tensors on their GPUs and pass a handle as the state:
`SessionModel` keys them by handle, starts a Session on an empty state and
forgets Sessions idle for `idle_s`.
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

SAMPLE_RATE_HZ = 16000
PCM16_SCALE = 32768
STATE_LENGTH = struct.Struct("<I")

logger = logging.getLogger("avatar_worker")


class BadRequest(Exception):
    """The request, not the worker, is at fault: a 400."""


class ChunkModel:
    """What a worker serves; a model sets `info` and implements `render`."""

    info: dict = {}

    def chunk_samples(self) -> int:
        return round(self.info["chunk_frames"] * SAMPLE_RATE_HZ / self.info["fps"])

    def window_samples(self) -> int:
        return self.chunk_samples() + self.info["lookahead_samples"]

    def render(
        self, state: bytes, speech: np.ndarray, listen: Optional[np.ndarray]
    ) -> Tuple[bytes, List[bytes]]:
        raise NotImplementedError

    def release(self, state: bytes) -> None:
        pass


class SessionModel(ChunkModel):
    """A model whose Sessions live in the worker: the state on the wire is a handle; a subclass implements `start` and `step`."""

    idle_s = 120.0
    max_sessions = 4

    def __init__(self) -> None:
        self.sessions: Dict[bytes, Any] = {}
        self.used: Dict[bytes, float] = {}

    def start(self) -> Any:
        """A new Session's state."""
        raise NotImplementedError

    def step(
        self, session: Any, speech: np.ndarray, listen: Optional[np.ndarray]
    ) -> List[bytes]:
        """The chunk's frames; updates the Session's state in place."""
        raise NotImplementedError

    def forget(self, handle: bytes) -> None:
        self.sessions.pop(handle, None)
        self.used.pop(handle, None)

    def render(self, state, speech, listen):
        now = time.monotonic()
        for handle, used in list(self.used.items()):
            if now - used > self.idle_s:
                logger.info("forgetting idle session %s", handle.decode())
                self.forget(handle)
        if state not in self.sessions:
            if state:
                logger.warning("unknown session %s: starting it afresh", state.decode())
            while len(self.sessions) >= self.max_sessions:
                oldest = min(self.used, key=self.used.get)
                logger.warning("too many sessions: forgetting %s", oldest.decode())
                self.forget(oldest)
            state = state or uuid.uuid4().hex.encode()
            self.sessions[state] = self.start()
        self.used[state] = now
        return state, self.step(self.sessions[state], speech, listen)

    def release(self, state):
        self.forget(state)


def parse_render_body(
    body: bytes, window: int, listens: bool
) -> Tuple[bytes, np.ndarray, Optional[np.ndarray]]:
    if len(body) < STATE_LENGTH.size:
        raise BadRequest("the body starts with a 4-byte state length")
    (state_length,) = STATE_LENGTH.unpack_from(body)
    state = body[STATE_LENGTH.size : STATE_LENGTH.size + state_length]
    pcm = body[STATE_LENGTH.size + state_length :]
    tracks = 2 if listens else 1
    if len(state) != state_length or len(pcm) != 2 * window * tracks:
        raise BadRequest(
            f"expected a {state_length}-byte state and {tracks} x {window} int16 samples, got {len(state)} bytes and {len(pcm)} bytes of audio"
        )
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / PCM16_SCALE
    return state, audio[:window], (audio[window:] if listens else None)


def serve(load: Callable[[], ChunkModel], host: str, port: int) -> None:
    """Answer /health with 503 while `load` builds and warms the model, then serve it, one chunk at a time."""
    holder: dict = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            pass

        def answer(
            self, code, body, content_type="application/octet-stream", headers=None
        ):
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
            elif path == "/stats":
                # The model's own split of its last chunks' time, when it keeps one: (generate s, encode s) per chunk.
                timings = list(getattr(model, "timings", []))[-100:]
                self.answer(200, json.dumps(timings).encode(), "application/json")
            else:
                self.answer(404, b"not found")

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            model = holder.get("model")
            path = urlparse(self.path).path
            if model is None:
                self.answer(503, b"warming up")
                return
            if path == "/release":
                with lock:
                    model.release(body)
                self.answer(200, b"")
                return
            if path != "/render":
                self.answer(404, b"not found")
                return
            try:
                state, speech, listen = parse_render_body(
                    body, model.window_samples(), bool(model.info.get("listens"))
                )
                with lock:
                    start = time.perf_counter()
                    next_state, payloads = model.render(state, speech, listen)
                    render_ms = (time.perf_counter() - start) * 1000
            except BadRequest as error:
                self.answer(400, str(error).encode())
                return
            except Exception:
                logger.error("render failed:\n%s", traceback.format_exc())
                self.answer(500, traceback.format_exc().encode())
                return
            logger.debug("chunk %.1f ms", render_ms)
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
    parser.add_argument("--warmup-chunks", type=int, default=4)
    return parser


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )


def warm_up(model: ChunkModel, chunks: int) -> List[float]:
    """Render `chunks` chunks of one Session, alternating silence and noise, then release it; returns their times in ms."""
    rng = np.random.default_rng(0)
    window = model.window_samples()
    listens = bool(model.info.get("listens"))
    state = b""
    times = []
    for index in range(chunks):
        audio = (
            np.zeros(window, dtype=np.float32)
            if index % 2 == 0
            else (0.1 * rng.standard_normal(window)).astype(np.float32)
        )
        start = time.perf_counter()
        state, _ = model.render(state, audio, audio if listens else None)
        times.append((time.perf_counter() - start) * 1000)
        logger.info("warm-up chunk %d: %.1f ms", index, times[-1])
    model.release(state)
    return times


def jpeg_of(rgb: np.ndarray, quality: int = 80) -> bytes:
    """One uint8 HxWx3 frame as a JPEG (Pillow)."""
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="JPEG", quality=quality)
    return out.getvalue()
