# SPDX-License-Identifier: Apache-2.0
"""The HTTP client of a speech-avatar worker (examples/speech_avatar_workers/chunk_server.py) and the schedulers of the two stages over a pool of them."""

from __future__ import annotations

import struct
import time

import httpx
import numpy as np

from sglang_omni.models.dyadic_audio.timeline import PCM16_SCALE
from sglang_omni.models.speech_avatar.node import (
    DEFAULT_MOTION_CHUNK_MS,
    DEFAULT_MOTION_FPS,
    DEFAULT_RENDER_LEAD_MS,
    DEFAULT_VIDEO_CHUNK_MS,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_SIZE,
)
from sglang_omni.models.speech_avatar.session import (
    REPLY_QUIET_MS,
    ChunkGeometry,
    SpeechMotionHooks,
    SpeechVideoHooks,
    WorkerUnavailable,
)
from sglang_omni.proto.streams import SESSION_INPUT_SAMPLE_RATE_HZ
from sglang_omni.scheduling.session import DEFAULT_MAX_CONCURRENCY, SessionScheduler

RENDER_PATH = "/render"
# A chunk renders in well under a second on an H100; a worker that has not answered for this long is treated as gone.
RENDER_TIMEOUT_S = 10.0
# A worker compiles and warms its model before /health answers; the stage waits this long for it at start-up.
DEFAULT_STARTUP_TIMEOUT_S = 900.0
STARTUP_PROBE_S = 5.0
DEFAULT_SESSIONS_PER_WORKER = 1
STATE_LENGTH = struct.Struct("<I")


def pcm16_of(samples: np.ndarray) -> bytes:
    return (
        np.clip(np.round(samples * PCM16_SCALE), -PCM16_SCALE, PCM16_SCALE - 1)
        .astype("<i2")
        .tobytes()
    )


class HttpChunkWorker:
    """One worker service; the Session's state goes with each chunk and its successor comes back first in the body."""

    def __init__(self, client: httpx.Client, info: dict) -> None:
        self.client = client
        self.info = info
        self.geometry = ChunkGeometry(
            fps=info["fps"],
            chunk_frames=info["chunk_frames"],
            context_samples=info["context_samples"],
            lookahead_samples=info["lookahead_samples"],
        )

    @classmethod
    def connect(
        cls, url: str, startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S
    ) -> HttpChunkWorker:
        """Wait for the worker at `url` to be warm, then read its geometry."""
        client = httpx.Client(base_url=url, timeout=RENDER_TIMEOUT_S)
        deadline = time.monotonic() + startup_timeout_s
        while True:
            try:
                if client.get("/health").status_code == 200:
                    break
                else:
                    pass
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"the speech avatar worker at {url} was not ready within {startup_timeout_s} s"
                )
            else:
                time.sleep(STARTUP_PROBE_S)
        answer = client.get("/info")
        answer.raise_for_status()
        return cls(client, answer.json())

    def idle_picture(self) -> bytes:
        answer = self.client.get("/idle")
        answer.raise_for_status()
        return answer.content

    def render(self, state: bytes, speech: np.ndarray) -> tuple[bytes, list[bytes]]:
        body = STATE_LENGTH.pack(len(state)) + state + pcm16_of(speech)
        try:
            answer = self.client.post(RENDER_PATH, content=body)
        except httpx.TransportError as error:
            raise WorkerUnavailable(
                f"the speech avatar worker at {self.client.base_url} failed: {error!r}"
            ) from error
        if answer.status_code >= 500:
            raise WorkerUnavailable(
                f"the speech avatar worker at {self.client.base_url} answered {answer.status_code}: {answer.text[-500:]}"
            )
        else:
            answer.raise_for_status()
        state_length = int(answer.headers["X-State-Length-Bytes"])
        lengths = [
            int(length)
            for length in answer.headers["X-Payload-Lengths"].split(",")
            if length
        ]
        body = answer.content
        if len(body) != state_length + sum(lengths):
            raise WorkerUnavailable(
                f"the speech avatar worker at {self.client.base_url} sent {len(body)} bytes, not {state_length + sum(lengths)}"
            )
        else:
            payloads = []
            offset = state_length
        for length in lengths:
            payloads.append(body[offset : offset + length])
            offset += length
        return body[:state_length], payloads

    def is_healthy(self) -> bool:
        try:
            return self.client.get("/health").status_code == 200
        except httpx.TransportError:
            return False


def connect_workers(
    urls: list[str], kind: str, fps: int, startup_timeout_s: float
) -> list[HttpChunkWorker]:
    """The pool's workers, which all serve one geometry of `kind` at `fps`."""
    workers = [HttpChunkWorker.connect(url, startup_timeout_s) for url in urls]
    for url, worker in zip(urls, workers):
        if worker.info["kind"] != kind or worker.geometry.fps != fps:
            raise ValueError(
                f"the speech avatar worker at {url} renders {worker.info['kind']} at {worker.geometry.fps} fps, not {kind} at {fps} fps"
            )
        elif worker.geometry != workers[0].geometry:
            raise ValueError(
                f"the speech avatar worker at {url} renders {worker.geometry}, unlike {urls[0]}: {workers[0].geometry}"
            )
        else:
            pass
    return workers


def check_chunk(worker: HttpChunkWorker, chunk_ms: float, lookahead_ms: float) -> None:
    """The port the stage announced must describe the workers' chunks."""
    geometry = worker.geometry
    if (
        abs(geometry.chunk_ms - chunk_ms) > 0.5
        or abs(geometry.lookahead_ms - lookahead_ms) > 0.5
    ):
        raise ValueError(
            f"the workers render {geometry.chunk_ms} ms chunks with {geometry.lookahead_ms} ms of look-ahead, but the stage is configured for {chunk_ms} and {lookahead_ms}"
        )
    else:
        pass


def create_video_scheduler(
    *,
    worker_urls: list[str],
    speech_sample_rate_hz: int,
    fps: int = DEFAULT_VIDEO_FPS,
    width: int = DEFAULT_VIDEO_SIZE,
    height: int = DEFAULT_VIDEO_SIZE,
    chunk_ms: float = DEFAULT_VIDEO_CHUNK_MS,
    lookahead_ms: float = 0.0,
    render_lead_ms: float = DEFAULT_RENDER_LEAD_MS,
    tail_ms: float = 0.0,
    reply_quiet_ms: float = REPLY_QUIET_MS,
    reply_starts_chunk: bool = False,
    input_sample_rate_hz: int = SESSION_INPUT_SAMPLE_RATE_HZ,
    sessions_per_worker: int = DEFAULT_SESSIONS_PER_WORKER,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
) -> SessionScheduler:
    workers = connect_workers(worker_urls, "video", fps, startup_timeout_s)
    check_chunk(workers[0], chunk_ms, lookahead_ms)
    for url, worker in zip(worker_urls, workers):
        if (worker.info["width"], worker.info["height"]) != (width, height):
            raise ValueError(
                f"the speech avatar worker at {url} renders {worker.info['width']}x{worker.info['height']}, not {width}x{height}"
            )
        else:
            pass
    return SessionScheduler(
        SpeechVideoHooks(
            workers,
            workers[0].geometry,
            speech_sample_rate_hz,
            idle_picture=workers[0].idle_picture(),
            render_lead_ms=render_lead_ms,
            tail_ms=tail_ms,
            reply_quiet_ms=reply_quiet_ms,
            reply_starts_chunk=reply_starts_chunk,
            input_sample_rate_hz=input_sample_rate_hz,
        ),
        max_open_sessions=len(workers) * sessions_per_worker,
        max_concurrency=DEFAULT_MAX_CONCURRENCY,
    )


def create_motion_scheduler(
    *,
    worker_urls: list[str],
    speech_sample_rate_hz: int,
    fps: int = DEFAULT_MOTION_FPS,
    chunk_ms: float = DEFAULT_MOTION_CHUNK_MS,
    lookahead_ms: float = 0.0,
    input_sample_rate_hz: int = SESSION_INPUT_SAMPLE_RATE_HZ,
    sessions_per_worker: int = DEFAULT_SESSIONS_PER_WORKER,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
) -> SessionScheduler:
    workers = connect_workers(worker_urls, "motion", fps, startup_timeout_s)
    check_chunk(workers[0], chunk_ms, lookahead_ms)
    return SessionScheduler(
        SpeechMotionHooks(
            workers,
            workers[0].geometry,
            speech_sample_rate_hz,
            input_sample_rate_hz=input_sample_rate_hz,
        ),
        max_open_sessions=len(workers) * sessions_per_worker,
        max_concurrency=DEFAULT_MAX_CONCURRENCY,
    )


__all__ = [
    "HttpChunkWorker",
    "create_motion_scheduler",
    "create_video_scheduler",
    "pcm16_of",
]
