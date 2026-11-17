# SPDX-License-Identifier: Apache-2.0
"""The chunk workers behind the avatar_worker stage, each an HTTP service in its model's own environment (examples/avatar_workers/chunk_server.py).

The wire, one request per chunk:

- GET /health: 200 once the model is loaded and warmed up.
- GET /info: the worker's geometry as JSON: fps, chunk_frames,
  lookahead_samples, listens, media ("jpeg_frame" or "bvh_motion"), and
  width and height or features.
- POST /render: a 4-byte little-endian state length, the state (empty for a
  Session's first chunk), the agent's speech as 16 kHz int16 PCM of the
  chunk and the look-ahead, and, when the worker listens, the user's audio of
  as many samples ending at the chunk's start. The answer's body is the next
  state followed by the chunk's frames; X-State-Length-Bytes and
  X-Payload-Lengths (comma separated) split it, X-Render-Ms tells the
  model's time.
- POST /release: the body is a Session's last state; the worker forgets what
  it keeps for it.

A worker that keeps a Session's state on its GPUs answers a handle of it, so
the stage pins each Session to the worker of its first chunk.
"""

from __future__ import annotations

import struct
import time
from collections.abc import Callable

import httpx
import numpy as np

from sglang_omni.models.avatar_worker.formats import (
    BVH_MOTION,
    JPEG_FRAME,
    MOTION,
    VIDEO,
)
from sglang_omni.models.avatar_worker.node import AvatarWorkerFactoryArgs, plugin_port
from sglang_omni.models.avtr.renderer import pcm16_of
from sglang_omni.models.avtr.session import (
    AvatarHooks,
    ChunkGeometry,
    RendererUnavailable,
)
from sglang_omni.proto.streams import SESSION_INPUT_SAMPLE_RATE_HZ
from sglang_omni.scheduling.session import DEFAULT_MAX_CONCURRENCY, SessionScheduler

STATE_LENGTH = struct.Struct("<I")
DEFAULT_SESSIONS_PER_RENDERER = 1
# Heavy video models take seconds for a chunk; a worker that has not answered for this long is treated as gone.
DEFAULT_RENDER_TIMEOUT_S = 30.0
# A worker loads and compiles its model for minutes; the stage waits this long for its /health before giving up.
DEFAULT_STARTUP_TIMEOUT_S = 900.0
STARTUP_PROBE_S = 5.0


class ChunkWorkerRenderer:
    """One chunk worker: the Session's state goes with each chunk and its successor comes back before the frames."""

    def __init__(self, client: httpx.Client, *, listens: bool) -> None:
        self.client = client
        self.listens = listens

    def render(
        self,
        state: bytes,
        speech: np.ndarray,
        listen: np.ndarray,
        frame: Callable[[bytes], None],
    ) -> bytes:
        body = STATE_LENGTH.pack(len(state)) + state + pcm16_of(speech)
        body += pcm16_of(listen) if self.listens else b""
        try:
            response = self.client.post(
                "/render",
                content=body,
                headers={"Content-Type": "application/octet-stream"},
            )
        except httpx.TransportError as error:
            raise RendererUnavailable(
                f"the avatar worker at {self.client.base_url} failed: {error!r}"
            ) from error
        if response.status_code >= 500:
            raise RendererUnavailable(
                f"the avatar worker at {self.client.base_url} answered {response.status_code}: {response.text[-300:]}"
            )
        else:
            response.raise_for_status()
        content = response.content
        state_bytes = int(response.headers["X-State-Length-Bytes"])
        lengths = [
            int(length)
            for length in response.headers.get("X-Payload-Lengths", "").split(",")
            if length
        ]
        if state_bytes + sum(lengths) != len(content):
            raise RendererUnavailable(
                f"the avatar worker at {self.client.base_url} sent {len(content)} bytes for a {state_bytes}-byte state and frames of {lengths}"
            )
        else:
            start = state_bytes
        for length in lengths:
            frame(content[start : start + length])
            start += length
        return content[:state_bytes]

    def release(self, state: bytes) -> None:
        try:
            self.client.post(
                "/release",
                content=state,
                headers={"Content-Type": "application/octet-stream"},
            )
        except httpx.HTTPError:
            pass

    def is_healthy(self) -> bool:
        try:
            return self.client.get("/health").status_code == 200
        except httpx.TransportError:
            return False


def geometry_of(factory: AvatarWorkerFactoryArgs) -> ChunkGeometry:
    params = plugin_port(factory).params
    fps = params["fps"]
    chunk_frames = factory.chunk_frames or 5
    lookahead_ms = params["reply_lookahead_ms"] - 1000 * chunk_frames / fps
    return ChunkGeometry(
        fps, chunk_frames, round(lookahead_ms * SESSION_INPUT_SAMPLE_RATE_HZ / 1000)
    )


def check_worker(url: str, info: dict, geometry: ChunkGeometry, factory) -> None:
    """The worker serves the stage's geometry and medium."""
    expected = {
        "fps": geometry.fps,
        "chunk_frames": geometry.chunk_frames,
        "lookahead_samples": geometry.lookahead_samples,
        "listens": bool(factory.listens),
        "media": factory.media,
    }
    if factory.media == JPEG_FRAME:
        port = plugin_port(factory).params
        expected |= {"width": port["width"], "height": port["height"]}
    else:
        pass
    wrong = {
        key: (value, info.get(key))
        for key, value in expected.items()
        if info.get(key) != value
    }
    if wrong:
        raise ValueError(
            f"the avatar worker at {url} serves another geometry than the stage's: {wrong} (stage, worker)"
        )
    else:
        pass


def wait_until_healthy(client: httpx.Client, timeout_s: float) -> None:
    """Wait for a worker that is still loading its model: its /health answers 503, or nothing, until it serves."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if client.get("/health").status_code == 200:
                return
            else:
                pass
        except httpx.TransportError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"the avatar worker at {client.base_url} did not answer /health within {timeout_s:.0f} s"
            )
        else:
            time.sleep(STARTUP_PROBE_S)


def create_scheduler(
    *,
    renderer_urls: list[str],
    speech_sample_rate_hz: int,
    input_sample_rate_hz: int = SESSION_INPUT_SAMPLE_RATE_HZ,
    reply_stream: bool = False,
    sessions_per_renderer: int = DEFAULT_SESSIONS_PER_RENDERER,
    render_timeout_s: float = DEFAULT_RENDER_TIMEOUT_S,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
    **geometry_args,
) -> SessionScheduler:
    factory = AvatarWorkerFactoryArgs(**geometry_args)
    geometry = geometry_of(factory)
    renderers = []
    for url in renderer_urls:
        client = httpx.Client(base_url=url, timeout=render_timeout_s)
        wait_until_healthy(client, startup_timeout_s)
        listed = client.get("/info")
        listed.raise_for_status()
        check_worker(url, listed.json(), geometry, factory)
        renderers.append(ChunkWorkerRenderer(client, listens=bool(factory.listens)))
    stream = MOTION if factory.media == BVH_MOTION else VIDEO
    return SessionScheduler(
        AvatarHooks(
            renderers,
            speech_sample_rate_hz,
            input_sample_rate_hz=input_sample_rate_hz,
            reply_stream=reply_stream,
            geometry=geometry,
            stream=stream,
            media_format=factory.media,
            pinned=True,
        ),
        max_open_sessions=len(renderers) * sessions_per_renderer,
        max_concurrency=DEFAULT_MAX_CONCURRENCY,
    )


__all__ = [
    "ChunkWorkerRenderer",
    "create_scheduler",
    "geometry_of",
    "wait_until_healthy",
]
