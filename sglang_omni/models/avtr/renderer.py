# SPDX-License-Identifier: Apache-2.0
"""The AVTR-1 renderer service behind the avatar stage: a pool of its workers, one render request per chunk, each frame encoded as JPEG here or, by a service that does it, there."""

from __future__ import annotations

import io
from collections.abc import Callable

import httpx
import numpy as np
from PIL import Image

from sglang_omni.models.avtr.formats import DEFAULT_HEIGHT, DEFAULT_WIDTH
from sglang_omni.models.avtr.session import (
    CHUNK_SAMPLES,
    AvatarHooks,
    RendererUnavailable,
)
from sglang_omni.models.dyadic_audio.timeline import PCM16_SCALE
from sglang_omni.proto.streams import SESSION_INPUT_SAMPLE_RATE_HZ
from sglang_omni.scheduling.session import DEFAULT_MAX_CONCURRENCY, SessionScheduler

RENDER_PATH = "/process-audio-v3"
# A worker renders one chunk at a time (its service runs each request on its one TensorRT pipeline); a profile sets how many Sessions one keeps at real time on its GPU.
DEFAULT_SESSIONS_PER_RENDERER = 1
# Chunks the stage keeps in flight at one worker: more than one only pays off on a service that renders concurrent chunks of different Sessions in one batch.
DEFAULT_REQUESTS_PER_RENDERER = 1
DEFAULT_JPEG_QUALITY = 75
# A chunk renders in about 55 ms on an H100; a worker that has not answered for this long is treated as gone.
RENDER_TIMEOUT_S = 10.0
# The service packs frames as BT.601 limited-range I420 (luma 16-235, chroma 16-240); JPEG stores full-range YCbCr.
CODES = np.arange(256, dtype=np.float32)
FULL_RANGE_LUMA = np.clip((CODES - 16) * 255 / 219 + 0.5, 0, 255).astype(np.uint8)
FULL_RANGE_CHROMA = np.clip((CODES - 128) * 255 / 224 + 128.5, 0, 255).astype(np.uint8)


def pcm16_of(samples: np.ndarray) -> bytes:
    """Float samples as the service's int16 PCM, the inverse of its division by 32768."""
    return (
        np.clip(np.round(samples * PCM16_SCALE), -PCM16_SCALE, PCM16_SCALE - 1)
        .astype("<i2")
        .tobytes()
    )


def crop_box(
    height: int, width: int, out_height: int, out_width: int
) -> tuple[float, float, float, float]:
    """The centered region of a height x width picture with the output's aspect ratio."""
    scale = min(width / out_width, height / out_height)
    left = (width - out_width * scale) / 2
    top = (height - out_height * scale) / 2
    return (left, top, left + out_width * scale, top + out_height * scale)


def jpeg_of_i420(
    frame: bytes,
    height: int,
    width: int,
    out_height: int,
    out_width: int,
    quality: int,
) -> bytes:
    """One I420 frame of the service as a JPEG of the output size, center-cropped to its aspect ratio; each plane is scaled before the range conversion, so the work follows the output size."""
    luma_bytes = height * width
    chroma_bytes = luma_bytes // 4
    box = crop_box(height, width, out_height, out_width)
    half_box = tuple(edge / 2 for edge in box)
    size = (out_width, out_height)
    luma = (
        Image.frombuffer("L", (width, height), frame[:luma_bytes], "raw", "L", 0, 1)
        .resize(size, Image.Resampling.BILINEAR, box=box)
        .point(FULL_RANGE_LUMA.tolist())
    )
    chroma = [
        Image.frombuffer(
            "L",
            (width // 2, height // 2),
            frame[start : start + chroma_bytes],
            "raw",
            "L",
            0,
            1,
        )
        .resize(size, Image.Resampling.BILINEAR, box=half_box)
        .point(FULL_RANGE_CHROMA.tolist())
        for start in (luma_bytes, luma_bytes + chroma_bytes)
    ]
    encoded = io.BytesIO()
    Image.merge("YCbCr", [luma, *chroma]).save(encoded, format="JPEG", quality=quality)
    return encoded.getvalue()


class AVTR1Renderer:
    """One worker of the AVTR-1 renderer service; the Session's state blob goes with each chunk and its successor comes back first in the body."""

    def __init__(
        self,
        client: httpx.Client,
        avatar: str,
        background: str,
        *,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        service_jpeg: bool = False,
    ) -> None:
        self.client = client
        self.params = {
            "avatar_id": avatar,
            "bg_id": background,
            "pixel_format": "yuv_i420",
        }
        if service_jpeg:
            self.params.update(
                {
                    "out_format": "jpeg",
                    "out_width": str(width),
                    "out_height": str(height),
                    "jpeg_quality": str(jpeg_quality),
                }
            )
        else:
            pass
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality

    def render(
        self,
        state: bytes,
        speech: np.ndarray,
        listen: np.ndarray,
        frame: Callable[[bytes], None],
    ) -> bytes:
        files = {
            "current_chunk": pcm16_of(speech[:CHUNK_SAMPLES]),
            "future_chunk": pcm16_of(speech[CHUNK_SAMPLES:]),
            "current_chunk_listen": pcm16_of(listen[:CHUNK_SAMPLES]),
            "future_chunk_listen": pcm16_of(listen[CHUNK_SAMPLES:]),
            "state": state,
        }
        try:
            with self.client.stream(
                "POST", RENDER_PATH, params=self.params, files=files
            ) as response:
                if response.status_code >= 500:
                    raise RendererUnavailable(
                        f"the AVTR-1 renderer at {self.client.base_url} answered {response.status_code}"
                    )
                else:
                    response.raise_for_status()
                state_bytes = int(response.headers["X-State-Length-Bytes"])
                height = int(response.headers["X-Frame-Height"])
                width = int(response.headers["X-Frame-Width"])
                jpeg_lengths = response.headers.get("X-Frame-Lengths")
                if jpeg_lengths is None:
                    frame_lengths = None
                    frame_bytes = int(response.headers["X-Frame-Length-Bytes"])
                else:
                    # A service that encodes the frames itself sends JPEGs of the asked size, each of its own length.
                    frame_lengths = [int(length) for length in jpeg_lengths.split(",")]
                    frame_bytes = frame_lengths[0]
                body = bytearray()
                next_state: bytes | None = None
                frame_start = state_bytes
                sent = 0
                for data in response.iter_bytes():
                    body += data
                    if next_state is None and len(body) >= state_bytes:
                        next_state = bytes(body[:state_bytes])
                    else:
                        pass
                    while frame_bytes and len(body) - frame_start >= frame_bytes:
                        payload = bytes(body[frame_start : frame_start + frame_bytes])
                        if frame_lengths is None:
                            frame(
                                jpeg_of_i420(
                                    payload,
                                    height,
                                    width,
                                    self.height,
                                    self.width,
                                    self.jpeg_quality,
                                )
                            )
                        else:
                            frame(payload)
                            sent += 1
                            frame_bytes = (
                                frame_lengths[sent] if sent < len(frame_lengths) else 0
                            )
                        frame_start += len(payload)
        except httpx.TransportError as error:
            raise RendererUnavailable(
                f"the AVTR-1 renderer at {self.client.base_url} failed: {error!r}"
            ) from error
        if next_state is None:
            raise RendererUnavailable(
                f"the AVTR-1 renderer at {self.client.base_url} ended a chunk before its state"
            )
        else:
            return next_state

    def is_healthy(self) -> bool:
        try:
            return self.client.get("/health").status_code == 200
        except httpx.TransportError:
            return False


def create_scheduler(
    *,
    renderer_urls: list[str],
    avatar: str,
    background: str,
    speech_sample_rate_hz: int,
    input_sample_rate_hz: int = SESSION_INPUT_SAMPLE_RATE_HZ,
    reply_stream: bool = False,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    sessions_per_renderer: int = DEFAULT_SESSIONS_PER_RENDERER,
    requests_per_renderer: int = DEFAULT_REQUESTS_PER_RENDERER,
    service_jpeg: bool = False,
) -> SessionScheduler:
    renderers = []
    for url in renderer_urls:
        client = httpx.Client(base_url=url, timeout=RENDER_TIMEOUT_S)
        listed = client.get("/avatars")
        listed.raise_for_status()
        loaded = listed.json()
        if avatar not in loaded["avatars"] or background not in loaded["backgrounds"]:
            raise ValueError(
                f"the AVTR-1 renderer at {url} serves the avatars {loaded['avatars']} and the backgrounds {loaded['backgrounds']}, not {avatar!r} on {background!r}"
            )
        else:
            renderers += [
                AVTR1Renderer(
                    client,
                    avatar,
                    background,
                    width=width,
                    height=height,
                    jpeg_quality=jpeg_quality,
                    service_jpeg=service_jpeg,
                )
                for _ in range(requests_per_renderer)
            ]
    return SessionScheduler(
        AvatarHooks(
            renderers,
            speech_sample_rate_hz,
            input_sample_rate_hz=input_sample_rate_hz,
            reply_stream=reply_stream,
        ),
        max_open_sessions=len(renderer_urls) * sessions_per_renderer,
        max_concurrency=DEFAULT_MAX_CONCURRENCY,
    )


__all__ = ["AVTR1Renderer", "create_scheduler", "jpeg_of_i420"]
