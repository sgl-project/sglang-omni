# SPDX-License-Identifier: Apache-2.0
"""The scheduler of the Live Avatar stage over its workers (examples/speech_avatar_workers/liveavatar_worker.py)."""

from __future__ import annotations

from sglang_omni.models.liveavatar.formats import (
    CHUNK_FRAMES,
    FPS,
    HEIGHT,
    PRIME_CHUNKS,
    RENDER_LEAD_MS,
    WIDTH,
)
from sglang_omni.models.liveavatar.session import PipelinedVideoHooks
from sglang_omni.models.speech_avatar.worker import (
    DEFAULT_STARTUP_TIMEOUT_S,
    HttpChunkWorker,
    connect_workers,
)
from sglang_omni.proto.streams import SESSION_INPUT_SAMPLE_RATE_HZ
from sglang_omni.scheduling.session import DEFAULT_MAX_CONCURRENCY, SessionScheduler

# A Live Avatar worker keeps the caches of one Session on its five GPUs.
SESSIONS_PER_WORKER = 1


def check_workers(
    urls: list[str], workers: list[HttpChunkWorker], width: int, height: int
) -> None:
    """The port the stage announced must describe the workers' pictures and blocks."""
    for url, worker in zip(urls, workers):
        info = worker.info
        if (info["width"], info["height"]) != (width, height):
            raise ValueError(
                f"the Live Avatar worker at {url} renders {info['width']}x{info['height']}, not {width}x{height}"
            )
        elif worker.geometry.chunk_frames != CHUNK_FRAMES:
            raise ValueError(
                f"the Live Avatar worker at {url} renders {worker.geometry.chunk_frames}-frame chunks, not {CHUNK_FRAMES}"
            )
        elif "pipeline_depth" not in info:
            raise ValueError(
                f"the worker at {url} is not a pipelined Live Avatar worker: its /info has no pipeline_depth"
            )
        else:
            pass


def create_scheduler(
    *,
    worker_urls: list[str],
    speech_sample_rate_hz: int,
    width: int = WIDTH,
    height: int = HEIGHT,
    render_lead_ms: float = RENDER_LEAD_MS,
    tail_ms: float = 0.0,
    pipeline_depth: int | None = None,
    prime_chunks: int = PRIME_CHUNKS,
    input_sample_rate_hz: int = SESSION_INPUT_SAMPLE_RATE_HZ,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
) -> SessionScheduler:
    workers = connect_workers(worker_urls, "video", FPS, startup_timeout_s)
    check_workers(worker_urls, workers, width, height)
    return SessionScheduler(
        PipelinedVideoHooks(
            workers,
            workers[0].geometry,
            speech_sample_rate_hz,
            idle_picture=workers[0].idle_picture(),
            render_lead_ms=render_lead_ms,
            tail_ms=tail_ms,
            depth=pipeline_depth or workers[0].info["pipeline_depth"],
            prime_chunks=prime_chunks,
            input_sample_rate_hz=input_sample_rate_hz,
        ),
        max_open_sessions=len(workers) * SESSIONS_PER_WORKER,
        max_concurrency=DEFAULT_MAX_CONCURRENCY,
    )


__all__ = ["SESSIONS_PER_WORKER", "check_workers", "create_scheduler"]
