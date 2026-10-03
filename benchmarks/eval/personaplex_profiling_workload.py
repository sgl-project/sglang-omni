# SPDX-License-Identifier: Apache-2.0
"""Collect timed PersonaPlex audio requests and concurrent measurement passes."""

from __future__ import annotations

import asyncio
import hashlib
import time
from contextlib import aclosing
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest

MILLISECONDS_PER_SECOND = 1000.0


@dataclass(kw_only=True)
class RequestMeasurement:
    request_id: str
    latency_milliseconds: float
    time_to_first_audio_milliseconds: float
    audio_duration_seconds: float
    real_time_factor: float
    sample_rate: int
    output_samples: int
    stream_chunk_count: int
    text: str
    audio_sha256: str


@dataclass(kw_only=True)
class PassMeasurement:
    wall_seconds: float
    audio_seconds_per_wall_second: float
    requests: list[RequestMeasurement]


async def collect_request(
    client: Client, request: GenerateRequest, request_id: str
) -> RequestMeasurement:
    started_seconds = time.perf_counter()
    first_audio_seconds: float | None = None
    final_audio: NDArray[np.float32] | None = None
    streamed_audio: list[NDArray[np.float32]] = []
    sample_rate: int | None = None
    final_text: str | None = None
    text_parts: list[str] = []
    async with aclosing(client.generate(request, request_id=request_id)) as chunks:
        async for chunk in chunks:
            if chunk.finish_reason is not None and (
                chunk.audio_data is None or chunk.text
            ):
                final_text = chunk.text
            elif chunk.text:
                text_parts.append(chunk.text)
            else:
                pass
            if chunk.audio_data is None:
                continue
            elif not isinstance(chunk.audio_data, np.ndarray):
                raise TypeError("PersonaPlex audio must be a NumPy waveform")
            elif chunk.sample_rate is None or chunk.sample_rate <= 0:
                raise ValueError("PersonaPlex audio must have a positive sample rate")
            else:
                audio = np.asarray(chunk.audio_data, dtype=np.float32).reshape(-1)
            if sample_rate is not None and sample_rate != chunk.sample_rate:
                raise ValueError("Sample rate changed during a request")
            else:
                sample_rate = chunk.sample_rate
            if audio.size and first_audio_seconds is None:
                first_audio_seconds = time.perf_counter() - started_seconds
            else:
                pass
            if chunk.finish_reason is None:
                streamed_audio.append(audio)
            else:
                final_audio = audio
    latency_seconds = time.perf_counter() - started_seconds
    if final_audio is None:
        raise ValueError("PersonaPlex did not return a terminal waveform")
    elif final_audio.size == 0 or sample_rate is None or first_audio_seconds is None:
        raise ValueError("PersonaPlex returned no audio samples")
    elif streamed_audio and not np.array_equal(
        np.concatenate(streamed_audio), final_audio
    ):
        raise ValueError("Streamed audio differs from the terminal waveform")
    else:
        audio_duration_seconds = final_audio.size / sample_rate
        return RequestMeasurement(
            request_id=request_id,
            latency_milliseconds=latency_seconds * MILLISECONDS_PER_SECOND,
            time_to_first_audio_milliseconds=first_audio_seconds
            * MILLISECONDS_PER_SECOND,
            audio_duration_seconds=audio_duration_seconds,
            real_time_factor=latency_seconds / audio_duration_seconds,
            sample_rate=sample_rate,
            output_samples=int(final_audio.size),
            stream_chunk_count=len(streamed_audio),
            text=final_text if final_text is not None else "".join(text_parts),
            audio_sha256=hashlib.sha256(
                final_audio.astype("<f4", copy=False).tobytes()
            ).hexdigest(),
        )


async def measure_request(
    client: Client, request: GenerateRequest, request_id: str, timeout_seconds: float
) -> RequestMeasurement:
    return await asyncio.wait_for(
        collect_request(client, request, request_id), timeout=timeout_seconds
    )


async def measure_pass(
    client: Client,
    request: GenerateRequest,
    pass_name: str,
    request_count: int,
    concurrency: int,
    timeout_seconds: float,
) -> PassMeasurement:
    semaphore = asyncio.Semaphore(concurrency)

    async def submit_request(request_index: int) -> RequestMeasurement:
        async with semaphore:
            return await measure_request(
                client, request, f"{pass_name}-{request_index}", timeout_seconds
            )

    started_seconds = time.perf_counter()
    request_tasks = [
        asyncio.create_task(submit_request(index)) for index in range(request_count)
    ]
    try:
        measurements = await asyncio.gather(*request_tasks)
    finally:
        for request_task in request_tasks:
            if not request_task.done():
                request_task.cancel()
            else:
                pass
        await asyncio.gather(*request_tasks, return_exceptions=True)
    wall_seconds = time.perf_counter() - started_seconds
    return PassMeasurement(
        wall_seconds=wall_seconds,
        audio_seconds_per_wall_second=sum(
            measurement.audio_duration_seconds for measurement in measurements
        )
        / wall_seconds,
        requests=measurements,
    )
