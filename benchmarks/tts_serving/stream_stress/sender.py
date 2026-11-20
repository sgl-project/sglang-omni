# SPDX-License-Identifier: Apache-2.0
"""Dispatch streaming TTS requests on an open-loop QPM schedule.

Requires aiohttp, which the sglang-omni runtime environment provides.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import aiohttp

from benchmarks.tts_serving.stream_stress.config_parser import StressConfig
from benchmarks.tts_serving.stream_stress.responder import Responder, StreamRecorder

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class StopSignal:
    """Stop dispatching new requests on interrupt without losing collected results."""

    stopped: bool = False
    reason: str | None = None

    def request(self, reason: str = "user_interrupt") -> None:
        if not self.stopped:
            self.stopped = True
            self.reason = reason


@dataclass(kw_only=True)
class LoadState:
    """Mutable counters shared by the scheduler and its spawned request tasks."""

    inflight: int = 0
    peak_inflight: int = 0
    peak_pending: int = 0
    max_generator_lag_s: float = 0.0
    load_generator_lagged: bool = False
    scheduled: int = 0
    dispatched: int = 0
    cancelled: bool = False
    last_completion_at: float | None = None


async def run_load(
    config: StressConfig,
    responder: Responder,
    *,
    stop: StopSignal | None = None,
) -> dict[str, Any]:
    """Run the whole load test and return scheduler statistics."""
    stop = stop or StopSignal()
    state = LoadState()
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(
        total=config.timeout_s,
        connect=config.connect_timeout_s,
        sock_read=config.chunk_timeout_s,
    )
    semaphore = (
        asyncio.Semaphore(config.max_concurrency)
        if config.max_concurrency > 0
        else None
    )
    random_generator = random.Random(config.arrival_seed)

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, headers=config.headers
    ) as session:
        if config.warmup_requests:
            await run_warmup(config, responder, session, state)

        load_start = time.perf_counter()
        tasks: set[asyncio.Task[None]] = set()
        try:
            for index, offset in enumerate(config.arrival_offsets):
                if stop.stopped:
                    break
                planned_at = load_start + offset
                delay = planned_at - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                actual = time.perf_counter()
                lag = max(0.0, actual - planned_at)
                state.max_generator_lag_s = max(state.max_generator_lag_s, lag)
                if lag > config.lag_threshold_s:
                    state.load_generator_lagged = True

                task = asyncio.create_task(
                    issue_request(
                        config=config,
                        responder=responder,
                        session=session,
                        semaphore=semaphore,
                        state=state,
                        sequence=index,
                        planned_at_s=offset,
                        load_start=load_start,
                        random_generator=random_generator,
                    )
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                state.scheduled += 1
                state.peak_pending = max(state.peak_pending, len(tasks))
        except asyncio.CancelledError:
            state.cancelled = True
            stop.request("cancelled")
            for task in list(tasks):
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

    wall_end = state.last_completion_at or time.perf_counter()
    return {
        "peak_inflight": state.peak_inflight,
        "peak_pending_tasks": state.peak_pending,
        "max_generator_lag_s": round(state.max_generator_lag_s, 6),
        "load_generator_lagged": state.load_generator_lagged,
        "scheduled": state.scheduled,
        "dispatched": state.dispatched,
        "cancelled": state.cancelled,
        "stop_reason": stop.reason,
        "wall_clock_s": max(0.0, wall_end - load_start),
    }


async def run_warmup(
    config: StressConfig,
    responder: Responder,
    session: aiohttp.ClientSession,
    state: LoadState,
) -> None:
    warmup_start = time.perf_counter()
    random_generator = random.Random(config.arrival_seed + 991)
    for index in range(config.warmup_requests):
        recorder = new_recorder(
            config,
            sequence=-index - 1,
            planned_at_s=0.0,
            warmup=True,
            random_generator=random_generator,
        )
        try:
            await perform_request(config, session, recorder)
        except asyncio.CancelledError:
            recorder.fail("cancelled", "warmup cancelled")
        except Exception as exc:
            recorder.fail("transport_error", f"{type(exc).__name__}: {exc}")
        recorder.mark_completed(time.perf_counter())
        await responder.submit(recorder.finalize(load_start=warmup_start))
    if not config.quiet and config.warmup_requests:
        LOGGER.info(f"warmup completed: {config.warmup_requests} requests")


async def issue_request(
    *,
    config: StressConfig,
    responder: Responder,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore | None,
    state: LoadState,
    sequence: int,
    planned_at_s: float,
    load_start: float,
    random_generator: random.Random,
) -> None:
    recorder = new_recorder(
        config,
        sequence=sequence,
        planned_at_s=planned_at_s,
        warmup=False,
        random_generator=random_generator,
    )
    acquired = False
    entered = False
    try:
        if semaphore is not None:
            await semaphore.acquire()
            acquired = True
        state.inflight += 1
        entered = True
        state.dispatched += 1
        state.peak_inflight = max(state.peak_inflight, state.inflight)
        recorder.mark_sent(time.perf_counter())
        try:
            await perform_request(config, session, recorder)
        except asyncio.TimeoutError as exc:
            recorder.fail(
                "chunk_timeout",
                f"chunk wait timed out: {exc}",
                observed_at=time.perf_counter(),
            )
        except aiohttp.ClientError as exc:
            recorder.fail(
                "transport_error",
                f"{type(exc).__name__}: {exc}",
                observed_at=time.perf_counter(),
            )
        except asyncio.CancelledError:
            recorder.fail(
                "cancelled", "request cancelled", observed_at=time.perf_counter()
            )
        except Exception as exc:
            recorder.fail(
                "transport_error",
                f"{type(exc).__name__}: {exc}",
                observed_at=time.perf_counter(),
            )
        finally:
            recorder.mark_completed(time.perf_counter())
    finally:
        if entered:
            state.inflight -= 1
        if acquired and semaphore is not None:
            semaphore.release()
        completed_at = time.perf_counter()
        state.last_completion_at = max(state.last_completion_at or 0.0, completed_at)

        try:
            if config.collect_outcomes and recorder.headers.get(
                "x-sglang-omni-speech-id"
            ):
                await fetch_outcome(config, session, recorder)
        except asyncio.CancelledError:
            recorder.set_outcome({"error": "outcome lookup cancelled"})
        await responder.submit(recorder.finalize(load_start=load_start))


def new_recorder(
    config: StressConfig,
    *,
    sequence: int,
    planned_at_s: float,
    warmup: bool,
    random_generator: random.Random,
) -> StreamRecorder:
    entry = (
        random_generator.choice(config.corpus)
        if config.text_repeat == "random"
        else config.entry_for(sequence)
    )
    request_id = f"{config.run_id}-{'warmup' if warmup else 'req'}-{abs(sequence):06d}"
    return StreamRecorder(
        config,
        sequence=sequence,
        request_id=request_id,
        sample_id=entry.sample_id,
        text=entry.text,
        warmup=warmup,
        planned_at_s=planned_at_s,
    )


async def perform_request(
    config: StressConfig,
    session: aiohttp.ClientSession,
    recorder: StreamRecorder,
) -> None:
    payload = config.payload_for(recorder.sequence, text=recorder.text)
    async with session.post(config.url, json=payload) as response:
        recorder.start_response(response.status, dict(response.headers.items()))
        if response.status != 200:
            error_body = await response.text()
            recorder.fail("http_error", f"HTTP {response.status}: {error_body[:500]}")
            return
        if config.stream_format == "sse":
            await read_sse_stream(response, recorder)
        else:
            await read_raw_stream(response, recorder)


async def read_raw_stream(
    response: aiohttp.ClientResponse, recorder: StreamRecorder
) -> None:
    """Aggregate HTTP chunk boundaries, timestamped at the first byte."""
    pending_bytes = bytearray()
    pending_start_at: float | None = None
    async for network_bytes, end_of_chunk in response.content.iter_chunks():
        observed_at = time.perf_counter()
        if network_bytes:
            if pending_start_at is None:
                pending_start_at = observed_at
            pending_bytes.extend(network_bytes)
        if end_of_chunk and pending_bytes:
            recorder.add_raw_bytes(
                bytes(pending_bytes), pending_start_at or observed_at
            )
            pending_bytes.clear()
            pending_start_at = None
    if pending_bytes:
        recorder.add_raw_bytes(
            bytes(pending_bytes), pending_start_at or time.perf_counter()
        )


async def read_sse_stream(
    response: aiohttp.ClientResponse, recorder: StreamRecorder
) -> None:
    buffer = bytearray()
    pending_start_at: float | None = None
    async for network_bytes, end_of_chunk in response.content.iter_chunks():
        observed_at = time.perf_counter()
        if network_bytes:
            if pending_start_at is None:
                pending_start_at = observed_at
            buffer.extend(network_bytes)
        if not end_of_chunk:
            continue
        while True:
            frame = pop_sse_frame(buffer)
            if frame is None:
                break
            handle_sse_frame(recorder, frame, pending_start_at or observed_at)
        pending_start_at = None
    while True:
        frame = pop_sse_frame(buffer)
        if frame is None:
            break
        handle_sse_frame(recorder, frame, time.perf_counter())
    if buffer.strip():
        recorder.fail("truncated_sse_frame", f"SSE tail left {len(buffer)} bytes")


def pop_sse_frame(buffer: bytearray) -> bytes | None:
    for separator in (b"\r\n\r\n", b"\n\n"):
        index = buffer.find(separator)
        if index >= 0:
            frame = bytes(buffer[:index])
            buffer[: index + len(separator)] = b""
            return frame
    return None


def handle_sse_frame(
    recorder: StreamRecorder,
    frame: bytes,
    observed_at: float,
) -> None:
    data_lines: list[str] = []
    for raw_line in frame.decode("utf-8", errors="replace").splitlines():
        line = raw_line.rstrip("\r")
        if not line or line.startswith(":"):
            continue
        field_name, _, field_value = line.partition(":")
        if field_name.strip() == "data":
            data_lines.append(field_value.lstrip())
    if not data_lines:
        return
    text = "\n".join(data_lines).strip()
    if text == "[DONE]":
        recorder.add_sse_done(
            {"type": "[DONE]", "finish_reason": "stop"}, len(frame), observed_at
        )
        return
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        recorder.fail("malformed_sse_frame", f"SSE data is not JSON: {text[:200]}")
        return
    if not isinstance(event, dict):
        recorder.fail("malformed_sse_frame", "SSE data is not an object")
        return
    kind = event.get("type")
    if kind == "speech.audio.delta":
        recorder.add_sse_audio(str(event.get("audio", "")), len(frame), observed_at)
    elif kind in ("speech.audio.done", "done"):
        recorder.add_sse_done(event, len(frame), observed_at)
    elif kind in ("speech.audio.error", "error"):
        recorder.add_sse_error(event, len(frame), observed_at)
    else:
        recorder.fail("unknown_sse_event", f"unknown SSE event type: {kind!r}")


async def fetch_outcome(
    config: StressConfig,
    session: aiohttp.ClientSession,
    recorder: StreamRecorder,
) -> None:
    speech_id = recorder.headers.get("x-sglang-omni-speech-id")
    if not speech_id:
        return
    lookup = f"{config.base_url}{config.endpoint.rstrip('/')}/{speech_id}"
    headers: dict[str, str] = {}
    worker = recorder.headers.get("x-sglang-omni-worker")
    if worker:
        headers["X-SGLang-Omni-Route-Worker"] = worker
    try:
        async with session.get(
            lookup,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=config.outcome_timeout_s),
        ) as response:
            if response.status == 200:
                try:
                    recorder.set_outcome(await response.json(content_type=None))
                except (ValueError, aiohttp.ContentTypeError) as exc:
                    recorder.set_outcome({"error": f"invalid outcome JSON: {exc}"})
            else:
                recorder.set_outcome(
                    {
                        "status": response.status,
                        "error": (await response.text())[:200],
                    }
                )
    except Exception as exc:
        recorder.set_outcome({"error": f"{type(exc).__name__}: {exc}"})
