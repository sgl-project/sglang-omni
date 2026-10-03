# SPDX-License-Identifier: Apache-2.0
"""End-to-end coverage for Qwen3-ASR realtime transcription.

Usage:
    CUDA_VISIBLE_DEVICES=0 pytest -s -x \
        tests/test_model/test_qwen3_asr_realtime.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import sys
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import pytest
import requests
import websockets
from pydantic import TypeAdapter
from typing_extensions import NotRequired, TypedDict

from benchmarks.benchmarker.utils import (
    disable_proxy,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)
from sglang_omni.serve.transcription_chunking import join_transcript_parts
from sglang_omni.utils import find_available_port
from tests.utils import ServerHandle


class EventReceiver(Protocol):
    async def recv(self) -> str | bytes: ...


class RealtimeEvent(TypedDict):
    type: str
    event_index: int
    segment_id: NotRequired[int | None]
    text: NotRequired[str]
    is_final: NotRequired[bool]
    error: NotRequired[dict[str, str]]


REALTIME_EVENT = TypeAdapter(RealtimeEvent)
MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"
STARTUP_TIMEOUT = 600
WS_TIMEOUT = 120
SAMPLE_RATE = 16000
AUDIO_FIXTURE = Path(__file__).parent.parent / "data" / "query_to_draw.wav"


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ServerHandle]:
    model_path = os.environ.get("QWEN3_ASR_REALTIME_MODEL_PATH", MODEL_PATH)
    port = find_available_port()
    log_file = server_log_file(tmp_path_factory, "qwen3_asr_realtime_logs")
    command = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        model_path,
        "--model-name",
        model_path,
        "--enable-realtime",
        "--port",
        str(port),
    ]
    command.extend(shlex.split(os.environ.get("QWEN3_ASR_REALTIME_SERVER_ARGS", "")))
    process = start_server_from_cmd(command, log_file, port, timeout=STARTUP_TIMEOUT)
    try:
        yield ServerHandle(proc=process, port=port, log_file=log_file)
    finally:
        stop_server(process)


@contextmanager
def disable_loopback_proxies() -> Iterator[None]:
    with disable_proxy():
        # note (PansaLegrand): Empty proxy variables still expose macOS system proxies.
        os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
        yield


def ws_url(port: int) -> str:
    return f"ws://localhost:{port}/v1/realtime?intent=transcription"


def load_pcm16_16k_mono(path: Path) -> bytes:
    with wave.open(str(path)) as wav_file:
        assert wav_file.getnchannels() == 1, "fixture must be mono"
        assert wav_file.getframerate() == SAMPLE_RATE, "fixture must be 16 kHz"
        assert wav_file.getsampwidth() == 2, "fixture must be PCM16"
        return wav_file.readframes(wav_file.getnframes())


def seconds_to_bytes(seconds: float) -> int:
    return int(seconds * SAMPLE_RATE) * 2


async def recv_event(websocket: EventReceiver) -> RealtimeEvent:
    event = REALTIME_EVENT.validate_json(
        await asyncio.wait_for(websocket.recv(), timeout=WS_TIMEOUT)
    )
    if event["type"] == "error":
        raise AssertionError(f"realtime server error: {event.get('error')}")
    else:
        return event


async def recv_until(
    websocket: EventReceiver,
    terminal_type: str,
    *,
    limit: int = 300,
    final_only: bool = False,
) -> list[RealtimeEvent]:
    events: list[RealtimeEvent] = []
    for _ in range(limit):
        event = await recv_event(websocket)
        events.append(event)
        if event.get("type") == terminal_type and (
            not final_only or event.get("is_final") is True
        ):
            return events
        else:
            pass
    raise AssertionError(
        f"did not see {terminal_type} after {limit} events; "
        f"saw {[event.get('type') for event in events]}"
    )


async def recv_partial(websocket) -> list[dict]:
    events: list[dict] = []
    for _ in range(300):
        event = await recv_event(websocket)
        events.append(event)
        if event.get("type") == "transcription.segment" and not event.get("is_final"):
            return events
    raise AssertionError("did not receive a partial transcription segment")


async def send_event(websocket, event: dict) -> None:
    await websocket.send(json.dumps(event))


async def stream_audio(websocket, pcm: bytes, *, chunk_ms: int = 200) -> None:
    chunk_bytes = SAMPLE_RATE * chunk_ms // 1000 * 2
    for offset in range(0, len(pcm), chunk_bytes):
        await send_event(
            websocket,
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[offset : offset + chunk_bytes]).decode(),
            },
        )


def assert_no_errors(events: list[dict]) -> None:
    errors = [event for event in events if event.get("type") == "error"]
    assert not errors, errors


def assert_ordered_event_indexes(events: list[dict]) -> None:
    indexes = [event["event_index"] for event in events]
    assert indexes == sorted(indexes)
    assert len(indexes) == len(set(indexes))


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_manual_commit_exercises_three_refreshes_and_rollback(
    server_process: ServerHandle,
) -> None:
    port = server_process.port
    fixture_pcm = load_pcm16_16k_mono(AUDIO_FIXTURE)
    pcm = (fixture_pcm * 2)[: seconds_to_bytes(6.2)]
    boundaries = [seconds_to_bytes(seconds) for seconds in (2.1, 4.1, 6.1)]

    with disable_loopback_proxies():
        async with websockets.connect(ws_url(port)) as websocket:
            created = await recv_event(websocket)
            assert created["type"] == "session.created", created
            await send_event(
                websocket,
                {"type": "session.update", "session": {"turn_detection": None}},
            )
            events = await recv_until(websocket, "session.updated")

            start = 0
            for boundary in boundaries:
                await stream_audio(websocket, pcm[start:boundary])
                events.extend(await recv_partial(websocket))
                start = boundary

            await send_event(websocket, {"type": "input_audio_buffer.commit"})
            await send_event(websocket, {"type": "transcription.done"})
            events.extend(await recv_until(websocket, "transcription.completed"))

    partials = [
        event
        for event in events
        if event.get("type") == "transcription.segment" and not event["is_final"]
    ]
    finals = [
        event
        for event in events
        if event.get("type") == "transcription.segment" and event["is_final"]
    ]
    completed = next(
        event for event in events if event["type"] == "transcription.completed"
    )
    assert len(partials) >= 3, partials
    assert {event["segment_id"] for event in partials} == {0}
    assert len(finals) == 1, finals
    assert finals[0]["segment_id"] == 0
    assert finals[0]["text"].strip()
    assert completed["text"] == finals[0]["text"].strip()
    assert_no_errors(events)
    assert_ordered_event_indexes(events)


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_server_vad_finalizes_without_manual_commit(
    server_process: ServerHandle,
) -> None:
    port = server_process.port
    pcm = load_pcm16_16k_mono(AUDIO_FIXTURE) + b"\x00\x00" * SAMPLE_RATE

    with disable_loopback_proxies():
        async with websockets.connect(ws_url(port)) as websocket:
            created = await recv_event(websocket)
            assert created["type"] == "session.created", created
            await stream_audio(websocket, pcm)
            await send_event(websocket, {"type": "transcription.done"})
            events = await recv_until(websocket, "transcription.completed")

    event_types = [event["type"] for event in events]
    finals = [
        event
        for event in events
        if event.get("type") == "transcription.segment" and event["is_final"]
    ]
    committed = [
        event for event in events if event["type"] == "input_audio_buffer.committed"
    ]
    completed = next(
        event for event in events if event["type"] == "transcription.completed"
    )

    assert "input_audio_buffer.speech_started" in event_types, event_types
    assert "input_audio_buffer.speech_stopped" in event_types, event_types
    assert finals, events
    assert len(committed) == len(finals)
    assert [event["segment_id"] for event in committed] == list(range(len(finals)))
    assert [event["segment_id"] for event in finals] == list(range(len(finals)))
    assert all(event["text"].strip() for event in finals)
    assert completed["text"] == join_transcript_parts(event["text"] for event in finals)
    assert_no_errors(events)
    assert_ordered_event_indexes(events)


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_disconnect_then_new_session_recovers(
    server_process: ServerHandle,
) -> None:
    port = server_process.port
    pcm = load_pcm16_16k_mono(AUDIO_FIXTURE)

    with disable_loopback_proxies():
        async with websockets.connect(ws_url(port)) as websocket:
            created = await recv_event(websocket)
            assert created["type"] == "session.created", created
            await send_event(
                websocket,
                {"type": "session.update", "session": {"turn_detection": None}},
            )
            await recv_until(websocket, "session.updated")
            await stream_audio(websocket, pcm[: seconds_to_bytes(2.1)])
            assert_no_errors(await recv_partial(websocket))

        response = await asyncio.to_thread(
            requests.get, f"http://localhost:{port}/health", timeout=10
        )
        assert response.status_code == 200, response.text

        async with websockets.connect(ws_url(port)) as websocket:
            created = await recv_event(websocket)
            assert created["type"] == "session.created", created
            await send_event(
                websocket,
                {"type": "session.update", "session": {"turn_detection": None}},
            )
            events = await recv_until(websocket, "session.updated")
            await stream_audio(websocket, pcm[: seconds_to_bytes(2.1)])
            await send_event(websocket, {"type": "input_audio_buffer.commit"})
            await send_event(websocket, {"type": "transcription.done"})
            events.extend(await recv_until(websocket, "transcription.completed"))

    finals = [
        event
        for event in events
        if event.get("type") == "transcription.segment" and event["is_final"]
    ]
    completed = next(
        event for event in events if event["type"] == "transcription.completed"
    )
    assert len(finals) == 1, finals
    assert finals[0]["segment_id"] == 0
    assert finals[0]["text"].strip()
    assert completed["text"] == finals[0]["text"].strip()
    assert_no_errors(events)
    assert_ordered_event_indexes(events)


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_clear_preserves_committed_text_and_accepts_new_audio(
    server_process: ServerHandle,
) -> None:
    pcm = load_pcm16_16k_mono(AUDIO_FIXTURE)
    replacement_pcm = pcm[: seconds_to_bytes(2.1)]
    with disable_loopback_proxies():
        async with websockets.connect(ws_url(server_process.port)) as websocket:
            created = await recv_event(websocket)
            assert created["type"] == "session.created", created
            await send_event(
                websocket,
                {"type": "session.update", "session": {"turn_detection": None}},
            )
            events = await recv_until(websocket, "session.updated")
            await stream_audio(websocket, pcm)
            await send_event(websocket, {"type": "input_audio_buffer.commit"})
            events.extend(
                await recv_until(websocket, "transcription.segment", final_only=True)
            )
            await stream_audio(websocket, pcm[: seconds_to_bytes(2.1)])
            partial_events = await recv_partial(websocket)
            assert partial_events[-1]["segment_id"] == 1
            events.extend(partial_events)
            await send_event(websocket, {"type": "input_audio_buffer.clear"})
            events.extend(await recv_until(websocket, "input_audio_buffer.cleared"))
            await stream_audio(websocket, replacement_pcm)
            await send_event(websocket, {"type": "input_audio_buffer.commit"})
            await send_event(websocket, {"type": "transcription.done"})
            after_clear = await recv_until(websocket, "transcription.completed")
            events.extend(after_clear)

    assert not any(
        event["type"] == "transcription.segment" and event["segment_id"] == 1
        for event in after_clear
    )
    finals = [
        event
        for event in events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert [event["segment_id"] for event in finals] == [0, 2]
    assert all(event["text"].strip() for event in finals)
    assert events[-1]["text"] == join_transcript_parts(
        event["text"] for event in finals
    )
    assert_no_errors(events)
    assert_ordered_event_indexes(events)


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_repeated_manual_commits_deliver_each_final_once(
    server_process: ServerHandle,
) -> None:
    turn_count = int(os.environ.get("QWEN3_ASR_REALTIME_TURNS", "3"))
    if turn_count <= 0:
        raise ValueError("QWEN3_ASR_REALTIME_TURNS must be positive")
    else:
        pass
    pcm = load_pcm16_16k_mono(AUDIO_FIXTURE)
    with disable_loopback_proxies():
        async with websockets.connect(ws_url(server_process.port)) as websocket:
            created = await recv_event(websocket)
            assert created["type"] == "session.created", created
            await send_event(
                websocket,
                {"type": "session.update", "session": {"turn_detection": None}},
            )
            events = await recv_until(websocket, "session.updated")
            for _ in range(turn_count):
                await stream_audio(websocket, pcm)
                await send_event(websocket, {"type": "input_audio_buffer.commit"})
                events.extend(
                    await recv_until(
                        websocket, "transcription.segment", final_only=True
                    )
                )
            await send_event(websocket, {"type": "transcription.done"})
            events.extend(await recv_until(websocket, "transcription.completed"))

    finals = [
        event
        for event in events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert [event["segment_id"] for event in finals] == list(range(turn_count))
    assert all(event["text"].strip() for event in finals)
    assert events[-1]["text"] == join_transcript_parts(
        event["text"] for event in finals
    )
    assert_no_errors(events)
    assert_ordered_event_indexes(events)
