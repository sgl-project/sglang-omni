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
import subprocess
import sys
import wave
from pathlib import Path

import pytest
import requests
import websockets

from sglang_omni.serve.transcription_chunking import join_transcript_parts
from sglang_omni.utils import find_available_port
from tests.utils import (
    disable_proxy,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"
MODEL_NAME = MODEL_PATH
STARTUP_TIMEOUT = 600
WS_TIMEOUT = 120
SAMPLE_RATE = 16000
AUDIO_FIXTURE = Path(__file__).parent.parent / "data" / "query_to_draw.wav"


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    port = find_available_port()
    log_file = server_log_file(tmp_path_factory, "qwen3_asr_realtime_logs")
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        MODEL_PATH,
        "--model-name",
        MODEL_NAME,
        "--enable-realtime",
        "--port",
        str(port),
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    proc.port = port  # type: ignore[attr-defined]
    yield proc
    stop_server(proc)


def _ws_url(port: int) -> str:
    return f"ws://localhost:{port}/v1/realtime?intent=transcription"


def _load_pcm16_16k_mono(path: Path) -> bytes:
    with wave.open(str(path)) as wav_file:
        assert wav_file.getnchannels() == 1, "fixture must be mono"
        assert wav_file.getframerate() == SAMPLE_RATE, "fixture must be 16 kHz"
        assert wav_file.getsampwidth() == 2, "fixture must be PCM16"
        return wav_file.readframes(wav_file.getnframes())


def _seconds_to_bytes(seconds: float) -> int:
    return int(seconds * SAMPLE_RATE) * 2


async def _recv_event(websocket) -> dict:
    return json.loads(await asyncio.wait_for(websocket.recv(), timeout=WS_TIMEOUT))


async def _recv_until(websocket, terminal_type: str, *, limit: int = 300) -> list[dict]:
    events: list[dict] = []
    for _ in range(limit):
        event = await _recv_event(websocket)
        events.append(event)
        if event.get("type") == terminal_type:
            return events
    raise AssertionError(
        f"did not see {terminal_type} after {limit} events; "
        f"saw {[event.get('type') for event in events]}"
    )


async def _recv_partial(websocket) -> list[dict]:
    events: list[dict] = []
    for _ in range(300):
        event = await _recv_event(websocket)
        events.append(event)
        if event.get("type") == "transcription.segment" and not event.get("is_final"):
            return events
    raise AssertionError("did not receive a partial transcription segment")


async def _send_event(websocket, event: dict) -> None:
    await websocket.send(json.dumps(event))


async def _stream_audio(websocket, pcm: bytes, *, chunk_ms: int = 200) -> None:
    chunk_bytes = SAMPLE_RATE * chunk_ms // 1000 * 2
    for offset in range(0, len(pcm), chunk_bytes):
        await _send_event(
            websocket,
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[offset : offset + chunk_bytes]).decode(),
            },
        )


def _assert_no_errors(events: list[dict]) -> None:
    errors = [event for event in events if event.get("type") == "error"]
    assert not errors, errors


def _assert_ordered_event_indexes(events: list[dict]) -> None:
    indexes = [event["event_index"] for event in events]
    assert indexes == sorted(indexes)
    assert len(indexes) == len(set(indexes))


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_manual_commit_exercises_three_refreshes_and_rollback(
    server_process: subprocess.Popen,
) -> None:
    port: int = server_process.port  # type: ignore[attr-defined]
    fixture_pcm = _load_pcm16_16k_mono(AUDIO_FIXTURE)
    pcm = (fixture_pcm * 2)[: _seconds_to_bytes(6.2)]
    boundaries = [_seconds_to_bytes(seconds) for seconds in (2.1, 4.1, 6.1)]

    with disable_proxy():
        async with websockets.connect(_ws_url(port)) as websocket:
            created = await _recv_event(websocket)
            assert created["type"] == "session.created", created
            await _send_event(
                websocket,
                {"type": "session.update", "session": {"turn_detection": None}},
            )
            events = await _recv_until(websocket, "session.updated")

            start = 0
            for boundary in boundaries:
                await _stream_audio(websocket, pcm[start:boundary])
                events.extend(await _recv_partial(websocket))
                start = boundary

            await _send_event(websocket, {"type": "input_audio_buffer.commit"})
            await _send_event(websocket, {"type": "transcription.done"})
            events.extend(await _recv_until(websocket, "transcription.completed"))

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
    _assert_no_errors(events)
    _assert_ordered_event_indexes(events)


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_server_vad_finalizes_without_manual_commit(
    server_process: subprocess.Popen,
) -> None:
    port: int = server_process.port  # type: ignore[attr-defined]
    pcm = _load_pcm16_16k_mono(AUDIO_FIXTURE) + b"\x00\x00" * SAMPLE_RATE

    with disable_proxy():
        async with websockets.connect(_ws_url(port)) as websocket:
            created = await _recv_event(websocket)
            assert created["type"] == "session.created", created
            await _stream_audio(websocket, pcm)
            await _send_event(websocket, {"type": "transcription.done"})
            events = await _recv_until(websocket, "transcription.completed")

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
    _assert_no_errors(events)
    _assert_ordered_event_indexes(events)


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_disconnect_then_new_session_recovers(
    server_process: subprocess.Popen,
) -> None:
    port: int = server_process.port  # type: ignore[attr-defined]
    pcm = _load_pcm16_16k_mono(AUDIO_FIXTURE)

    with disable_proxy():
        async with websockets.connect(_ws_url(port)) as websocket:
            created = await _recv_event(websocket)
            assert created["type"] == "session.created", created
            await _send_event(
                websocket,
                {"type": "session.update", "session": {"turn_detection": None}},
            )
            await _recv_until(websocket, "session.updated")
            await _stream_audio(websocket, pcm[: _seconds_to_bytes(2.1)])

        response = await asyncio.to_thread(
            requests.get, f"http://localhost:{port}/health", timeout=10
        )
        assert response.status_code == 200, response.text

        async with websockets.connect(_ws_url(port)) as websocket:
            created = await _recv_event(websocket)
            assert created["type"] == "session.created", created
            await _send_event(
                websocket,
                {"type": "session.update", "session": {"turn_detection": None}},
            )
            events = await _recv_until(websocket, "session.updated")
            await _stream_audio(websocket, pcm[: _seconds_to_bytes(2.1)])
            await _send_event(websocket, {"type": "input_audio_buffer.commit"})
            await _send_event(websocket, {"type": "transcription.done"})
            events.extend(await _recv_until(websocket, "transcription.completed"))

    finals = [
        event
        for event in events
        if event.get("type") == "transcription.segment" and event["is_final"]
    ]
    completed = next(
        event for event in events if event["type"] == "transcription.completed"
    )
    assert len(finals) == 1, finals
    assert finals[0]["text"].strip()
    assert completed["text"] == finals[0]["text"].strip()
    _assert_no_errors(events)
    _assert_ordered_event_indexes(events)
