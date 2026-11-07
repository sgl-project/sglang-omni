# SPDX-License-Identifier: Apache-2.0
"""Real-service coverage for Whisper realtime input and segment replacement."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest
import requests
from websockets.asyncio.client import ClientConnection, connect

from sglang_omni.serve.transcription_chunking import join_transcript_parts
from sglang_omni.utils import find_available_port
from sglang_omni.utils.audio import load_audio
from tests.utils import (
    disable_proxy,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

pytestmark = [pytest.mark.accelerator, pytest.mark.asyncio]
MODEL_NAME = "openai/whisper-large-v3"
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
EVENT_TIMEOUT_SECONDS = 120
AUDIO_DIRECTORY = Path(__file__).parents[1] / "data"


class TranscriptEvent(TypedDict, total=False):
    type: str
    text: str
    is_final: bool
    segment_id: int
    event_index: int
    audio_start_ms: int
    audio_end_ms: int
    language: str | None


@pytest.fixture(scope="module")
def realtime_server_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    external_port = os.environ.get("WHISPER_REALTIME_PORT")
    if external_port:
        server_url = f"http://127.0.0.1:{int(external_port)}"
        requests.get(f"{server_url}/health", timeout=10).raise_for_status()
        yield server_url
    else:
        port = find_available_port()
        command = [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--model-path",
            os.environ.get("WHISPER_MODEL_PATH", MODEL_NAME),
            "--model-name",
            MODEL_NAME,
            "--enable-realtime",
            "--asr.factory.dtype",
            "float16",
            "--asr.engine.max_running_requests",
            "4",
            "--asr.engine.mem_fraction_static",
            "0.6",
            "--asr.engine.cuda_graph_max_bs_prefill",
            "256",
            "--asr.factory.pre_lm_max_batch_size",
            "4",
            "--asr.factory.pre_lm_cache_max_entries",
            "1",
            "--asr.engine.enable_torch_compile",
            "false",
            "--port",
            str(port),
        ]
        process = start_server_from_cmd(
            command,
            server_log_file(tmp_path_factory, "whisper_realtime"),
            port,
            timeout=600,
        )
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            stop_server(process)


def load_pcm(filename: str) -> bytes:
    samples = load_audio(
        str(AUDIO_DIRECTORY / filename), target_sample_rate=SAMPLE_RATE
    )
    return (
        (np.clip(samples, -1.0, 1.0) * np.iinfo(np.int16).max).astype("<i2").tobytes()
    )


async def receive_until(
    websocket: ClientConnection,
    *,
    partial: bool = False,
    terminal_type: str = "transcription.completed",
) -> list[TranscriptEvent]:
    events: list[TranscriptEvent] = []
    for _ in range(300):
        event: TranscriptEvent = json.loads(
            await asyncio.wait_for(websocket.recv(), timeout=EVENT_TIMEOUT_SECONDS)
        )
        assert event["type"] != "error", event
        events.append(event)
        if partial:
            if event["type"] == "transcription.segment" and not event["is_final"]:
                return events
            else:
                pass
        elif event["type"] == terminal_type:
            return events
        else:
            pass
    raise AssertionError(f"No terminal event in {events}")


async def configure_session(
    websocket: ClientConnection,
    *,
    language: str | None = "en",
    server_vad: bool = False,
) -> None:
    created = json.loads(await websocket.recv())
    assert created["type"] == "session.created", created
    await websocket.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "language": language,
                    "turn_detection": {"type": "server_vad"} if server_vad else None,
                },
            }
        )
    )
    updated = json.loads(await websocket.recv())
    assert updated["type"] == "session.updated", updated


async def send_audio(websocket: ClientConnection, pcm: bytes) -> None:
    packet_bytes = SAMPLE_RATE * BYTES_PER_SAMPLE // 5
    for offset in range(0, len(pcm), packet_bytes):
        await websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(
                        pcm[offset : offset + packet_bytes]
                    ).decode(),
                }
            )
        )


def final_text(events: list[TranscriptEvent]) -> str:
    indexes = [event["event_index"] for event in events]
    assert indexes == sorted(set(indexes)), events
    finals = [
        event
        for event in events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert finals and all(event["text"].strip() for event in finals), events
    completed = events[-1]
    assert completed["type"] == "transcription.completed", completed
    assert completed["text"] == join_transcript_parts(event["text"] for event in finals)
    return completed["text"]


def offline_text(server_url: str, pcm: bytes, language: str = "en") -> str:
    audio_bytes = io.BytesIO()
    with wave.open(audio_bytes, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(BYTES_PER_SAMPLE)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(pcm)
    response = requests.post(
        f"{server_url}/v1/audio/transcriptions",
        data={"model": MODEL_NAME, "language": language, "temperature": 0},
        files={"file": ("reference.wav", audio_bytes.getvalue(), "audio/wav")},
        timeout=EVENT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["text"]


@pytest.mark.parametrize("language", [None, "en"])
async def test_refreshes_replace_text_and_final_matches_offline(
    realtime_server_url: str,
    language: str | None,
) -> None:
    pcm = load_pcm("query_to_draw.wav")
    websocket_url = realtime_server_url.replace("http:", "ws:")
    with disable_proxy():
        expected = await asyncio.to_thread(offline_text, realtime_server_url, pcm)
        async with connect(
            f"{websocket_url}/v1/realtime?intent=transcription"
        ) as socket:
            await configure_session(socket, language=language)
            events: list[TranscriptEvent] = []
            start = 0
            for seconds in (2.1, 4.2):
                end = int(seconds * SAMPLE_RATE) * BYTES_PER_SAMPLE
                await send_audio(socket, pcm[start:end])
                events.extend(await receive_until(socket, partial=True))
                start = end
            await send_audio(socket, pcm[start:])
            await socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await socket.send(json.dumps({"type": "transcription.done"}))
            events.extend(await receive_until(socket))
    assert final_text(events) == expected
    segments = [event for event in events if event["type"] == "transcription.segment"]
    assert len(segments) >= 3, events
    assert {event["segment_id"] for event in segments} == {0}
    assert all(event["language"] in ("en", "english") for event in segments)


async def test_short_input_detects_language_on_done(realtime_server_url: str) -> None:
    pcm = load_pcm("query_to_draw.wav")[: int(1.5 * SAMPLE_RATE) * BYTES_PER_SAMPLE]
    websocket_url = realtime_server_url.replace("http:", "ws:")
    with disable_proxy():
        expected = await asyncio.to_thread(offline_text, realtime_server_url, pcm)
        async with connect(
            f"{websocket_url}/v1/realtime?intent=transcription"
        ) as socket:
            await configure_session(socket, language=None)
            await send_audio(socket, pcm)
            await socket.send(json.dumps({"type": "transcription.done"}))
            events = await receive_until(socket)
    assert final_text(events) == expected
    assert events[-1]["language"] in ("en", "english")


async def test_vad_finalizes_without_manual_commit(realtime_server_url: str) -> None:
    pcm = load_pcm("query_to_cars.wav")
    websocket_url = realtime_server_url.replace("http:", "ws:")
    with disable_proxy():
        async with connect(
            f"{websocket_url}/v1/realtime?intent=transcription"
        ) as socket:
            await configure_session(socket, server_vad=True)
            await send_audio(socket, pcm + bytes(SAMPLE_RATE * BYTES_PER_SAMPLE))
            events = await receive_until(
                socket, terminal_type="input_audio_buffer.speech_stopped"
            )
            while not any(
                event["type"] == "transcription.segment" and event["is_final"]
                for event in events
            ):
                events.extend(
                    await receive_until(socket, terminal_type="transcription.segment")
                )
            await socket.send(json.dumps({"type": "transcription.done"}))
            events.extend(await receive_until(socket))
    assert final_text(events)
    assert any(event["type"] == "input_audio_buffer.speech_started" for event in events)
    assert any(event["type"] == "input_audio_buffer.speech_stopped" for event in events)
    committed_ids = [
        event["segment_id"]
        for event in events
        if event["type"] == "input_audio_buffer.committed"
    ]
    final_ids = [
        event["segment_id"]
        for event in events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert committed_ids == final_ids == sorted(set(final_ids))


async def test_long_input_splits_at_model_window(realtime_server_url: str) -> None:
    pcm = load_pcm("query_to_cars.wav")
    websocket_url = realtime_server_url.replace("http:", "ws:")
    long_pcm = (pcm * 8)[: 31 * SAMPLE_RATE * BYTES_PER_SAMPLE]
    with disable_proxy():
        async with connect(
            f"{websocket_url}/v1/realtime?intent=transcription"
        ) as socket:
            await configure_session(socket)
            await send_audio(socket, long_pcm)
            await socket.send(json.dumps({"type": "transcription.done"}))
            events = await receive_until(socket)
    assert final_text(events)
    finals = [
        event
        for event in events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert [event["segment_id"] for event in finals] == [0, 1], events
    assert [(event["audio_start_ms"], event["audio_end_ms"]) for event in finals] == [
        (0, 30000),
        (30000, 31000),
    ]


async def test_clear_discards_previous_segment(
    realtime_server_url: str,
) -> None:
    discarded = load_pcm("query_to_draw.wav")
    retained = load_pcm("query_to_cars.wav")
    websocket_url = realtime_server_url.replace("http:", "ws:")
    with disable_proxy():
        expected = await asyncio.to_thread(offline_text, realtime_server_url, retained)
        async with connect(
            f"{websocket_url}/v1/realtime?intent=transcription"
        ) as socket:
            await configure_session(socket)
            await send_audio(socket, discarded)
            await receive_until(socket, partial=True)
            await socket.send(json.dumps({"type": "input_audio_buffer.clear"}))
            await receive_until(socket, terminal_type="input_audio_buffer.cleared")
            await send_audio(socket, retained)
            await socket.send(json.dumps({"type": "transcription.done"}))
            assert final_text(await receive_until(socket)) == expected


async def test_disconnect_leaves_new_session_usable(realtime_server_url: str) -> None:
    discarded = load_pcm("query_to_draw.wav")
    retained = load_pcm("query_to_cars.wav")
    websocket_url = realtime_server_url.replace("http:", "ws:")
    with disable_proxy():
        expected = await asyncio.to_thread(offline_text, realtime_server_url, retained)
        async with connect(
            f"{websocket_url}/v1/realtime?intent=transcription"
        ) as socket:
            await configure_session(socket)
            await send_audio(socket, discarded)
        async with connect(
            f"{websocket_url}/v1/realtime?intent=transcription"
        ) as socket:
            await configure_session(socket)
            await send_audio(socket, retained)
            await socket.send(json.dumps({"type": "transcription.done"}))
            assert final_text(await receive_until(socket)) == expected


async def test_concurrent_sessions_keep_audio_isolated(
    realtime_server_url: str,
) -> None:
    recordings = [load_pcm("query_to_draw.wav"), load_pcm("query_to_cars.wav")]
    websocket_url = realtime_server_url.replace("http:", "ws:")

    async def transcribe(pcm: bytes) -> str:
        async with connect(
            f"{websocket_url}/v1/realtime?intent=transcription"
        ) as socket:
            await configure_session(socket)
            await send_audio(socket, pcm)
            await socket.send(json.dumps({"type": "transcription.done"}))
            return final_text(await receive_until(socket))

    with disable_proxy():
        expected = [
            await asyncio.to_thread(offline_text, realtime_server_url, pcm)
            for pcm in recordings
        ]
        transcripts = await asyncio.gather(*(transcribe(pcm) for pcm in recordings))
    assert transcripts == expected
    assert transcripts[0] != transcripts[1]
