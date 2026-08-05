# SPDX-License-Identifier: Apache-2.0
"""Realtime speech-output integration test for the Qwen3-Omni speech pipeline.

Usage:
    pytest tests/test_model/test_qwen3_omni_realtime_audio.py -s -x
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
import websockets

from sglang_omni.utils import find_available_port
from tests.utils import (
    disable_proxy,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_NAME = "qwen3-omni"
STARTUP_TIMEOUT = 600
WS_TIMEOUT = 120
AUDIO_FIXTURE = Path(__file__).parent.parent / "data" / "query_to_draw.wav"


@pytest.fixture(scope="module")
def speech_server(tmp_path_factory: pytest.TempPathFactory):
    port = find_available_port()
    log_file = server_log_file(tmp_path_factory, "realtime_audio_logs")
    cmd = [
        sys.executable,
        "examples/run_qwen3_omni_speech_server.py",
        "--model-path",
        MODEL_PATH,
        "--model-name",
        MODEL_NAME,
        "--gpu-thinker",
        "0",
        "--gpu-talker",
        "1",
        "--gpu-code2wav",
        "1",
        "--enable-realtime",
        "--port",
        str(port),
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    proc.port = port  # type: ignore[attr-defined]
    yield proc
    stop_server(proc)


def _load_pcm16_with_silence(path: Path) -> bytes:
    with wave.open(str(path)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16000
        assert wav.getsampwidth() == 2
        pcm = wav.readframes(wav.getnframes())
    return pcm + b"\x00\x00" * 16000


async def _recv_event(ws) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))


async def _recv_until(ws, terminal_type: str, *, limit: int = 500) -> list[dict]:
    events: list[dict] = []
    for _ in range(limit):
        event = await _recv_event(ws)
        events.append(event)
        if event.get("type") == terminal_type:
            return events
    raise AssertionError(
        f"did not see {terminal_type}; saw {[event.get('type') for event in events]}"
    )


async def _stream_audio(ws, pcm: bytes, *, chunk_ms: int = 200) -> None:
    chunk_bytes = 16000 * chunk_ms // 1000 * 2
    for start in range(0, len(pcm), chunk_bytes):
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(
                        pcm[start : start + chunk_bytes]
                    ).decode(),
                }
            )
        )


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_vad_turn_streams_raw_pcm_audio(
    speech_server: subprocess.Popen,
) -> None:
    """A VAD-committed turn emits non-empty audio and completes."""
    port: int = speech_server.port  # type: ignore[attr-defined]
    with disable_proxy():
        async with websockets.connect(f"ws://localhost:{port}/v1/realtime") as ws:
            assert (await _recv_event(ws))["type"] == "session.created"
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "modalities": ["text", "audio"],
                            "input_audio_format": "pcm16",
                            "output_audio_format": "pcm16",
                        },
                    }
                )
            )
            updated = await _recv_event(ws)
            assert updated["type"] == "session.updated", updated

            await _stream_audio(ws, _load_pcm16_with_silence(AUDIO_FIXTURE))
            events = await _recv_until(
                ws, "conversation.item.input_audio_transcription.completed"
            )

    types = [event["type"] for event in events]
    response_done_index = types.index("response.done")
    assert types.index("response.audio.done") < response_done_index

    encoded_deltas = [
        event["delta"] for event in events if event["type"] == "response.audio.delta"
    ]
    pcm = b"".join(base64.b64decode(delta, validate=True) for delta in encoded_deltas)
    assert pcm

    response = events[response_done_index]["response"]
    assert response["status"] == "completed"
    assert {content["type"] for content in response["output"][0]["content"]} == {
        "text",
        "audio",
    }
