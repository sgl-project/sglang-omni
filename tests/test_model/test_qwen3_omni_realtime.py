# SPDX-License-Identifier: Apache-2.0
"""Realtime WebSocket integration tests for Qwen3-Omni (audio in → response + transcription).

Launches a Qwen3-Omni thinker-only server with ``--enable-realtime`` and drives
/v1/realtime via the ``websockets`` client. VAD is always on (default config);
manual commit / response.create / conversation.item.create are gone. Each VAD
auto-commit triggers two engine passes:

  Pass 1 — response (assistant reply, user sees this first):
    response.created → response.text.delta × N → response.text.done → response.done
  Pass 2 — transcription (user-side, fills conversation history):
    conversation.item.input_audio_transcription.delta × N → .completed

Tests cover:
  - the full VAD → response → transcription sequence on a real wav;
  - clean teardown when the client disconnects mid-flight.

Usage:
    pytest tests/test_model/test_qwen3_omni_realtime.py -s -x

Author:
    Huapeng Zhou https://github.com/PopSoda2002
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
WS_TIMEOUT = 60
AUDIO_FIXTURE = Path(__file__).parent.parent / "data" / "query_to_draw.wav"


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    port = find_available_port()
    log_file = server_log_file(tmp_path_factory, "realtime_logs")
    cmd = [
        sys.executable,
        "examples/run_qwen3_omni_server.py",
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
    return f"ws://localhost:{port}/v1/realtime"


def _load_pcm16_16k_mono(path: Path) -> bytes:
    with wave.open(str(path)) as wf:
        assert wf.getnchannels() == 1, "fixture must be mono"
        assert wf.getframerate() == 16000, "fixture must be 16 kHz"
        assert wf.getsampwidth() == 2, "fixture must be PCM16"
        return wf.readframes(wf.getnframes())


def _wav_with_silence_trailer() -> bytes:
    """Load fixture + 1 s trailing silence so VAD reliably fires speech_stopped."""
    return _load_pcm16_16k_mono(AUDIO_FIXTURE) + b"\x00\x00" * 16000


async def _recv_event(ws) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))


async def _recv_until(ws, terminal_type: str, *, limit: int = 300) -> list[dict]:
    events: list[dict] = []
    for _ in range(limit):
        evt = await _recv_event(ws)
        events.append(evt)
        if evt.get("type") == terminal_type:
            return events
    raise AssertionError(
        f"did not see {terminal_type} after {limit} events; "
        f"saw {[e.get('type') for e in events]}"
    )


async def _stream_audio(ws, pcm: bytes, chunk_ms: int = 200) -> None:
    """Stream PCM16 to the server in fixed-duration chunks."""
    chunk_bytes = 16000 * chunk_ms // 1000 * 2
    for i in range(0, len(pcm), chunk_bytes):
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[i : i + chunk_bytes]).decode(),
                }
            )
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_vad_audio_emits_response_then_transcription(
    server_process: subprocess.Popen,
) -> None:
    """VAD auto-commit drives full lifecycle: VAD → response.* → transcription.*."""
    port: int = server_process.port  # type: ignore[attr-defined]
    pcm = _wav_with_silence_trailer()

    with disable_proxy():
        async with websockets.connect(_ws_url(port)) as ws:
            await _recv_event(ws)  # session.created
            await _stream_audio(ws, pcm)
            # transcription.completed is the last event in the per-turn sequence.
            events = await _recv_until(
                ws, "conversation.item.input_audio_transcription.completed"
            )

    types = [e["type"] for e in events]
    # VAD lifecycle
    assert "input_audio_buffer.speech_started" in types, types
    assert "input_audio_buffer.speech_stopped" in types, types
    assert "input_audio_buffer.committed" in types, types
    # Response pass (assistant reply)
    assert "response.created" in types, types
    assert "response.text.delta" in types, types
    assert "response.text.done" in types, types
    assert "response.done" in types, types
    # Transcription pass (user-side)
    assert "conversation.item.input_audio_transcription.delta" in types, types

    # Response must come before transcription per design (user sees reply first).
    response_done_idx = types.index("response.done")
    transcription_completed_idx = types.index(
        "conversation.item.input_audio_transcription.completed"
    )
    assert (
        response_done_idx < transcription_completed_idx
    ), f"response.done must precede transcription.completed; got {types}"

    # Both passes must produce non-empty text.
    response_done = next(e for e in events if e["type"] == "response.done")
    response_text = response_done["response"]["output"][0]["content"][0]["text"]
    assert (
        response_text.strip()
    ), f"expected non-empty response text; got {response_done!r}"

    completed = next(
        e
        for e in events
        if e["type"] == "conversation.item.input_audio_transcription.completed"
    )
    assert completed.get(
        "transcript", ""
    ).strip(), f"expected non-empty transcript; got {completed!r}"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_disconnect_during_response_keeps_server_healthy(
    server_process: subprocess.Popen,
) -> None:
    """Mid-flight disconnect must not leak tasks — /health stays up and a fresh WS works."""
    port: int = server_process.port  # type: ignore[attr-defined]
    pcm = _wav_with_silence_trailer()

    with disable_proxy():
        async with websockets.connect(_ws_url(port)) as ws:
            await _recv_event(ws)  # session.created
            await _stream_audio(ws, pcm)
            # Take a few events to confirm the response task is alive on the
            # server, then close abruptly without draining the rest.
            for _ in range(5):
                await _recv_event(ws)
        # Context manager exit closes the WebSocket.

        resp = requests.get(f"http://localhost:{port}/health", timeout=10)
        assert resp.status_code == 200, resp.text

        async with websockets.connect(_ws_url(port)) as ws:
            evt = await _recv_event(ws)
            assert evt["type"] == "session.created", evt
