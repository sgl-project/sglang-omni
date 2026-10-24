# SPDX-License-Identifier: Apache-2.0
"""Realtime end-to-end test for the streaming cascade behind /v1/realtime.

The cascade is served as a turn-based interaction on the native realtime
protocol: server VAD ends each utterance and the reply streams back as text and
audio. Requires the services described in
docs/developer_reference/full_duplex_interactive_model.md, a profile's launch config in
SGLANG_OMNI_CASCADE_CONFIG and a 16 kHz mono PCM16 WAV question in
SGLANG_OMNI_CASCADE_AUDIO. Set SGLANG_OMNI_TEST_REALTIME_PORT to use an already
running server.
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import websockets
from websockets.asyncio.client import ClientConnection

from benchmarks.realtime_turn_latency import (
    MAX_FRAME_BYTES,
    RECV_TIMEOUT_S,
    RealtimeClient,
    read_pcm16,
)
from sglang_omni.utils import find_available_port
from tests.utils import (
    disable_proxy,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

CONFIG = os.environ.get("SGLANG_OMNI_CASCADE_CONFIG")
AUDIO = os.environ.get("SGLANG_OMNI_CASCADE_AUDIO")
STARTUP_TIMEOUT = 300
OUTPUT_SAMPLE_RATE = 24000
SILENCE_DURATION_MS = 800
TRANSCRIPT_DONE = "conversation.item.input_audio_transcription.completed"

pytestmark = pytest.mark.skipif(
    CONFIG is None or AUDIO is None,
    reason="SGLANG_OMNI_CASCADE_CONFIG and SGLANG_OMNI_CASCADE_AUDIO are required",
)


@pytest.fixture(scope="module")
def cascade_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[int]:
    external_port = os.environ.get("SGLANG_OMNI_TEST_REALTIME_PORT")
    if external_port is not None:
        yield int(external_port)
        return
    else:
        pass
    port = find_available_port()
    log_file = server_log_file(tmp_path_factory, "cascade_realtime_logs")
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--config",
        CONFIG,
        "--enable-realtime",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    yield port
    stop_server(proc)


@dataclass
class Turn:
    types: list[str]
    audio: bytes
    transcript: str
    text: str
    reason: str


class CascadeClient(RealtimeClient):
    """The benchmark's realtime client, reading whole turns."""

    async def run_turn(self, pcm: bytes) -> Turn:
        """Stream one utterance and read until the reply and the transcript are done."""
        events: list[dict] = []
        sender = asyncio.create_task(self.stream_audio(pcm))
        try:
            deadline = time.monotonic() + RECV_TIMEOUT_S + len(pcm) / 32000
            while time.monotonic() < deadline:
                event = await self.recv()
                if event["type"] not in (
                    "sglang.unit.done",
                    "sglang.input_audio.accepted",
                ):
                    events.append(event)
                else:
                    pass
                types = {seen["type"] for seen in events}
                if {"response.done", TRANSCRIPT_DONE} <= types:
                    break
                else:
                    pass
            else:
                raise AssertionError(
                    f"turn did not complete; saw {[e['type'] for e in events]}"
                )
            await sender
        finally:
            sender.cancel()
        audio = b"".join(
            base64.b64decode(event["delta"])
            for event in events
            if event["type"] == "response.output_audio.delta"
        )
        transcript = next(
            e["transcript"] for e in events if e["type"] == TRANSCRIPT_DONE
        )
        done = next(e for e in events if e["type"] == "response.done")
        text = next(
            part["text"]
            for part in done["response"]["output"][0]["content"]
            if part["type"] == "output_text"
        )
        return Turn(
            [event["type"] for event in events],
            audio,
            transcript,
            text,
            done["response"]["status_details"]["reason"],
        )

    async def run_barge_in(self, pcm: bytes) -> list[dict]:
        """Speak, wait for the reply's audio, speak again; return the response.done events."""
        done: list[dict] = []
        sender = asyncio.create_task(self.stream_audio(pcm))
        try:
            await self.wait_for("response.output_audio.delta")
            await sender
            sender = asyncio.create_task(self.stream_audio(pcm))
            deadline = time.monotonic() + RECV_TIMEOUT_S + len(pcm) / 16000
            while time.monotonic() < deadline and len(done) < 2:
                event = await self.recv()
                if event["type"] == "response.done":
                    done.append(event)
                else:
                    pass
            await sender
        finally:
            sender.cancel()
        return done

    async def close(self) -> None:
        await self.send("session.close")
        await self.wait_for("session.closed")


def assert_spoken_turn(turn: Turn) -> None:
    assert "input_audio_buffer.speech_started" in turn.types
    assert "input_audio_buffer.speech_stopped" in turn.types
    assert turn.types.index("response.output_audio.delta") < turn.types.index(
        "response.done"
    )
    assert len(turn.audio) > OUTPUT_SAMPLE_RATE * 2, len(turn.audio)
    assert turn.reason == "stop", turn.reason
    assert turn.transcript.strip() and turn.text.strip(), (turn.transcript, turn.text)


async def open_session(websocket: ClientConnection, instructions: str) -> CascadeClient:
    client = CascadeClient(websocket)
    assert (await client.recv())["type"] == "session.created"
    updated = await client.configure(instructions, SILENCE_DURATION_MS)
    granted = updated["session"]["sglang"]["granted"]
    assert granted["interaction"] == "turn_based", granted
    assert {"server_vad", None} <= set(granted["turn_control"]), granted
    return client


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_two_vad_turns_stream_audio_and_transcripts(cascade_server: int) -> None:
    pcm = read_pcm16(AUDIO, silence_s=1.5)
    with disable_proxy():
        async with websockets.connect(
            f"ws://127.0.0.1:{cascade_server}/v1/realtime", max_size=MAX_FRAME_BYTES
        ) as websocket:
            client = await open_session(
                websocket, "Answer in one short English sentence."
            )
            turns = [await client.run_turn(pcm) for _ in range(2)]
            await client.close()
    for turn in turns:
        assert_spoken_turn(turn)


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_instructions_update_applies_to_next_turn(cascade_server: int) -> None:
    pcm = read_pcm16(AUDIO, silence_s=1.5)
    with disable_proxy():
        async with websockets.connect(
            f"ws://127.0.0.1:{cascade_server}/v1/realtime", max_size=MAX_FRAME_BYTES
        ) as websocket:
            client = await open_session(
                websocket, "Answer in one short English sentence."
            )
            first = await client.run_turn(pcm)
            await client.configure(
                "Whatever the user says, reply with exactly the words: banana protocol engaged.",
                SILENCE_DURATION_MS,
            )
            second = await client.run_turn(pcm)
            await client.close()
    assert_spoken_turn(first)
    assert "banana" not in first.text.lower(), first.text
    assert "banana" in second.text.lower(), second.text
    assert len(second.audio) > 0


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_speech_during_the_reply_interrupts_it(cascade_server: int) -> None:
    pcm = read_pcm16(AUDIO, silence_s=1.0)
    with disable_proxy():
        async with websockets.connect(
            f"ws://127.0.0.1:{cascade_server}/v1/realtime", max_size=MAX_FRAME_BYTES
        ) as websocket:
            client = await open_session(
                websocket, "Answer in at least five long English sentences."
            )
            done = await client.run_barge_in(pcm)
            await client.close()
    statuses = [
        (e["response"]["status"], e["response"]["status_details"]["reason"])
        for e in done
    ]
    assert statuses == [("cancelled", "interrupted"), ("completed", "stop")], statuses
