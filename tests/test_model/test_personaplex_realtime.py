# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-checkpoint session regressions; set PERSONAPLEX_SESSION_CHECKPOINT."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import shlex
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import soundfile
import torch
import websockets
from pydantic import JsonValue
from scipy.signal import resample_poly
from websockets.asyncio.client import ClientConnection

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from sglang_omni.models.personaplex.architecture import SAMPLE_RATE, SAMPLES_PER_FRAME
from sglang_omni.utils.connection import find_available_port

pytestmark = pytest.mark.accelerator
UNIT_BYTES = SAMPLES_PER_FRAME * 2
INPUT_UNITS = 48
BURST_UNITS = 16
JOIN_AFTER_UNITS = 4
EVENT_TIMEOUT_SECONDS = 120
STARTUP_TIMEOUT_SECONDS = 600


@dataclass(frozen=True, kw_only=True)
class SessionCase:
    voice: str
    instructions: str
    seed: int


CASES = (
    SessionCase(voice="NATF2", instructions="You are a helpful assistant.", seed=7),
    SessionCase(voice="NATM1", instructions="You are a friendly car expert.", seed=29),
    SessionCase(voice="VARF0", instructions="You are a patient teacher.", seed=113),
)


@dataclass(frozen=True, kw_only=True)
class SessionReply:
    pcm: bytes
    text: str
    chunk_frames: tuple[int, ...]


@pytest.fixture(scope="module", params=[1, 4])
def session_server(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[int, int]]:
    checkpoint = os.environ.get("PERSONAPLEX_SESSION_CHECKPOINT")
    if checkpoint is None or not torch.cuda.is_available():
        pytest.skip("Set PERSONAPLEX_SESSION_CHECKPOINT and select a CUDA GPU")
    else:
        pass
    coalesced_units = request.param
    port = find_available_port()
    output = tmp_path_factory.mktemp(f"personaplex-session-{coalesced_units}")
    command = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        checkpoint,
        "--variant",
        "session",
        "--enable-realtime",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-sessions",
        str(len(CASES)),
        "--max-coalesced-units",
        str(coalesced_units),
        "--lm.engine.context_length",
        "1024",
        "--lm.engine.enable_deterministic_inference",
        "true",
        "--lm.engine.attention_backend",
        "triton",
        *shlex.split(os.environ.get("PERSONAPLEX_SESSION_STAGE_ARGS", "")),
    ]
    (output / "command.json").write_text(json.dumps(command, indent=2))
    process = start_server_from_cmd(
        command,
        output / "server.log",
        port,
        timeout=int(
            os.environ.get(
                "PERSONAPLEX_SESSION_STARTUP_TIMEOUT", STARTUP_TIMEOUT_SECONDS
            )
        ),
        strip_proxy=True,
        health_path="/v1/realtime/capabilities",
        health_body_contains='"native_full_duplex":true',
    )
    try:
        yield port, coalesced_units
    finally:
        stop_server(process)


@pytest.fixture(scope="module")
def caller_pcm() -> bytes:
    waveform, rate = soundfile.read(
        Path(__file__).parents[1] / "data" / "query_to_cars.wav", dtype="float32"
    )
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
    else:
        pass
    divisor = math.gcd(rate, SAMPLE_RATE)
    waveform = resample_poly(waveform, SAMPLE_RATE // divisor, rate // divisor)
    waveform = np.resize(waveform, INPUT_UNITS * SAMPLES_PER_FRAME)
    return np.round(np.clip(waveform, -1, 1) * 32767).astype("<i2").tobytes()


@pytest.fixture(scope="module")
def reference_replies() -> dict[tuple[float, SessionCase], SessionReply]:
    return {}


async def receive_event(websocket: ClientConnection) -> dict[str, JsonValue]:
    event = json.loads(await asyncio.wait_for(websocket.recv(), EVENT_TIMEOUT_SECONDS))
    assert event["type"] != "error", json.dumps(event)
    return event


async def run_call(
    port: int,
    pcm: bytes,
    case: SessionCase,
    temperature: float,
    mode: Literal["serial", "burst"],
    output: Path,
    *,
    start_after: asyncio.Event | None = None,
    allow_join: asyncio.Event | None = None,
    opened: asyncio.Barrier | None = None,
) -> SessionReply:
    if start_after is not None:
        await asyncio.wait_for(start_after.wait(), EVENT_TIMEOUT_SECONDS)
    else:
        pass
    audio = bytearray()
    text: list[str] = []
    receipts: list[str] = []
    chunk_frames: list[int] = []
    events: list[dict[str, JsonValue]] = []
    async with websockets.connect(
        f"ws://127.0.0.1:{port}/v1/realtime", max_size=8 * 1024 * 1024, proxy=None
    ) as websocket:
        assert (await receive_event(websocket))["type"] == "session.created"
        await websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "event_id": "open",
                    "session": {
                        "output_modalities": ["audio"],
                        "instructions": case.instructions,
                        "audio": {
                            direction: {
                                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE}
                            }
                            for direction in ("input", "output")
                        },
                        "sglang": {
                            "voice": case.voice,
                            "sampling": {
                                "temperature": temperature,
                                "audio_temperature": 0.8 if temperature else 0.0,
                                "seed": case.seed,
                            },
                        },
                    },
                }
            )
        )
        assert (await receive_event(websocket))["type"] == "session.updated"
        if opened is not None:
            await asyncio.wait_for(opened.wait(), EVENT_TIMEOUT_SECONDS)
        else:
            pass

        async def receive_output() -> str:
            event = await receive_event(websocket)
            events.append(event)
            event_type = event["type"]
            if event_type == "response.output_audio.delta":
                chunk = base64.b64decode(event["delta"], validate=True)
                assert len(chunk) % UNIT_BYTES == 0
                audio.extend(chunk)
                chunk_frames.append(len(chunk) // UNIT_BYTES)
            elif event_type == "response.output_audio_transcript.delta":
                text.append(event["delta"])
            elif event_type == "sglang.unit.done":
                receipts.append(event["unit_id"])
                if allow_join is not None and len(receipts) >= JOIN_AFTER_UNITS:
                    allow_join.set()
                else:
                    pass
            else:
                pass
            return event_type

        quantum = 1 if mode == "serial" else BURST_UNITS
        for start in range(0, INPUT_UNITS, quantum):
            end = min(start + quantum, INPUT_UNITS)
            for sequence in range(start, end):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "event_id": f"input-{sequence}",
                            "audio": base64.b64encode(
                                pcm[sequence * UNIT_BYTES : (sequence + 1) * UNIT_BYTES]
                            ).decode("ascii"),
                            "sglang": {"seq": sequence, "t_start_ms": sequence * 80.0},
                        }
                    )
                )
            while len(receipts) < end:
                assert await receive_output() != "sglang.input_audio.drained"
        await websocket.send(
            json.dumps({"type": "sglang.input_audio.end", "event_id": "end"})
        )
        while await receive_output() != "sglang.input_audio.drained":
            pass
        await websocket.send(json.dumps({"type": "session.close", "event_id": "close"}))
        assert await receive_output() == "session.closed"
    assert len(receipts) in (INPUT_UNITS, INPUT_UNITS + 1)
    assert receipts == [f"unit_{index}" for index in range(len(receipts))]
    assert len(audio) == (INPUT_UNITS + 1) * UNIT_BYTES
    assert sum(event["type"] == "response.done" for event in events) == 1
    output.with_suffix(".pcm").write_bytes(audio)
    output.with_suffix(".json").write_text(json.dumps(events))
    return SessionReply(
        pcm=bytes(audio), text="".join(text), chunk_frames=tuple(chunk_frames)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("temperature", [0.0, 0.7])
async def test_session_schedules_preserve_pcm_and_text(
    session_server: tuple[int, int],
    caller_pcm: bytes,
    reference_replies: dict[tuple[float, SessionCase], SessionReply],
    temperature: float,
    tmp_path: Path,
) -> None:
    """Changing batch size, joins or coalescing used to change seeded PCM and text."""
    port, coalesced_units = session_server
    for case in CASES:
        reply = await run_call(
            port,
            caller_pcm,
            case,
            temperature,
            "serial",
            tmp_path / f"solo-{case.voice}",
        )
        reference = reference_replies.setdefault((temperature, case), reply)
        assert (reply.pcm, reply.text) == (reference.pcm, reference.text)
    assert len({reference_replies[temperature, case].pcm for case in CASES}) == len(
        CASES
    )
    assert any(reference_replies[temperature, case].text for case in CASES)
    for schedule in ("concurrent", "joining"):
        opened = asyncio.Barrier(len(CASES)) if schedule == "concurrent" else None
        allow_join = asyncio.Event()
        replies = await asyncio.gather(
            *(
                run_call(
                    port,
                    caller_pcm,
                    case,
                    temperature,
                    "burst" if index == 0 or opened else "serial",
                    tmp_path / f"{schedule}-{case.voice}",
                    opened=opened,
                    start_after=allow_join if index and opened is None else None,
                    allow_join=allow_join if index == 0 else None,
                )
                for index, case in enumerate(CASES)
            )
        )
        for case, reply in zip(CASES, replies, strict=True):
            reference = reference_replies[temperature, case]
            assert (reply.pcm, reply.text) == (reference.pcm, reference.text), (
                schedule,
                case,
            )
        if coalesced_units > 1:
            assert any(frames > 2 for reply in replies for frames in reply.chunk_frames)
        else:
            assert all(
                frames <= 2 for reply in replies for frames in reply.chunk_frames
            )


@pytest.mark.asyncio
async def test_session_close_releases_capacity_for_reopen(
    session_server: tuple[int, int], caller_pcm: bytes, tmp_path: Path
) -> None:
    """Closing with queued audio must release slots and preserve a reopened call."""
    port, _ = session_server
    reference = await run_call(
        port, caller_pcm, CASES[0], 0.0, "serial", tmp_path / "reference"
    )
    async with websockets.connect(
        f"ws://127.0.0.1:{port}/v1/realtime", proxy=None
    ) as websocket:
        assert (await receive_event(websocket))["type"] == "session.created"
        await websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "event_id": "open",
                    "session": {"output_modalities": ["audio"]},
                }
            )
        )
        assert (await receive_event(websocket))["type"] == "session.updated"
        await websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": "backlog",
                    "audio": base64.b64encode(
                        caller_pcm[: BURST_UNITS * UNIT_BYTES]
                    ).decode("ascii"),
                    "sglang": {"seq": 0, "t_start_ms": 0.0},
                }
            )
        )
        while (await receive_event(websocket))["type"] != "sglang.unit.done":
            pass
        await websocket.send(json.dumps({"type": "session.close", "event_id": "close"}))
        while (await receive_event(websocket))["type"] != "session.closed":
            pass
    opened = asyncio.Barrier(len(CASES))
    replies = await asyncio.gather(
        *(
            run_call(
                port,
                caller_pcm,
                CASES[0],
                0.0,
                "burst",
                tmp_path / f"reopen-{index}",
                opened=opened,
            )
            for index in range(len(CASES))
        )
    )
    assert all(
        (reply.pcm, reply.text) == (reference.pcm, reference.text) for reply in replies
    )
