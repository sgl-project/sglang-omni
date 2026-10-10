# SPDX-License-Identifier: Apache-2.0
"""Legacy (turn-based) protocol: client handshake, oracle, reconstruction, export."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import hashlib
import io
import json
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import soundfile
import websockets
from pydantic import JsonValue
from websockets.asyncio.server import ServerConnection

from benchmarks.duplex.client import TurnDetection, run_session
from benchmarks.duplex.oracle import evaluate_trace
from benchmarks.duplex.profiles import PROFILES, DuplexProfile
from benchmarks.duplex.reference_capture import TraceFormat
from benchmarks.duplex.reference_export import export_runs
from benchmarks.duplex.v15_audio import LEGACY_BARGE_IN_CUT, reconstruct_output
from benchmarks.duplex.v15_runner import run_samples
from benchmarks.eval.benchmark_duplex_reference import main as reference_main
from benchmarks.eval.benchmark_duplex_v10 import main as v10_main
from benchmarks.eval.benchmark_duplex_v15 import main as v15_main
from tests.unit_test.benchmarks.test_duplex_client import FIXTURE_PACKETS, FIXTURE_PCM
from tests.unit_test.benchmarks.test_duplex_oracle import find_event, trace_fixture
from tests.unit_test.benchmarks.test_duplex_v10 import write_dataset as write_v10
from tests.unit_test.benchmarks.test_duplex_v15_runner import (
    SERVER,
    SERVER_REVISION,
    write_dataset,
)

LEGACY = "qwen3-omni-half-duplex"
NATIVE = "minicpmo-native-pr2377"
OUTPUT_RATE = 24000
INPUT_RATE = 16000
PACKET_SAMPLES = 1280
SERVER_VAD = TurnDetection(type="server_vad", silence_duration_ms=500)
SEMANTIC_VAD = TurnDetection(type="semantic_vad", eagerness="high")
TAIL_S = 0.1
PeerMode = Literal[
    "healthy", "interrupt", "silent", "degraded", "not_supported", "no_probe_ack"
]


def encode(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode()


def samples(count: int, amplitude: int) -> bytes:
    return np.full(count, amplitude, "<i2").tobytes()


class LegacyPeer:
    """Fake LegacyRealtimeFacade: beta session object, response.audio.delta at 24 kHz."""

    def __init__(self, mode: PeerMode = "healthy") -> None:
        self.mode = mode
        self.received: list[dict[str, JsonValue]] = []
        self.sent: list[dict[str, JsonValue]] = []
        self.pcm = bytearray()
        self.appends = 0
        self.session: dict[str, JsonValue] = {
            "id": "session_legacy",
            "object": "realtime.session",
            "model": "fixture",
            "capabilities": {"turn_detection": ["server_vad", "semantic_vad"]},
            "modalities": ["text"],
            "instructions": "",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "temperature": 0.8,
            "max_response_output_tokens": "inf",
        }

    async def send(self, websocket: ServerConnection, event: dict) -> None:
        event["event_id"] = f"server_{len(self.sent)}"
        self.sent.append(event)
        await websocket.send(json.dumps(event))

    async def response_done(
        self, websocket: ServerConnection, response_id: str, status: str, reason: str
    ) -> None:
        await self.send(
            websocket,
            {
                "type": "response.done",
                "response": {
                    "id": response_id,
                    "object": "realtime.response",
                    "status": status,
                    "status_details": {"reason": reason},
                    "output": [
                        {
                            "id": f"item_{response_id}",
                            "object": "realtime.item",
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "hi"},
                                {"type": "audio", "transcript": "hi"},
                            ],
                        }
                    ],
                    "usage": None,
                },
            },
        )

    async def part(
        self, websocket: ServerConnection, kind: str, response_id: str, **fields
    ) -> None:
        await self.send(
            websocket,
            {
                "type": kind,
                "response_id": response_id,
                "item_id": f"item_{response_id}",
                "output_index": 0,
                "content_index": 1 if ".audio." in kind else 0,
                **fields,
            },
        )

    async def on_append(self, websocket: ServerConnection) -> None:
        if self.appends == 1:
            await self.send(
                websocket,
                {
                    "type": "input_audio_buffer.speech_started",
                    "audio_start_ms": 0,
                    "item_id": "item_user0",
                },
            )
        elif self.appends == 4:
            await self.send(
                websocket,
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "audio_end_ms": 320,
                    "item_id": "item_user0",
                },
            )
            await self.send(
                websocket,
                {"type": "input_audio_buffer.committed", "item_id": "item_user0"},
            )
            await self.send(
                websocket,
                {
                    "type": "response.created",
                    "response": {
                        "id": "r0",
                        "object": "realtime.response",
                        "status": "in_progress",
                        "output": [],
                    },
                },
            )
            for _ in range(2):
                await self.part(
                    websocket,
                    "response.audio.delta",
                    "r0",
                    delta=encode(samples(2400, 1)),
                )
            await self.part(websocket, "response.text.delta", "r0", delta="hi")
            await self.part(websocket, "response.audio.done", "r0")
            await self.part(websocket, "response.text.done", "r0", text="hi")
            await self.response_done(websocket, "r0", "completed", "stop")
        elif self.appends == 6 and self.mode == "interrupt":
            await self.send(
                websocket,
                {
                    "type": "response.created",
                    "response": {
                        "id": "r1",
                        "object": "realtime.response",
                        "status": "in_progress",
                        "output": [],
                    },
                },
            )
            await self.part(
                websocket, "response.audio.delta", "r1", delta=encode(samples(4800, 2))
            )
        elif self.appends == 7 and self.mode == "interrupt":
            await self.send(
                websocket,
                {
                    "type": "input_audio_buffer.speech_started",
                    "audio_start_ms": 480,
                    "item_id": "item_user1",
                },
            )
            await self.send(
                websocket,
                {
                    "type": "output_audio_buffer.cleared",
                    "response_id": "r1",
                    "item_id": "item_r1",
                },
            )
            await self.response_done(websocket, "r1", "cancelled", "turn_detected")
        else:
            pass

    async def handler(self, websocket: ServerConnection) -> None:
        await self.send(
            websocket, {"type": "session.created", "session": dict(self.session)}
        )
        async for raw in websocket:
            event = json.loads(raw)
            self.received.append(event)
            kind = event["type"]
            if kind == "session.update":
                session = event["session"]
                if "turn_detection" in session:
                    requested = session["turn_detection"]
                    self.session["modalities"] = session["modalities"]
                    self.session["turn_detection"] = (
                        {**requested, "type": "server_vad"}
                        if self.mode == "degraded"
                        else requested
                    )
                elif self.mode == "no_probe_ack":
                    await websocket.wait_closed()
                    return
                else:
                    pass
                await self.send(
                    websocket,
                    {"type": "session.updated", "session": dict(self.session)},
                )
            elif kind == "input_audio_buffer.append":
                if self.mode == "not_supported":
                    await self.send(
                        websocket,
                        {
                            "type": "error",
                            "sglang": {"fatal": False},
                            "error": {
                                "type": "invalid_request_error",
                                "code": "not_supported",
                                "message": "unsupported conversation event",
                                "event_id": event["event_id"],
                                "param": None,
                            },
                        },
                    )
                    continue
                else:
                    pass
                self.pcm.extend(base64.b64decode(event["audio"], validate=True))
                self.appends += 1
                if self.mode != "silent":
                    await self.on_append(websocket)
                else:
                    pass
            else:
                raise AssertionError(f"Unexpected client event: {kind}")


async def capture(
    peer: LegacyPeer,
    trace_path: Path,
    *,
    turn_detection: TurnDetection = SERVER_VAD,
    timeout_s: float = 5.0,
) -> list[dict[str, JsonValue]]:
    existing_tasks = asyncio.all_tasks()
    async with websockets.serve(peer.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        await run_session(
            f"ws://127.0.0.1:{port}/v1/realtime",
            FIXTURE_PCM,
            trace_path=trace_path,
            timeout_s=timeout_s,
            profile=LEGACY,
            turn_detection=turn_detection,
            tail_s=TAIL_S,
        )
    assert asyncio.all_tasks() <= existing_tasks
    return [json.loads(line) for line in trace_path.read_text().splitlines()]


def sent_types(records: list[dict[str, JsonValue]]) -> list[str]:
    return [r["event"]["type"] for r in records if r["direction"] == "send"]


def test_profiles_declare_their_protocol() -> None:
    assert PROFILES[LEGACY] == DuplexProfile(
        native_unit_ms=20,
        output_sample_rate=24000,
        stop_requires_eos=False,
        continuous_output=False,
        protocol="legacy",
    )
    assert asdict(PROFILES["nemotron-voicechat-pr2188"]) == {
        "native_unit_ms": 80,
        "output_sample_rate": 22050,
        "stop_requires_eos": True,
        "continuous_output": True,
        "protocol": "native",
    }
    assert asdict(PROFILES[NATIVE]) == {
        "native_unit_ms": 1000,
        "output_sample_rate": 24000,
        "stop_requires_eos": False,
        "continuous_output": False,
        "protocol": "native",
    }


def test_client_records_a_passing_legacy_session(tmp_path: Path) -> None:
    peer = LegacyPeer("interrupt")
    records = asyncio.run(capture(peer, tmp_path / "continuous.jsonl"))
    result = evaluate_trace(records, profile=LEGACY)

    assert result["status"] == "pass", result
    assert bytes(peer.pcm) == FIXTURE_PCM
    assert [r["event"] for r in records if r["direction"] == "send"] == peer.received
    assert [r["event"] for r in records if r["direction"] == "receive"] == peer.sent
    assert not [r for r in records if r["direction"] == "error"]
    assert peer.received[0]["session"] == {
        "modalities": ["text", "audio"],
        "turn_detection": {"type": "server_vad", "silence_duration_ms": 500},
    }
    assert peer.received[-1] == {
        "type": "session.update",
        "event_id": peer.received[-1]["event_id"],
        "session": {},
    }
    assert sent_types(records) == (
        ["session.update"] + ["input_audio_buffer.append"] * FIXTURE_PACKETS
    ) + ["session.update"]
    appends = [r for r in records if r["event"]["type"] == "input_audio_buffer.append"]
    assert [r["event"]["sglang"]["seq"] for r in appends] == list(
        range(FIXTURE_PACKETS)
    )
    probe = next(
        r
        for r in records
        if r["direction"] == "send" and r["event"].get("session") == {}
    )
    assert probe["time_s"] - appends[0]["time_s"] >= len(FIXTURE_PCM) / 32000 + TAIL_S
    assert records[-1]["direction"] == "receive"
    assert records[-1]["event"]["type"] == "session.updated"
    metrics = result["metrics"]
    assert metrics["probe_ack_s"] >= 0
    assert metrics["responses_created"] == 2
    assert metrics["responses_overlapped_by_speech"] == 1
    assert metrics["responses_open_at_close"] == 0
    assert metrics["output_audio_s"] == pytest.approx(0.4)
    assert (tmp_path / "input-send-receipts.json").is_file()

    summary = reconstruct_output(tmp_path, profile=LEGACY)
    assert summary["errors"] == []
    assert summary["sample_rate"] == OUTPUT_RATE
    assert summary["media_samples"] == 9600
    assert summary["barge_in"]["cuts"] == 1
    transcript = json.loads((tmp_path / "transcript.json").read_text())
    assert transcript["text"] == {"response.text.delta": "hi"}


def test_client_fails_the_session_when_the_echo_differs(tmp_path: Path) -> None:
    peer = LegacyPeer("degraded")
    records = asyncio.run(
        capture(peer, tmp_path / "trace.jsonl", turn_detection=SEMANTIC_VAD)
    )
    result = evaluate_trace(records, profile=LEGACY)

    assert result["status"] == "fail"
    assert sent_types(records) == ["session.update"]
    assert records[-1]["direction"] == "error"
    assert records[-1]["event"]["message"] == (
        "RuntimeError: session.updated echoed turn_detection.type 'server_vad', "
        "requested 'semantic_vad'"
    )
    assert any("does not echo" in item for item in result["violations"])


def test_client_treats_a_nonfatal_error_as_fatal(tmp_path: Path) -> None:
    peer = LegacyPeer("not_supported")
    records = asyncio.run(capture(peer, tmp_path / "trace.jsonl"))
    result = evaluate_trace(records, profile=LEGACY)

    assert result["status"] == "fail"
    assert any(
        r["direction"] == "receive"
        and r["event"]["type"] == "error"
        and r["event"]["sglang"]["fatal"] is False
        for r in records
    )
    assert sent_types(records).count("session.update") == 1
    assert sent_types(records).count("input_audio_buffer.append") < FIXTURE_PACKETS
    assert any("code=not_supported" in item for item in result["violations"])


def test_client_times_out_without_the_probe_ack(tmp_path: Path) -> None:
    peer = LegacyPeer("no_probe_ack")
    started_s = time.perf_counter()
    records = asyncio.run(capture(peer, tmp_path / "trace.jsonl", timeout_s=2.0))
    elapsed_s = time.perf_counter() - started_s

    assert 2.0 <= elapsed_s < 5.0, elapsed_s
    assert sent_types(records)[-1] == "session.update"
    assert records[-1]["direction"] == "error"
    assert "Session timeout after 2.0s" in records[-1]["event"]["message"]
    assert evaluate_trace(records, profile=LEGACY)["status"] == "fail"


def test_turn_detection_must_match_the_profile_protocol(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="required by legacy-protocol profiles"):
        asyncio.run(
            run_session(
                "ws://127.0.0.1:1/v1/realtime",
                FIXTURE_PCM,
                trace_path=tmp_path / "legacy.jsonl",
                profile=LEGACY,
            )
        )
    with pytest.raises(ValueError, match="rejected by native ones"):
        asyncio.run(
            run_session(
                "ws://127.0.0.1:1/v1/realtime",
                FIXTURE_PCM,
                trace_path=tmp_path / "native.jsonl",
                profile=NATIVE,
                turn_detection=SERVER_VAD,
            )
        )
    assert not list(tmp_path.iterdir())


def legacy_trace() -> list[dict[str, JsonValue]]:
    pcm = encode(bytes(2 * PACKET_SAMPLES))
    output = encode(samples(2400, 1))
    session = {
        "id": "session_A",
        "modalities": ["text", "audio"],
        "turn_detection": {"type": "server_vad"},
    }
    result: list[dict[str, JsonValue]] = []

    def add(time_s: float, direction: str, kind: str, **event: JsonValue) -> None:
        result.append(
            {
                "direction": direction,
                "time_s": 100 + time_s,
                "event": {"type": kind, "event_id": f"event_{len(result)}", **event},
            }
        )

    def part(time_s: float, kind: str, **fields: JsonValue) -> None:
        add(
            time_s,
            "receive",
            kind,
            response_id="r0",
            item_id="a0",
            output_index=0,
            content_index=1 if ".audio." in kind else 0,
            **fields,
        )

    add(0.00, "receive", "session.created", session={"id": "session_A"})
    add(
        0.01,
        "send",
        "session.update",
        session={
            "modalities": ["text", "audio"],
            "turn_detection": {"type": "server_vad"},
        },
    )
    add(0.02, "receive", "session.updated", session=copy.deepcopy(session))
    add(
        0.10,
        "send",
        "input_audio_buffer.append",
        audio=pcm,
        sglang={"seq": 0, "t_start_ms": 0},
    )
    add(
        0.11,
        "receive",
        "input_audio_buffer.speech_started",
        audio_start_ms=0,
        item_id="item0",
    )
    add(
        0.18,
        "send",
        "input_audio_buffer.append",
        audio=pcm,
        sglang={"seq": 1, "t_start_ms": 80},
    )
    add(
        0.19,
        "receive",
        "input_audio_buffer.speech_stopped",
        audio_end_ms=160,
        item_id="item0",
    )
    add(0.191, "receive", "input_audio_buffer.committed", item_id="item0")
    add(
        0.20,
        "receive",
        "response.created",
        response={"id": "r0", "status": "in_progress"},
    )
    part(0.21, "response.audio.delta", delta=output)
    part(0.22, "response.text.delta", delta="hi")
    part(0.23, "response.audio.done")
    part(0.235, "response.text.done", text="hi")
    add(
        0.24,
        "receive",
        "response.done",
        response={
            "id": "r0",
            "status": "completed",
            "status_details": {"reason": "stop"},
        },
    )
    add(1.26, "send", "session.update", event_id="probe", session={})
    add(1.27, "receive", "session.updated", session=copy.deepcopy(session))
    return result


def test_legacy_trace_and_hand_calculated_metrics() -> None:
    result = evaluate_trace(legacy_trace(), profile=LEGACY)
    assert result["violations"] == []
    assert result["status"] == "pass"
    assert result["coverage"] == {"input_output_overlap": False}
    metrics = result["metrics"]
    assert metrics["input_audio_s"] == pytest.approx(0.16)
    assert metrics["output_audio_s"] == pytest.approx(0.1)
    assert metrics["first_audio_packet_s"] == pytest.approx(0.11)
    assert metrics["audio_packet_gap_max_s"] is None
    assert metrics["probe_ack_s"] == pytest.approx(0.01)
    assert metrics["responses_created"] == 1
    assert metrics["responses_overlapped_by_speech"] == 0
    assert metrics["responses_open_at_close"] == 0


def test_legacy_profiles_and_native_profiles_reject_each_other() -> None:
    assert evaluate_trace(legacy_trace(), profile=NATIVE)["status"] == "fail"
    assert evaluate_trace(trace_fixture(), profile=LEGACY)["status"] == "fail"


@pytest.mark.parametrize(
    ("event_type", "key", "value", "violation"),
    [
        ("response.audio.delta", "content_index", 0, "invalid audio indexes"),
        ("response.audio.delta", "response_id", "foreign", "unknown response"),
        ("response.audio.delta", "delta", "AA==", "whole nonempty samples"),
        ("response.audio.delta", "item_id", None, "audio missing item"),
        (
            "response.done",
            "response",
            {
                "id": "r0",
                "status": "cancelled",
                "status_details": {"reason": "client_cancelled"},
            },
            "unsupported response terminal cancelled/client_cancelled",
        ),
        (
            "response.done",
            "response",
            {"id": "r0", "status": "failed", "status_details": {"reason": "error"}},
            "unsupported response terminal failed/error",
        ),
        (
            "session.updated",
            "session",
            {
                "id": "session_A",
                "modalities": ["text"],
                "turn_detection": {"type": "server_vad"},
            },
            "does not grant audio output",
        ),
        (
            "session.updated",
            "session",
            {
                "id": "session_A",
                "modalities": ["text", "audio"],
                "turn_detection": {"type": "semantic_vad"},
            },
            "'semantic_vad' does not echo the requested 'server_vad'",
        ),
        (
            "session.updated",
            "session",
            {"id": "session_A", "modalities": ["text", "audio"]},
            "None does not echo the requested 'server_vad'",
        ),
        ("session.updated", "session", {"id": "session_B"}, "session ID changed"),
        (
            "input_audio_buffer.append",
            "sglang",
            {"seq": 1},
            "sequence is not contiguous",
        ),
        ("output_audio_buffer.cleared", "response_id", "foreign", "unknown response"),
    ],
)
def test_legacy_wire_mutations_fail(
    event_type: str, key: str, value: JsonValue, violation: str
) -> None:
    trace = legacy_trace()
    if event_type == "output_audio_buffer.cleared":
        trace.insert(
            10,
            {
                "direction": "receive",
                "time_s": 100.215,
                "event": {
                    "type": event_type,
                    "event_id": "cleared",
                    "response_id": "r0",
                    "item_id": "a0",
                },
            },
        )
    else:
        pass
    find_event(trace, event_type)[key] = value
    result = evaluate_trace(trace, profile=LEGACY)
    assert result["status"] == "fail"
    assert any(violation in item for item in result["violations"]), result


@pytest.mark.parametrize(
    ("event_type", "violation"),
    [
        ("session.created", "exactly one receive session.created"),
        ("session.updated", "exactly two receive session.updated"),
        ("input_audio_buffer.append", "missing audio input"),
    ],
)
def test_legacy_missing_steps_fail(event_type: str, violation: str) -> None:
    trace = [r for r in legacy_trace() if r["event"]["type"] != event_type]
    result = evaluate_trace(trace, profile=LEGACY)
    assert result["status"] == "fail"
    assert any(violation in item for item in result["violations"]), result


def test_legacy_probe_must_follow_the_input_and_be_answered() -> None:
    trace = legacy_trace()
    unprobed = [r for r in trace if r["event"]["event_id"] != "probe"][:-1]
    result = evaluate_trace(unprobed, profile=LEGACY)
    assert "expected exactly two send session.update" in result["violations"]
    assert "expected exactly two receive session.updated" in result["violations"]

    late_input = copy.deepcopy(trace)
    append = [
        r for r in late_input if r["event"]["type"] == "input_audio_buffer.append"
    ][-1]
    append["time_s"] = 101.265
    late_input.sort(key=lambda record: record["time_s"])
    result = evaluate_trace(late_input, profile=LEGACY)
    assert "input after the liveness probe" in result["violations"]

    unanswered = copy.deepcopy(trace)
    unanswered.insert(
        3,
        {
            "direction": "receive",
            "time_s": 100.03,
            "event": {
                "type": "session.updated",
                "event_id": "stray",
                "session": {"id": "session_A", "modalities": ["text", "audio"]},
            },
        },
    )
    result = evaluate_trace(unanswered, profile=LEGACY)
    assert "session.updated without a pending session.update" in result["violations"]


@pytest.mark.parametrize(
    ("event_type", "violation"),
    [
        ("sglang.input_audio.accepted", "unsupported server event"),
        ("sglang.unit.done", "unsupported server event"),
        ("sglang.input_audio.drained", "unsupported server event"),
        ("session.closed", "unsupported server event"),
        ("input_audio_buffer.cleared", "server discarded buffered input"),
        ("error", "server error"),
    ],
)
def test_legacy_rejects_native_receipts_closes_and_errors(
    event_type: str, violation: str
) -> None:
    trace = legacy_trace()
    trace.insert(
        4,
        {
            "direction": "receive",
            "time_s": 100.105,
            "event": {
                "type": event_type,
                "event_id": "foreign",
                "sglang": {"fatal": False},
                "error": {"code": "not_supported", "message": "nope"},
            },
        },
    )
    result = evaluate_trace(trace, profile=LEGACY)
    assert result["status"] == "fail"
    assert any(violation in item for item in result["violations"]), result


def test_legacy_turn_detected_cancel_needs_a_preceding_speech_start() -> None:
    trace = legacy_trace()
    find_event(trace, "response.done")["response"] = {
        "id": "r0",
        "status": "cancelled",
        "status_details": {"reason": "turn_detected"},
    }
    uninterrupted = evaluate_trace(trace, profile=LEGACY)
    assert uninterrupted["status"] == "fail"
    assert any(
        "without a preceding speech_started" in item
        for item in uninterrupted["violations"]
    )

    trace.insert(
        10,
        {
            "direction": "receive",
            "time_s": 100.215,
            "event": {
                "type": "input_audio_buffer.speech_started",
                "event_id": "barge",
                "audio_start_ms": 120,
                "item_id": "item1",
            },
        },
    )
    trace.insert(
        11,
        {
            "direction": "receive",
            "time_s": 100.216,
            "event": {
                "type": "output_audio_buffer.cleared",
                "event_id": "cleared",
                "response_id": "r0",
                "item_id": "a0",
            },
        },
    )
    interrupted = evaluate_trace(trace, profile=LEGACY)
    assert interrupted["violations"] == []
    assert interrupted["metrics"]["responses_overlapped_by_speech"] == 1


def test_legacy_accepts_length_finish_and_open_responses_at_close() -> None:
    trace = legacy_trace()
    find_event(trace, "response.done")["response"]["status_details"] = {
        "reason": "length"
    }
    assert evaluate_trace(trace, profile=LEGACY)["status"] == "pass"

    open_at_close = [r for r in legacy_trace() if r["event"]["type"] != "response.done"]
    result = evaluate_trace(open_at_close, profile=LEGACY)
    assert result["violations"] == []
    assert result["metrics"]["responses_open_at_close"] == 1


def write_trace(path: Path, records: list[dict[str, JsonValue]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def reconstruction_records(
    *,
    turn_detection: dict[str, JsonValue],
    interrupt_event: str | None,
    deltas: bool = True,
) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = [
        {
            "direction": "send",
            "time_s": 9.9,
            "event": {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "turn_detection": turn_detection,
                },
            },
        }
    ]
    for index in range(FIXTURE_PACKETS):
        records.append(
            {
                "direction": "send",
                "time_s": 10.0 + index * 0.08,
                "event": {
                    "type": "input_audio_buffer.append",
                    "audio": encode(bytes(2 * PACKET_SAMPLES)),
                    "sglang": {"seq": index, "t_start_ms": index * 80},
                },
            }
        )

    def receive(time_s: float, kind: str, **fields: JsonValue) -> None:
        records.append(
            {
                "direction": "receive",
                "time_s": time_s,
                "event": {"type": kind, **fields},
            }
        )

    if deltas:
        receive(
            10.5,
            "response.audio.delta",
            response_id="r0",
            delta=encode(samples(2400, 1)),
        )
        receive(
            10.55,
            "response.audio.delta",
            response_id="r0",
            delta=encode(samples(2400, 2)),
        )
        receive(10.6, "response.text.delta", response_id="r0", delta="hi")
        receive(
            10.605,
            "response.output_audio.delta",
            response_id="r0",
            delta=encode(samples(2400, 9)),
        )
    else:
        pass
    if interrupt_event is not None:
        receive(
            10.62, interrupt_event, response_id="r0", item_id="a0", audio_start_ms=620
        )
    else:
        pass
    if deltas:
        receive(
            10.7,
            "response.audio.delta",
            response_id="r0",
            delta=encode(samples(1000, 3)),
        )
    else:
        pass
    records.sort(key=lambda record: record["time_s"])
    return records


@pytest.mark.parametrize(
    ("turn_detection", "interrupt_event", "cut"),
    [
        ({"type": "server_vad"}, "output_audio_buffer.cleared", True),
        ({"type": "server_vad"}, "input_audio_buffer.speech_started", True),
        (
            {"type": "server_vad", "interrupt_response": False},
            "input_audio_buffer.speech_started",
            False,
        ),
        (
            {"type": "server_vad", "interrupt_response": False},
            "output_audio_buffer.cleared",
            True,
        ),
    ],
)
def test_reconstruction_cuts_playout_on_server_barge_in(
    tmp_path: Path,
    turn_detection: dict[str, JsonValue],
    interrupt_event: str,
    cut: bool,
) -> None:
    write_trace(
        tmp_path / "continuous.jsonl",
        reconstruction_records(
            turn_detection=turn_detection, interrupt_event=interrupt_event
        ),
    )

    summary = reconstruct_output(tmp_path, profile=LEGACY)

    assert summary["errors"] == []
    assert summary["sample_rate"] == OUTPUT_RATE
    assert summary["media_samples"] == 5800
    assert summary["barge_in"] == {"rule": LEGACY_BARGE_IN_CUT, "cuts": int(cut)}
    playout = json.loads((tmp_path / "playout.json").read_text())
    assert [c["playout_start_sample"] for c in playout["chunks"]] == [
        12000,
        14400,
        16800,
    ]
    played, rate = soundfile.read(str(tmp_path / "output-playout.wav"), dtype="int16")
    assert rate == OUTPUT_RATE and len(played) == 17800
    assert not played[:12000].any()
    assert (played[12000:14400] == 1).all()
    assert (played[14400:14880] == 2).all()
    assert (played[16800:] == 3).all()
    if cut:
        assert playout["cuts"] == [
            {
                "trace_line": 14,
                "type": interrupt_event,
                "receipt_s": pytest.approx(0.62),
                "cut_sample": 14880,
                "dropped_samples": 1920,
            }
        ]
        assert not played[14880:16800].any()
    else:
        assert playout["cuts"] == []
        assert (played[14880:16800] == 2).all()
    transcript = json.loads((tmp_path / "transcript.json").read_text())
    assert transcript["provenance"].startswith("legacy server transcript deltas")
    assert transcript["text"] == {"response.text.delta": "hi"}


def test_reconstruction_pads_a_silent_legacy_session_to_the_input(
    tmp_path: Path,
) -> None:
    write_trace(
        tmp_path / "continuous.jsonl",
        reconstruction_records(
            turn_detection={"type": "server_vad"}, interrupt_event=None, deltas=False
        ),
    )

    summary = reconstruct_output(tmp_path, profile=LEGACY)

    assert summary["errors"] == []
    assert summary["media_samples"] == 0
    assert summary["playout_samples"] == FIXTURE_PACKETS * PACKET_SAMPLES * 3 // 2
    assert soundfile.info(str(tmp_path / "output-media.wav")).frames == 0
    played, rate = soundfile.read(str(tmp_path / "output-playout.wav"), dtype="int16")
    assert rate == OUTPUT_RATE and len(played) == 15360 and not played.any()
    assert summary["barge_in"] == {"rule": LEGACY_BARGE_IN_CUT, "cuts": 0}


def make_legacy_run(
    root: Path,
    input_pcm: bytes,
    *,
    profile: str = LEGACY,
    verdicts: dict[str, str] | None = None,
    cleared_at_s: float | None = None,
) -> Path:
    """A paired capture whose traces speak the legacy protocol at 24 kHz."""
    run = root / "run"
    sample_id = "user_interruption/1"
    input_samples = np.frombuffer(input_pcm, "<i2")
    window_s = len(input_samples) / INPUT_RATE
    rows: list[tuple[float, str, dict[str, JsonValue]]] = [
        (
            999.9,
            "send",
            {
                "type": "session.update",
                "event_id": "configure",
                "session": {
                    "modalities": ["text", "audio"],
                    "turn_detection": {"type": "server_vad"},
                },
            },
        ),
        (
            1000.1,
            "receive",
            {
                "type": "response.audio.delta",
                "response_id": "r0",
                "delta": encode(samples(round(0.3 * OUTPUT_RATE), 1000)),
            },
        ),
        (
            1000 + window_s + 0.1,
            "send",
            {"type": "session.update", "event_id": "probe", "session": {}},
        ),
        (
            1000 + window_s + 0.15,
            "receive",
            {"type": "session.updated", "session": {"id": "s"}},
        ),
    ]
    for index in range(-(-len(input_samples) // PACKET_SAMPLES)):
        chunk = input_samples[index * PACKET_SAMPLES : (index + 1) * PACKET_SAMPLES]
        rows.append(
            (
                1000 + index * 0.08,
                "send",
                {
                    "type": "input_audio_buffer.append",
                    "event_id": f"a{index}",
                    "audio": encode(chunk.tobytes()),
                    "sglang": {"seq": index, "t_start_ms": index * 80},
                },
            )
        )
    if cleared_at_s is not None:
        rows.append(
            (
                1000 + cleared_at_s,
                "receive",
                {"type": "output_audio_buffer.cleared", "response_id": "r0"},
            )
        )
    else:
        pass
    rows.sort(key=lambda row: row[0])
    trace_text = "".join(
        json.dumps({"direction": direction, "time_s": time_s, "event": event}) + "\n"
        for time_s, direction, event in rows
    )
    states = {}
    for variant in ("overlap", "clean"):
        directory = Path("samples") / sample_id / variant
        (run / directory).mkdir(parents=True)
        (run / directory / "input.pcm").write_bytes(input_pcm)
        (run / directory / "continuous.jsonl").write_text(trace_text)
        states[variant] = {
            "directory": str(directory),
            "status": "pass",
            "protocol_verdict": (verdicts or {}).get(variant, "pass"),
            "input": {"sha256": hashlib.sha256(input_pcm).hexdigest()},
            "source": {"file": f"{sample_id}/input.wav", "sha256": None},
        }
    (run / "manifest.json").write_text(json.dumps({"profile": profile}))
    (run / "run.json").write_text(
        json.dumps(
            {"status": "complete", "samples": [{"id": sample_id, "variants": states}]}
        )
    )
    return run


def test_export_takes_the_rate_from_the_profile_and_gates_on_the_verdict(
    tmp_path: Path,
) -> None:
    pcm = samples(8000, 0)
    run = make_legacy_run(tmp_path, pcm, verdicts={"clean": "fail"})

    result = export_runs(
        [run],
        tmp_path / "export",
        "half-duplex",
        trace_format=TraceFormat.LEGACY_PCM16.value,
    )

    assert result["trace_format"] == "realtime-legacy-pcm16-v1"
    assert result["counts"]["eligible_variants"] == 1
    variants = result["samples"][0]["variants"]
    assert variants["overlap"]["eligible"] is True
    assert variants["overlap"]["output"]["native_rate"] == OUTPUT_RATE
    assert variants["overlap"]["output"]["resample"] == {
        "method": "scipy.signal.resample_poly",
        "up": 2,
        "down": 3,
    }
    assert variants["overlap"]["output"]["barge_in_cuts"] == 0
    assert variants["clean"]["eligible"] is False
    assert variants["clean"]["reasons"] == [
        "recorder protocol verdict 'fail' is not pass"
    ]
    audio, rate = soundfile.read(
        tmp_path / "export/user_interruption/1/output.wav", dtype="int16"
    )
    assert rate == INPUT_RATE and len(audio) == 8000
    assert not audio[:1500].any() and audio[1700:6300].min() > 900
    assert not (tmp_path / "export/user_interruption/1/clean_output.wav").exists()


def test_export_cuts_the_window_at_the_cleared_receipt(tmp_path: Path) -> None:
    pcm = samples(8000, 0)
    run = make_legacy_run(tmp_path, pcm, cleared_at_s=0.2)

    result = export_runs(
        [run],
        tmp_path / "export",
        "half-duplex",
        trace_format="realtime-legacy-pcm16-v1",
    )

    overlap = result["samples"][0]["variants"]["overlap"]
    assert overlap["eligible"] is True
    assert overlap["output"]["barge_in_cuts"] == 1
    audio, _ = soundfile.read(
        tmp_path / "export/user_interruption/1/output.wav", dtype="int16"
    )
    assert audio[1700:3100].min() > 900
    assert not audio[3300:].any()


def test_export_refuses_a_trace_format_that_contradicts_the_profile(
    tmp_path: Path,
) -> None:
    pcm = samples(8000, 0)
    legacy_run = make_legacy_run(tmp_path / "legacy", pcm)
    with pytest.raises(ValueError, match="recorded the legacy protocol"):
        export_runs([legacy_run], tmp_path / "native-export", "half-duplex")
    assert not (tmp_path / "native-export").exists()
    for profile in (NATIVE, "sglang"):
        native_run = make_legacy_run(tmp_path / profile, pcm, profile=profile)
        with pytest.raises(ValueError, match="did not record the legacy protocol"):
            export_runs(
                [native_run],
                tmp_path / f"{profile}-export",
                "half-duplex",
                trace_format="realtime-legacy-pcm16-v1",
            )


def test_export_cli_accepts_the_legacy_trace_format(tmp_path: Path) -> None:
    run = make_legacy_run(tmp_path, samples(8000, 0))
    output = tmp_path / "export"
    assert (
        reference_main(
            [
                "export",
                "--engine",
                "half-duplex",
                "--trace-format",
                "realtime-legacy-pcm16-v1",
                "--run",
                str(run),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads((output / "reference-manifest.json").read_text())
    assert manifest["counts"]["eligible_pairs"] == 1


@contextlib.contextmanager
def legacy_peer_server(mode: PeerMode = "interrupt") -> Iterator[str]:
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    state: dict = {}

    async def handler(websocket: ServerConnection) -> None:
        await LegacyPeer(mode).handler(websocket)

    async def serve() -> None:
        state["stop"] = asyncio.Event()
        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            state["port"] = server.sockets[0].getsockname()[1]
            ready.set()
            await state["stop"].wait()

    thread = threading.Thread(target=loop.run_until_complete, args=(serve(),))
    thread.start()
    ready.wait(5)
    try:
        yield f"ws://127.0.0.1:{state['port']}/v1/realtime"
    finally:
        loop.call_soon_threadsafe(state["stop"].set)
        thread.join(5)
        loop.close()


def run_cli(main: Callable[[list[str]], int], argv: list[str]) -> tuple[int, dict]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = main(argv)
    return code, json.loads(stdout.getvalue())


RECORD_ARGS = [
    "--server-revision",
    SERVER_REVISION,
    "--model",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "--dataset-revision",
    "fixture",
    "--timeout",
    "5",
    "--profile",
    LEGACY,
    "--legacy-tail",
    str(TAIL_S),
]


def test_v15_recorder_records_a_legacy_profile(tmp_path: Path) -> None:
    dataset, run = tmp_path / "data", tmp_path / "run"
    write_dataset(dataset)
    with legacy_peer_server() as url:
        code, summary = run_cli(
            v15_main,
            ["record", "--dataset-root", str(dataset), "--url", url]
            + ["--output", str(run), "--sample-id", "user_interruption/1"]
            + RECORD_ARGS
            + [
                "--turn-detection",
                '{"type": "server_vad", "silence_duration_ms": 500}',
            ],
        )
    assert code == 0, summary
    assert summary["variant_status"] == {"pass": 2}
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["profile"] == LEGACY
    assert manifest["config"]["protocol"] == "legacy"
    legacy_config = manifest["config"]["legacy"]
    assert legacy_config["turn_detection"] == {
        "type": "server_vad",
        "silence_duration_ms": 500,
    }
    assert legacy_config["tail_s"] == TAIL_S
    assert legacy_config["barge_in_cut"] == LEGACY_BARGE_IN_CUT
    assert "session_end" in legacy_config
    result = json.loads((run / "run.json").read_text())
    variant = result["samples"][0]["variants"]["overlap"]
    directory = run / variant["directory"]
    variant_manifest = json.loads((directory / "manifest.json").read_text())
    assert variant_manifest["config"]["legacy"] == legacy_config
    assert soundfile.info(directory / "output-media.wav").samplerate == OUTPUT_RATE
    assert variant["output"]["media_duration_s"] == pytest.approx(0.4)
    report = json.loads((directory / "report.json").read_text())
    assert report["cases"][0]["metrics"]["probe_ack_s"] >= 0


def test_v15_recorder_rejects_inconsistent_protocol_arguments(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    write_dataset(dataset)
    base = ["record", "--dataset-root", str(dataset), "--url", "ws://127.0.0.1:1"]
    with pytest.raises(SystemExit) as error:
        v15_main(
            base
            + ["--output", str(tmp_path / "bad-json")]
            + RECORD_ARGS
            + ["--turn-detection", '{"type": "manual"}']
        )
    assert error.value.code == 2
    with pytest.raises(ValueError, match="required by legacy-protocol profiles"):
        v15_main(base + ["--output", str(tmp_path / "missing")] + RECORD_ARGS)
    with pytest.raises(ValueError, match="rejected by native ones"):
        v15_main(
            base
            + ["--output", str(tmp_path / "native")]
            + RECORD_ARGS[:-4]
            + ["--profile", NATIVE, "--turn-detection", '{"type": "server_vad"}']
        )
    assert not list(tmp_path.glob("bad-json")) and not list(tmp_path.glob("missing"))
    assert not list(tmp_path.glob("native"))


def test_run_samples_validates_the_protocol_before_creating_the_run(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data"
    write_dataset(dataset)
    with pytest.raises(ValueError, match="required by legacy-protocol profiles"):
        asyncio.run(
            run_samples(
                dataset,
                url="ws://127.0.0.1:1/v1/realtime",
                output=tmp_path / "never",
                server=SERVER,
                dataset_revision="fixture",
                timeout_s=5.0,
                profile=LEGACY,
            )
        )
    assert not (tmp_path / "never").exists()


def test_v10_recorder_records_a_legacy_profile(tmp_path: Path) -> None:
    dataset, run = tmp_path / "data", tmp_path / "run"
    write_v10(dataset)
    with legacy_peer_server("healthy") as url:
        code, summary = run_cli(
            v10_main,
            ["record", "--dataset-root", str(dataset), "--url", url]
            + ["--output", str(run), "--sample-id", "candor_turn_taking/1"]
            + RECORD_ARGS
            + ["--turn-detection", '{"type": "semantic_vad", "eagerness": "low"}'],
        )
    assert code == 0, summary
    assert summary["variant_status"] == {"pass": 1}
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["kind"] == "full-duplex-bench-v1.0"
    assert manifest["config"]["legacy"]["turn_detection"] == {
        "type": "semantic_vad",
        "eagerness": "low",
    }
