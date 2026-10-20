# SPDX-License-Identifier: Apache-2.0
"""Turn-based session protocol: client handshake, oracle, reconstruction, export."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import hashlib
import io
import json
import threading
from collections.abc import Callable, Iterator
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
from benchmarks.duplex.v15_audio import TURN_BASED_BARGE_IN_CUT, reconstruct_output
from benchmarks.eval.benchmark_duplex_reference import main as reference_main
from benchmarks.eval.benchmark_duplex_v10 import main as v10_main
from benchmarks.eval.benchmark_duplex_v15 import main as v15_main
from tests.unit_test.fixtures.duplex_fakes import (
    FIXTURE_PACKETS,
    FIXTURE_PCM,
    SERVER_REVISION,
    find_event,
    write_dataset,
    write_v10_dataset,
)

TURN_BASED = "turn-based"
LEGACY = "qwen3-omni-half-duplex"
NATIVE = "minicpmo-native-pr2377"
OUTPUT_RATE = 24000
INPUT_RATE = 16000
PACKET_SAMPLES = 1280
UNIT_MS = 200
UNIT_BYTES = INPUT_RATE * UNIT_MS // 1000 * 2
SERVER_VAD = TurnDetection(
    type="server_vad", threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500
)
GRANTED = {
    "interaction": "turn_based",
    "native_full_duplex": False,
    "proactive_output": False,
    "turn_control": ["server_vad"],
    "client_commit": False,
    "input_modalities": ["audio"],
    "output_modalities": ["audio"],
    "input_audio_format": {"type": "audio/pcm", "rate": 16000},
    "output_audio_format": {"type": "audio/pcm", "rate": 24000},
    "native_unit_ms": 200,
    "first_unit_ms": 200,
    "microturn_ms": None,
    "tail_policy": "flush",
    "supports_server_interrupt": False,
    "supports_truncate": True,
    "supports_resume": False,
    "partial_style": "append_only",
    "pressure_policy": "reject",
    "strict_order": True,
    "sampling_parameters": [],
    "supports_reference_audio": False,
    "limits": {
        "max_input_bytes": 1920000,
        "max_output_bytes": 33554432,
        "max_history_chars": 65536,
        "cleanup_timeout_s": 30,
    },
    "rejections": [],
}
PeerMode = Literal["interrupt", "empty_turn", "degraded"]


def encode(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode()


def samples(count: int, amplitude: int) -> bytes:
    return np.full(count, amplitude, "<i2").tobytes()


def configure(turn_detection: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "output_modalities": ["audio"],
        "audio": {"input": {"turn_detection": turn_detection}},
    }


class TurnBasedPeer:
    """Fake turn-based session server: 200 ms units, flush tail, 24 kHz replies.

    The user starts speaking on the first append and stops on the third, which
    the server answers; in interrupt mode the user speaks again on the fifth,
    cancelling that reply, and the server answers the second utterance after
    the input ends.
    """

    def __init__(self, mode: PeerMode = "interrupt") -> None:
        self.mode = mode
        self.received: list[dict[str, JsonValue]] = []
        self.sent: list[dict[str, JsonValue]] = []
        self.pcm = bytearray()
        self.appends = 0
        self.units = 0

    async def send(self, websocket: ServerConnection, event: dict) -> None:
        event["event_id"] = f"server_{len(self.sent)}"
        self.sent.append(event)
        await websocket.send(json.dumps(event))

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
                "content_index": 0,
                **fields,
            },
        )

    async def created(self, websocket: ServerConnection, response_id: str) -> None:
        await self.send(
            websocket,
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "in_progress",
                    "output": [],
                },
            },
        )

    async def done(
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
                    "output": [],
                    "usage": None,
                },
            },
        )

    async def reply(
        self, websocket: ServerConnection, response_id: str, audio_samples: int
    ) -> None:
        await self.created(websocket, response_id)
        await self.part(
            websocket, "response.output_audio_transcript.delta", response_id, delta="hi"
        )
        await self.part(
            websocket,
            "response.output_audio.delta",
            response_id,
            delta=encode(samples(audio_samples, 1)),
        )

    async def finish(
        self, websocket: ServerConnection, response_id: str, status: str, reason: str
    ) -> None:
        await self.part(
            websocket,
            "response.output_audio_transcript.done",
            response_id,
            transcript="hi",
        )
        await self.part(websocket, "response.output_audio.done", response_id)
        await self.done(websocket, response_id, status, reason)

    async def speech(self, websocket: ServerConnection, kind: str, t_ms: float) -> None:
        field = "audio_start_ms" if kind.endswith("started") else "audio_end_ms"
        await self.send(
            websocket,
            {
                "type": f"input_audio_buffer.{kind}",
                "item_id": f"item_user_{self.appends}",
                field: t_ms,
            },
        )

    async def unit_done(self, websocket: ServerConnection, duration_ms: float) -> None:
        await self.send(
            websocket,
            {
                "type": "sglang.unit.done",
                "unit_id": f"unit_{self.units}",
                "sglang": {
                    "unit_id": f"unit_{self.units}",
                    "chunk_seq": 0,
                    "media_time": {
                        "t_start_ms": self.units * UNIT_MS,
                        "duration_ms": duration_ms,
                    },
                },
            },
        )
        self.units += 1

    async def on_append(self, websocket: ServerConnection) -> None:
        if self.appends == 1:
            await self.speech(websocket, "speech_started", 0)
        elif self.appends == 3:
            await self.speech(websocket, "speech_stopped", 240)
            await self.send(
                websocket,
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "item_user_1",
                    "content_index": 0,
                    "transcript": "hello",
                },
            )
            await self.reply(websocket, "resp_0", round(0.5 * OUTPUT_RATE))
        elif self.appends == 5 and self.mode == "interrupt":
            await self.speech(websocket, "speech_started", 400)
            await self.finish(websocket, "resp_0", "cancelled", "interrupted")
        elif self.appends == 5:
            await self.finish(websocket, "resp_0", "completed", "stop")
        else:
            pass

    async def handler(self, websocket: ServerConnection) -> None:
        await self.send(
            websocket,
            {
                "type": "session.created",
                "session": {
                    "id": "sess_fixture",
                    "object": "realtime.session",
                    "type": "realtime",
                    "model": "fixture",
                    "sglang": {"granted": None},
                },
            },
        )
        async for raw in websocket:
            event = json.loads(raw)
            self.received.append(event)
            kind = event["type"]
            if kind == "session.update":
                session = event["session"]
                if set(session) != {"output_modalities", "audio"}:
                    await self.send(
                        websocket,
                        {
                            "type": "error",
                            "sglang": {"fatal": False},
                            "error": {
                                "type": "invalid_request_error",
                                "code": "invalid_request",
                                "message": "invalid session configuration",
                                "event_id": event["event_id"],
                                "param": "session",
                            },
                        },
                    )
                    continue
                else:
                    pass
                audio_input = {
                    "format": {"type": "audio/pcm", "rate": INPUT_RATE},
                    **session["audio"]["input"],
                }
                if self.mode == "degraded":
                    audio_input.pop("turn_detection")
                else:
                    pass
                await self.send(
                    websocket,
                    {
                        "type": "session.updated",
                        "client_event_id": event["event_id"],
                        "session": {
                            "output_modalities": ["audio"],
                            "audio": {
                                "input": audio_input,
                                "output": {
                                    "format": {"type": "audio/pcm", "rate": OUTPUT_RATE}
                                },
                            },
                            "model": "fixture",
                            "type": "realtime",
                            "id": "sess_fixture",
                            "object": "realtime.session",
                            "sglang": {"granted": copy.deepcopy(GRANTED)},
                        },
                    },
                )
            elif kind == "input_audio_buffer.append":
                self.pcm.extend(base64.b64decode(event["audio"], validate=True))
                self.appends += 1
                await self.send(
                    websocket,
                    {
                        "type": "sglang.input_audio.accepted",
                        "seq": event["sglang"]["seq"],
                        "accepted_end_ms": len(self.pcm) / 32,
                        "client_event_id": event["event_id"],
                    },
                )
                await self.on_append(websocket)
                while (self.units + 1) * UNIT_BYTES <= len(self.pcm):
                    await self.unit_done(websocket, UNIT_MS)
            elif kind == "sglang.input_audio.end":
                accepted_ms = len(self.pcm) / 32
                await self.send(
                    websocket,
                    {
                        "type": "sglang.input_audio.ended",
                        "accepted_end_ms": accepted_ms,
                        "tail_policy": "flush",
                        "client_event_id": event["event_id"],
                    },
                )
                if self.mode == "interrupt":
                    await self.reply(websocket, "resp_1", round(0.2 * OUTPUT_RATE))
                    await self.finish(websocket, "resp_1", "completed", "stop")
                else:
                    await self.created(websocket, "resp_1")
                    await self.done(websocket, "resp_1", "completed", "empty_turn")
                await self.unit_done(websocket, accepted_ms - self.units * UNIT_MS)
                await self.send(
                    websocket,
                    {
                        "type": "sglang.input_audio.drained",
                        "accepted_end_ms": accepted_ms,
                        "consumed_ms": accepted_ms,
                        "discarded_ms": 0,
                        "padding_ms": 0,
                        "client_event_id": event["event_id"],
                    },
                )
            elif kind == "session.close":
                await self.send(
                    websocket,
                    {
                        "type": "session.closed",
                        "reason": "client_closed",
                        "client_event_id": event["event_id"],
                    },
                )
                return
            else:
                raise AssertionError(f"Unexpected client event: {kind}")


async def capture(
    peer: TurnBasedPeer, trace_path: Path, timeout_s: float = 5.0
) -> list[dict[str, JsonValue]]:
    existing_tasks = asyncio.all_tasks()
    async with websockets.serve(peer.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        await run_session(
            f"ws://127.0.0.1:{port}/v1/realtime",
            FIXTURE_PCM,
            trace_path=trace_path,
            timeout_s=timeout_s,
            profile=TURN_BASED,
            turn_detection=SERVER_VAD,
        )
    assert asyncio.all_tasks() <= existing_tasks
    return [json.loads(line) for line in trace_path.read_text().splitlines()]


def sent_types(records: list[dict[str, JsonValue]]) -> list[str]:
    return [r["event"]["type"] for r in records if r["direction"] == "send"]


def test_profile_declares_the_turn_based_protocol() -> None:
    assert PROFILES[TURN_BASED] == DuplexProfile(
        native_unit_ms=200,
        output_sample_rate=24000,
        stop_requires_eos=False,
        continuous_output=False,
        protocol="turn_based",
    )


def test_client_records_a_passing_turn_based_session(tmp_path: Path) -> None:
    peer = TurnBasedPeer("interrupt")
    records = asyncio.run(capture(peer, tmp_path / "continuous.jsonl"))
    result = evaluate_trace(records, profile=TURN_BASED)

    assert result["status"] == "pass", result
    assert bytes(peer.pcm) == FIXTURE_PCM
    assert [r["event"] for r in records if r["direction"] == "send"] == peer.received
    assert [r["event"] for r in records if r["direction"] == "receive"] == peer.sent
    assert peer.received[0]["session"] == {
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                }
            }
        },
    }
    assert sent_types(records) == (
        ["session.update"]
        + ["input_audio_buffer.append"] * FIXTURE_PACKETS
        + ["sglang.input_audio.end", "session.close"]
    )
    appends = [r for r in records if r["event"]["type"] == "input_audio_buffer.append"]
    end = next(r for r in records if r["event"]["type"] == "sglang.input_audio.end")
    assert end["time_s"] - appends[0]["time_s"] >= len(FIXTURE_PCM) / 32000
    assert records[-1]["event"]["type"] == "session.closed"
    metrics = result["metrics"]
    assert metrics["responses_created"] == 2
    assert metrics["responses_interrupted"] == 1
    assert metrics["output_audio_s"] == pytest.approx(0.7)
    assert metrics["drain_after_eos_s"] >= 0

    summary = reconstruct_output(tmp_path, profile=TURN_BASED)
    assert summary["errors"] == []
    assert summary["sample_rate"] == OUTPUT_RATE
    assert summary["media_samples"] == round(0.7 * OUTPUT_RATE)
    assert summary["barge_in"] == {"rule": TURN_BASED_BARGE_IN_CUT, "cuts": 1}
    playout = json.loads((tmp_path / "playout.json").read_text())
    (cut,) = playout["cuts"]
    assert cut["type"] == "response.done" and cut["dropped_samples"] > 0
    transcript = json.loads((tmp_path / "transcript.json").read_text())
    assert transcript["provenance"].startswith("turn_based server transcript deltas")
    assert transcript["text"] == {"response.output_audio_transcript.delta": "hihi"}


def test_client_records_an_empty_turn(tmp_path: Path) -> None:
    peer = TurnBasedPeer("empty_turn")
    records = asyncio.run(capture(peer, tmp_path / "continuous.jsonl"))
    result = evaluate_trace(records, profile=TURN_BASED)

    assert result["status"] == "pass", result
    assert result["metrics"]["responses_created"] == 2
    assert result["metrics"]["responses_interrupted"] == 0


def test_client_fails_the_session_when_the_echo_lacks_turn_detection(
    tmp_path: Path,
) -> None:
    peer = TurnBasedPeer("degraded")
    records = asyncio.run(capture(peer, tmp_path / "trace.jsonl"))
    result = evaluate_trace(records, profile=TURN_BASED)

    assert result["status"] == "fail"
    assert sent_types(records) == ["session.update"]
    assert records[-1]["direction"] == "error"
    assert records[-1]["event"]["message"] == (
        "RuntimeError: session.updated echoed turn_detection.type None, "
        "requested 'server_vad'"
    )
    assert any("does not echo" in item for item in result["violations"])


def test_turn_detection_is_required_by_the_turn_based_profile(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="turn-based profiles"):
        asyncio.run(
            run_session(
                "ws://127.0.0.1:1/v1/realtime",
                FIXTURE_PCM,
                trace_path=tmp_path / "trace.jsonl",
                profile=TURN_BASED,
            )
        )
    assert not list(tmp_path.iterdir())


def turn_based_trace() -> list[dict[str, JsonValue]]:
    """240 ms of input in three appends, cut into a 200 ms and a 40 ms unit.

    The user speaks, the server answers, the user barges in and cancels that
    reply, and the server answers the second utterance after the input ends.
    """
    pcm = encode(bytes(2 * PACKET_SAMPLES))
    output = encode(samples(2400, 1))
    turn_detection = {"type": "server_vad", "silence_duration_ms": 500}
    result: list[dict[str, JsonValue]] = []

    def add(time_s: float, direction: str, kind: str, **event: JsonValue) -> None:
        result.append(
            {
                "direction": direction,
                "time_s": 100 + time_s,
                "event": {"type": kind, "event_id": f"event_{len(result)}", **event},
            }
        )

    def append(time_s: float, seq: int) -> None:
        add(
            time_s,
            "send",
            "input_audio_buffer.append",
            event_id=f"input{seq}",
            audio=pcm,
            sglang={"seq": seq, "t_start_ms": seq * 80},
        )
        add(
            time_s + 0.005,
            "receive",
            "sglang.input_audio.accepted",
            seq=seq,
            accepted_end_ms=(seq + 1) * 80,
            client_event_id=f"input{seq}",
        )

    def unit_done(time_s: float, index: int, duration_ms: float) -> None:
        add(
            time_s,
            "receive",
            "sglang.unit.done",
            unit_id=f"unit_{index}",
            sglang={
                "unit_id": f"unit_{index}",
                "chunk_seq": 0,
                "media_time": {"t_start_ms": index * 200, "duration_ms": duration_ms},
            },
        )

    def part(time_s: float, kind: str, response_id: str, **fields: JsonValue) -> None:
        add(
            time_s,
            "receive",
            kind,
            response_id=response_id,
            item_id=f"item_{response_id}",
            output_index=0,
            content_index=0,
            **fields,
        )

    def done(time_s: float, response_id: str, status: str, reason: str) -> None:
        add(
            time_s,
            "receive",
            "response.done",
            response={
                "id": response_id,
                "status": status,
                "status_details": {"reason": reason},
            },
        )

    add(
        0.00,
        "receive",
        "session.created",
        session={"id": "sess_A", "sglang": {"granted": None}},
    )
    add(
        0.01,
        "send",
        "session.update",
        event_id="configure",
        session=configure(turn_detection),
    )
    add(
        0.02,
        "receive",
        "session.updated",
        client_event_id="configure",
        session={
            "id": "sess_A",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": INPUT_RATE},
                    "turn_detection": copy.deepcopy(turn_detection),
                },
                "output": {"format": {"type": "audio/pcm", "rate": OUTPUT_RATE}},
            },
            "sglang": {"granted": copy.deepcopy(GRANTED)},
        },
    )
    append(0.10, 0)
    add(
        0.11,
        "receive",
        "input_audio_buffer.speech_started",
        item_id="item_user_0",
        audio_start_ms=0,
    )
    append(0.18, 1)
    add(
        0.19,
        "receive",
        "input_audio_buffer.speech_stopped",
        item_id="item_user_0",
        audio_end_ms=120,
    )
    add(
        0.195,
        "receive",
        "conversation.item.input_audio_transcription.delta",
        item_id="item_user_0",
        content_index=0,
        delta="hel",
    )
    add(
        0.196,
        "receive",
        "conversation.item.input_audio_transcription.completed",
        item_id="item_user_0",
        content_index=0,
        transcript="hello",
    )
    add(
        0.20,
        "receive",
        "response.created",
        response={"id": "resp_0", "status": "in_progress"},
    )
    part(0.21, "response.output_audio_transcript.delta", "resp_0", delta="hi")
    part(0.215, "response.output_audio.delta", "resp_0", delta=output)
    append(0.26, 2)
    unit_done(0.27, 0, 200)
    add(
        0.28,
        "receive",
        "input_audio_buffer.speech_started",
        item_id="item_user_1",
        audio_start_ms=200,
    )
    part(0.285, "response.output_audio_transcript.done", "resp_0", transcript="hi")
    part(0.286, "response.output_audio.done", "resp_0")
    done(0.29, "resp_0", "cancelled", "interrupted")
    add(0.35, "send", "sglang.input_audio.end", event_id="end")
    add(
        0.36,
        "receive",
        "sglang.input_audio.ended",
        accepted_end_ms=240,
        tail_policy="flush",
        client_event_id="end",
    )
    add(
        0.37,
        "receive",
        "response.created",
        response={"id": "resp_1", "status": "in_progress"},
    )
    part(0.38, "response.output_audio.delta", "resp_1", delta=output)
    part(0.385, "response.output_audio_transcript.delta", "resp_1", delta="ok")
    part(0.39, "response.output_audio_transcript.done", "resp_1", transcript="ok")
    part(0.391, "response.output_audio.done", "resp_1")
    done(0.40, "resp_1", "completed", "stop")
    unit_done(0.41, 1, 40)
    add(
        0.42,
        "receive",
        "sglang.input_audio.drained",
        accepted_end_ms=240,
        consumed_ms=240,
        discarded_ms=0,
        padding_ms=0,
        client_event_id="end",
    )
    add(0.43, "send", "session.close", event_id="close")
    add(
        0.44,
        "receive",
        "session.closed",
        client_event_id="close",
        reason="client_closed",
    )
    return result


def receive_events(
    trace: list[dict[str, JsonValue]], event_type: str
) -> list[dict[str, JsonValue]]:
    return [
        r["event"]
        for r in trace
        if r["direction"] == "receive" and r["event"]["type"] == event_type
    ]


def test_turn_based_trace_and_hand_calculated_metrics() -> None:
    result = evaluate_trace(turn_based_trace(), profile=TURN_BASED)
    assert result["violations"] == []
    assert result["status"] == "pass"
    metrics = result["metrics"]
    assert metrics["input_audio_s"] == pytest.approx(0.24)
    assert metrics["output_audio_s"] == pytest.approx(0.2)
    assert metrics["first_audio_packet_s"] == pytest.approx(0.115)
    assert metrics["drain_after_eos_s"] == pytest.approx(0.07)
    assert metrics["close_ack_s"] == pytest.approx(0.01)
    assert metrics["responses_created"] == 2
    assert metrics["responses_interrupted"] == 1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("supports_truncate", True),
        ("supports_truncate", False),
        ("client_commit", True),
        ("client_commit", False),
        ("turn_control", ["server_vad"]),
        ("turn_control", ["server_vad", None]),
    ],
)
def test_deployment_grants_beyond_server_vad_turns_pass(
    name: str, value: JsonValue
) -> None:
    trace = turn_based_trace()
    session = find_event(trace, "session.updated")["session"]
    session["sglang"]["granted"][name] = value
    assert evaluate_trace(trace, profile=TURN_BASED)["violations"] == []


def with_progress(response_id: str) -> list[dict[str, JsonValue]]:
    """The turn-based trace with an alignment progress report after the first
    response.created."""
    trace = turn_based_trace()
    created = next(
        index
        for index, record in enumerate(trace)
        if record["direction"] == "receive"
        and record["event"]["type"] == "response.created"
    )
    progress = {
        "type": "sglang.response.progress",
        "event_id": "evt_progress",
        "response_id": response_id,
        "item_id": "item_0",
        "output_index": 0,
        "content_index": 0,
        "text_end": 5,
        "audio_ms": 200.0,
    }
    record = {**trace[created], "event": progress}
    return [*trace[: created + 1], record, *trace[created + 1 :]]


def test_progress_reports_must_name_a_created_response() -> None:
    first = receive_events(turn_based_trace(), "response.created")[0]
    known = with_progress(first["response"]["id"])
    assert evaluate_trace(known, profile=TURN_BASED)["violations"] == []
    unknown = evaluate_trace(with_progress("resp_missing"), profile=TURN_BASED)
    assert unknown["violations"] == ["sglang.response.progress: unknown response"]


def granted_with(name: str, value: JsonValue) -> Callable[[list], None]:
    def mutate(trace: list[dict[str, JsonValue]]) -> None:
        session = find_event(trace, "session.updated")["session"]
        session["sglang"]["granted"][name] = value

    return mutate


def set_field(event_type: str, key: str, value: JsonValue) -> Callable[[list], None]:
    def mutate(trace: list[dict[str, JsonValue]]) -> None:
        find_event(trace, event_type)[key] = value

    return mutate


def finish_last_response(status: str, reason: str) -> Callable[[list], None]:
    def mutate(trace: list[dict[str, JsonValue]]) -> None:
        receive_events(trace, "response.done")[-1]["response"]["status"] = status
        receive_events(trace, "response.done")[-1]["response"]["status_details"] = {
            "reason": reason
        }

    return mutate


def echo_type(echoed: str) -> Callable[[list], None]:
    def mutate(trace: list[dict[str, JsonValue]]) -> None:
        session = find_event(trace, "session.updated")["session"]
        session["audio"]["input"]["turn_detection"]["type"] = echoed

    return mutate


@pytest.mark.parametrize(
    ("mutate", "violation"),
    [
        (granted_with("interaction", "native"), "unsupported capability: interaction"),
        (
            granted_with("native_full_duplex", True),
            "unsupported capability: native_full_duplex",
        ),
        (granted_with("tail_policy", "pad"), "unsupported capability: tail_policy"),
        (granted_with("turn_control", [None]), "unsupported capability: turn_control"),
        (
            granted_with("client_commit", "yes"),
            "unsupported capability: client_commit",
        ),
        (granted_with("native_unit_ms", 80), "unsupported capability: native_unit_ms"),
        (
            granted_with("output_audio_format", {"type": "audio/pcm", "rate": 22050}),
            "unsupported capability: output_audio_format",
        ),
        (
            set_field("sglang.input_audio.ended", "tail_policy", "pad"),
            "ended tail policy mismatch",
        ),
        (
            set_field("sglang.input_audio.drained", "padding_ms", 160),
            "drained padding mismatch",
        ),
        (
            echo_type("semantic_vad"),
            "'semantic_vad' does not echo the requested 'server_vad'",
        ),
        (
            set_field(
                "session.update",
                "session",
                {
                    "modalities": ["text", "audio"],
                    "turn_detection": {"type": "server_vad"},
                },
            ),
            "configure update lacks audio.input.turn_detection.type",
        ),
        (
            finish_last_response("cancelled", "client_closed"),
            "unsupported response terminal cancelled/client_closed",
        ),
        (
            finish_last_response("failed", "error"),
            "unsupported response terminal failed/error",
        ),
        (
            set_field("response.output_audio.delta", "content_index", 1),
            "invalid audio indexes",
        ),
    ],
)
def test_turn_based_wire_mutations_fail(
    mutate: Callable[[list], None], violation: str
) -> None:
    trace = turn_based_trace()
    mutate(trace)
    result = evaluate_trace(trace, profile=TURN_BASED)
    assert result["status"] == "fail"
    assert any(violation in item for item in result["violations"]), result


def test_interrupted_cancel_needs_its_own_preceding_speech_start() -> None:
    trace = turn_based_trace()
    assert evaluate_trace(trace, profile=TURN_BASED)["violations"] == []

    unannounced = [
        r for r in trace if r["event"]["type"] != "input_audio_buffer.speech_started"
    ]
    result = evaluate_trace(unannounced, profile=TURN_BASED)
    assert result["status"] == "fail"
    assert (
        "response resp_0 cancelled by an interruption without a preceding "
        "speech_started"
    ) in result["violations"]

    twice = [
        r
        for r in trace
        if not (
            r["event"]["type"] == "input_audio_buffer.speech_started"
            and r["event"]["item_id"] == "item_user_0"
        )
    ]
    finish_last_response("cancelled", "interrupted")(twice)
    result = evaluate_trace(twice, profile=TURN_BASED)
    assert result["violations"] == [
        "response resp_1 cancelled by an interruption without a preceding "
        "speech_started"
    ]


def test_a_reply_created_after_its_barge_in_may_be_cancelled() -> None:
    trace = turn_based_trace()
    barge_in = next(
        r
        for r in trace
        if r["event"]["type"] == "input_audio_buffer.speech_started"
        and r["event"]["item_id"] == "item_user_1"
    )
    trace.remove(barge_in)
    barge_in["time_s"] = 100.30
    trace.insert(
        trace.index(
            next(r for r in trace if r["event"]["type"] == "sglang.input_audio.end")
        ),
        barge_in,
    )
    finish_last_response("cancelled", "interrupted")(trace)
    receive_events(trace, "response.done")[0]["response"]["status"] = "completed"
    receive_events(trace, "response.done")[0]["response"]["status_details"] = {
        "reason": "stop"
    }

    result = evaluate_trace(trace, profile=TURN_BASED)
    assert result["violations"] == []
    assert result["metrics"]["responses_interrupted"] == 1


def test_empty_turn_terminal_is_accepted() -> None:
    trace = [r for r in turn_based_trace() if r["event"].get("response_id") != "resp_1"]
    finish_last_response("completed", "empty_turn")(trace)
    result = evaluate_trace(trace, profile=TURN_BASED)
    assert result["violations"] == []
    assert result["metrics"]["responses_created"] == 2


def test_responses_must_end_before_the_session_closes() -> None:
    trace = [
        r
        for r in turn_based_trace()
        if not (
            r["event"]["type"] == "response.done"
            and r["event"]["response"]["id"] == "resp_1"
        )
    ]
    result = evaluate_trace(trace, profile=TURN_BASED)
    assert result["status"] == "fail"
    assert "response resp_1 has no response.done" in result["violations"]


@pytest.mark.parametrize(
    ("event_type", "violation"),
    [
        ("output_audio_buffer.cleared", "unsupported server event"),
        ("input_audio_buffer.committed", "unsupported server event"),
        ("response.audio.delta", "unsupported server event"),
    ],
)
def test_turn_based_rejects_events_outside_the_wire(
    event_type: str, violation: str
) -> None:
    trace = turn_based_trace()
    first_audio = next(
        r for r in trace if r["event"]["type"] == "response.output_audio.delta"
    )
    trace.insert(
        trace.index(first_audio) + 1,
        {
            "direction": "receive",
            "time_s": first_audio["time_s"],
            "event": {
                "type": event_type,
                "event_id": "foreign",
                "response_id": "resp_0",
            },
        },
    )
    result = evaluate_trace(trace, profile=TURN_BASED)
    assert result["status"] == "fail"
    assert any(violation in item for item in result["violations"]), result


def test_turn_based_and_other_profiles_reject_each_other() -> None:
    assert evaluate_trace(turn_based_trace(), profile=NATIVE)["status"] == "fail"
    assert evaluate_trace(turn_based_trace(), profile=LEGACY)["status"] == "fail"


def write_trace(path: Path, records: list[dict[str, JsonValue]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def reconstruction_records(status: str) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = []
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

    receive(
        10.5,
        "response.output_audio.delta",
        response_id="resp_0",
        delta=encode(samples(2400, 1)),
    )
    receive(
        10.55,
        "response.output_audio.delta",
        response_id="resp_0",
        delta=encode(samples(2400, 2)),
    )
    receive(
        10.6,
        "response.output_audio_transcript.delta",
        response_id="resp_0",
        delta="hi",
    )
    receive(10.605, "response.audio.delta", delta=encode(samples(2400, 9)))
    receive(10.61, "input_audio_buffer.speech_started", audio_start_ms=600)
    receive(
        10.62,
        "response.done",
        response={
            "id": "resp_0",
            "status": status,
            "status_details": {
                "reason": "interrupted" if status == "cancelled" else "stop"
            },
        },
    )
    receive(
        10.7,
        "response.output_audio.delta",
        response_id="resp_1",
        delta=encode(samples(1000, 3)),
    )
    records.sort(key=lambda record: record["time_s"])
    return records


@pytest.mark.parametrize("status", ["cancelled", "completed"])
def test_reconstruction_cuts_playout_at_a_cancelled_response(
    tmp_path: Path, status: str
) -> None:
    write_trace(tmp_path / "continuous.jsonl", reconstruction_records(status))

    summary = reconstruct_output(tmp_path, profile=TURN_BASED)

    assert summary["errors"] == []
    assert summary["sample_rate"] == OUTPUT_RATE
    assert summary["media_samples"] == 5800
    cut = status == "cancelled"
    assert summary["barge_in"] == {"rule": TURN_BASED_BARGE_IN_CUT, "cuts": int(cut)}
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
                "type": "response.done",
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
    assert transcript["text"] == {"response.output_audio_transcript.delta": "hi"}


def make_turn_based_run(
    root: Path,
    input_pcm: bytes,
    *,
    profile: str = TURN_BASED,
    verdicts: dict[str, str] | None = None,
    done_at_s: float | None = None,
    done_status: str = "cancelled",
) -> Path:
    """A paired capture whose traces speak the turn-based protocol at 24 kHz."""
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
                "session": configure({"type": "server_vad"}),
            },
        ),
        (
            999.95,
            "receive",
            {
                "type": "session.updated",
                "session": {
                    "id": "s",
                    "audio": {
                        "output": {"format": {"type": "audio/pcm", "rate": OUTPUT_RATE}}
                    },
                },
            },
        ),
        (
            1000.1,
            "receive",
            {
                "type": "response.output_audio.delta",
                "response_id": "resp_0",
                "delta": encode(samples(round(0.3 * OUTPUT_RATE), 1000)),
            },
        ),
        (
            1000 + window_s + 0.1,
            "receive",
            {"type": "sglang.input_audio.drained", "padding_ms": 0},
        ),
        (
            1000 + window_s + 0.15,
            "receive",
            {"type": "session.closed", "reason": "client_closed"},
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
    if done_at_s is not None:
        rows.append(
            (
                1000 + done_at_s,
                "receive",
                {
                    "type": "response.done",
                    "response": {"id": "resp_0", "status": done_status},
                },
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


def test_export_reads_the_declared_rate_and_gates_on_the_verdict(
    tmp_path: Path,
) -> None:
    run = make_turn_based_run(tmp_path, samples(8000, 0), verdicts={"clean": "fail"})

    result = export_runs(
        [run],
        tmp_path / "export",
        "turn-based",
        trace_format=TraceFormat.TURN_BASED_PCM16.value,
    )

    assert result["trace_format"] == "realtime-turn-based-pcm16-v1"
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


@pytest.mark.parametrize("status", ["cancelled", "completed"])
def test_export_cuts_the_window_at_a_cancelled_response(
    tmp_path: Path, status: str
) -> None:
    run = make_turn_based_run(
        tmp_path, samples(8000, 0), done_at_s=0.2, done_status=status
    )

    result = export_runs(
        [run],
        tmp_path / "export",
        "turn-based",
        trace_format="realtime-turn-based-pcm16-v1",
    )

    overlap = result["samples"][0]["variants"]["overlap"]
    assert overlap["eligible"] is True
    cut = status == "cancelled"
    assert overlap["output"]["barge_in_cuts"] == int(cut)
    audio, _ = soundfile.read(
        tmp_path / "export/user_interruption/1/output.wav", dtype="int16"
    )
    assert audio[1700:3100].min() > 900
    if cut:
        assert not audio[3300:].any()
    else:
        assert audio[3300:6300].min() > 900


def test_export_refuses_a_trace_format_that_contradicts_the_profile(
    tmp_path: Path,
) -> None:
    pcm = samples(8000, 0)
    turn_based_run = make_turn_based_run(tmp_path / "turn-based", pcm)
    with pytest.raises(
        ValueError,
        match="recorded the turn_based protocol; export it with --trace-format "
        "realtime-turn-based-pcm16-v1",
    ):
        export_runs([turn_based_run], tmp_path / "native-export", "turn-based")
    assert not (tmp_path / "native-export").exists()
    with pytest.raises(ValueError, match="did not record the legacy protocol"):
        export_runs(
            [turn_based_run],
            tmp_path / "legacy-export",
            "turn-based",
            trace_format="realtime-legacy-pcm16-v1",
        )
    for profile in (NATIVE, LEGACY, "sglang"):
        other_run = make_turn_based_run(tmp_path / profile, pcm, profile=profile)
        with pytest.raises(ValueError, match="did not record the turn_based protocol"):
            export_runs(
                [other_run],
                tmp_path / f"{profile}-export",
                "turn-based",
                trace_format="realtime-turn-based-pcm16-v1",
            )


def test_export_cli_accepts_the_turn_based_trace_format(tmp_path: Path) -> None:
    run = make_turn_based_run(tmp_path, samples(8000, 0))
    output = tmp_path / "export"
    assert (
        reference_main(
            [
                "export",
                "--engine",
                "turn-based",
                "--trace-format",
                "realtime-turn-based-pcm16-v1",
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
def turn_based_peer_server(mode: PeerMode = "interrupt") -> Iterator[str]:
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    state: dict = {}

    async def handler(websocket: ServerConnection) -> None:
        await TurnBasedPeer(mode).handler(websocket)

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


RECORD_ARGS = [
    "--server-revision",
    SERVER_REVISION,
    "--model",
    "fixture/turn-based",
    "--dataset-revision",
    "fixture",
    "--timeout",
    "5",
    "--profile",
    TURN_BASED,
]


def test_v15_recorder_records_a_turn_based_profile(tmp_path: Path) -> None:
    dataset, run = tmp_path / "data", tmp_path / "run"
    write_dataset(dataset)
    stdout = io.StringIO()
    with turn_based_peer_server() as url, contextlib.redirect_stdout(stdout):
        code = v15_main(
            ["record", "--dataset-root", str(dataset), "--url", url]
            + ["--output", str(run), "--sample-id", "user_interruption/1"]
            + RECORD_ARGS
            + ["--turn-detection", '{"type": "server_vad", "silence_duration_ms": 500}']
        )
    summary = json.loads(stdout.getvalue())
    assert code == 0, summary
    assert summary["variant_status"] == {"pass": 2}
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["profile"] == TURN_BASED
    assert manifest["config"]["protocol"] == "turn_based"
    assert manifest["config"]["legacy"] is None
    turn_based_config = manifest["config"]["turn_based"]
    assert turn_based_config["turn_detection"] == {
        "type": "server_vad",
        "silence_duration_ms": 500,
    }
    assert turn_based_config["barge_in_cut"] == TURN_BASED_BARGE_IN_CUT
    assert "session_end" in turn_based_config
    result = json.loads((run / "run.json").read_text())
    variant = result["samples"][0]["variants"]["overlap"]
    directory = run / variant["directory"]
    variant_manifest = json.loads((directory / "manifest.json").read_text())
    assert variant_manifest["config"]["turn_based"] == turn_based_config
    assert soundfile.info(directory / "output-media.wav").samplerate == OUTPUT_RATE
    report = json.loads((directory / "report.json").read_text())
    assert report["cases"][0]["metrics"]["responses_interrupted"] == 1


def test_v15_recorder_requires_turn_detection_for_a_turn_based_profile(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data"
    write_dataset(dataset)
    with pytest.raises(ValueError, match="turn-based profiles"):
        v15_main(
            ["record", "--dataset-root", str(dataset), "--url", "ws://127.0.0.1:1"]
            + ["--output", str(tmp_path / "missing")]
            + RECORD_ARGS
        )
    assert not (tmp_path / "missing").exists()


def run_v10(argv: list[str]) -> int:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = v10_main(argv)
    assert code == 0, stdout.getvalue()
    return code


def record_v10_interruption(tmp_path: Path) -> tuple[Path, Path]:
    dataset, run = tmp_path / "data", tmp_path / "run"
    write_v10_dataset(dataset)
    with turn_based_peer_server() as url:
        run_v10(
            ["record", "--dataset-root", str(dataset), "--url", url]
            + ["--output", str(run), "--sample-id", "synthetic_user_interruption/1"]
            + RECORD_ARGS
            + ["--turn-detection", '{"type": "server_vad"}']
        )
    return dataset, run


def export_v10_turn_based(dataset: Path, run: Path, output: Path) -> dict:
    run_v10(
        ["reference-export", "--run", str(run), "--dataset-root", str(dataset)]
        + ["--out", str(output), "--trace-format", TraceFormat.TURN_BASED_PCM16.value]
    )
    return json.loads((output / "manifest.json").read_text())


def test_v10_export_reads_a_turn_based_run_at_the_profile_rate(
    tmp_path: Path,
) -> None:
    dataset, run = record_v10_interruption(tmp_path)

    manifest = export_v10_turn_based(dataset, run, tmp_path / "export")

    assert manifest["trace_format"] == TraceFormat.TURN_BASED_PCM16.value
    assert manifest["counts"]["synthetic_user_interruption"] == {
        "selected": 1,
        "eligible": 1,
    }
    [row] = manifest["samples"]
    assert row["output"]["native_rate"] == OUTPUT_RATE
    _, rate = soundfile.read(
        tmp_path / "export/synthetic_user_interruption/1/output.wav", dtype="int16"
    )
    assert rate == INPUT_RATE


def test_v10_export_needs_a_passing_turn_based_protocol_verdict(
    tmp_path: Path,
) -> None:
    dataset, run = record_v10_interruption(tmp_path)
    state = json.loads((run / "run.json").read_text())
    state["samples"][0]["variants"]["input"]["protocol_verdict"] = "fail"
    (run / "run.json").write_text(json.dumps(state))

    manifest = export_v10_turn_based(dataset, run, tmp_path / "export")

    [row] = manifest["samples"]
    assert row["eligible"] is False
    assert row["reasons"] == ["recorder protocol verdict 'fail' is not pass"]
    assert not (tmp_path / "export/synthetic_user_interruption").exists()
