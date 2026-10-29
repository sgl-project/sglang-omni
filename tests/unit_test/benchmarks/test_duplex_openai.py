# SPDX-License-Identifier: Apache-2.0
"""OpenAI Realtime, OpenAI Live and Gemini Live wires: recording, canonical trace,
oracle, serving."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import websockets
from pydantic import JsonValue
from scipy.signal import resample_poly
from websockets.asyncio.server import ServerConnection

import benchmarks.duplex.openai_client as openai_client
from benchmarks.duplex.artifacts import server_identity
from benchmarks.duplex.client import TurnDetection, ZeroBufferPlayer, run_session
from benchmarks.duplex.openai_client import (
    DEFAULT_MODELS,
    DIALECTS,
    echo_mismatches,
    gemini_configure,
    provider_manifest,
)
from benchmarks.duplex.oracle import evaluate_trace
from benchmarks.duplex.profiles import PROFILES
from benchmarks.duplex.serving import score_session
from benchmarks.duplex.v15_audio import reconstruct_output
from benchmarks.eval import benchmark_duplex_v10, benchmark_duplex_v15

API_KEY = "sk-test-0123456789abcdef-never-on-disk"
SERVER_VAD = TurnDetection(type="server_vad", silence_duration_ms=800)
INPUT_RATE = 16000
# note (wenyao): 0.64 s of a 440 Hz tone, eight 80 ms packets.
PCM = (
    (3000 * np.sin(2 * np.pi * 440 * np.arange(10240) / INPUT_RATE))
    .astype("<i2")
    .tobytes()
)
TAIL_SILENCE_S = 0.16
REPLY_SECONDS = 0.5
Mode = Literal["clean", "cancel", "mismatch", "late"]


@pytest.fixture(autouse=True)
def provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    monkeypatch.delenv("OPENAI_DUPLEX_INSTRUCTIONS_FILE", raising=False)
    monkeypatch.setattr(openai_client, "LIVE_QUIET_S", 0.2)


def encode(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


class RealtimePeer:
    """Fake GA Realtime server: speech on the second append, a reply on the fourth.

    In cancel mode the reply is cancelled while it plays; in mismatch mode the
    session.updated echo names a different turn detector; in late mode the reply
    starts on the last append and finishes after the input end."""

    def __init__(self, mode: Mode = "clean") -> None:
        self.mode = mode
        self.received: list[dict[str, JsonValue]] = []
        self.authorization: str | None = None

    async def send(self, websocket: ServerConnection, event: dict) -> None:
        await websocket.send(
            json.dumps({**event, "event_id": f"server_{len(self.received)}"})
        )

    async def handler(self, websocket: ServerConnection) -> None:
        self.authorization = websocket.request.headers.get("Authorization")
        await self.send(websocket, {"type": "session.created", "session": {"id": "s"}})
        appends = 0
        async for frame in websocket:
            event = json.loads(frame)
            self.received.append(event)
            if event["type"] == "session.update":
                session = json.loads(json.dumps(event["session"]))
                if self.mode == "mismatch":
                    session["audio"]["input"]["turn_detection"]["type"] = "semantic_vad"
                else:
                    pass
                await self.send(
                    websocket,
                    {"type": "session.updated", "session": {"id": "s", **session}},
                )
                await self.send(websocket, {"type": "rate_limits.updated"})
                await self.send(websocket, {"type": "provider.future_event"})
            elif event["type"] == "input_audio_buffer.append":
                appends += 1
                if appends == 2:
                    await self.send(
                        websocket,
                        {"type": "input_audio_buffer.speech_started", "item_id": "u1"},
                    )
                elif appends == (10 if self.mode == "late" else 4):
                    await self.reply(websocket)
                else:
                    pass
            elif event["type"] == "conversation.item.truncate":
                await self.send(
                    websocket,
                    {
                        "type": "conversation.item.truncated",
                        "item_id": event["item_id"],
                        "content_index": 0,
                        "audio_end_ms": event["audio_end_ms"],
                    },
                )
            else:
                pass

    async def reply(self, websocket: ServerConnection) -> None:
        await self.send(
            websocket, {"type": "input_audio_buffer.speech_stopped", "item_id": "u1"}
        )
        await self.send(
            websocket,
            {
                "type": "response.created",
                "response": {"id": "r1", "status": "in_progress"},
            },
        )
        await self.send(
            websocket,
            {
                "type": "response.output_audio.delta",
                "response_id": "r1",
                "item_id": "a1",
                "output_index": 0,
                "content_index": 0,
                "delta": encode(bytes(2 * round(24000 * REPLY_SECONDS))),
            },
        )
        cancelled = self.mode == "cancel"
        if cancelled or self.mode == "late":
            await asyncio.sleep(0.2 if cancelled else 0.4)
        else:
            pass
        await self.send(
            websocket,
            {
                "type": "response.done",
                "response": {
                    "id": "r1",
                    "status": "cancelled" if cancelled else "completed",
                    "status_details": (
                        {"type": "cancelled", "reason": "turn_detected"}
                        if cancelled
                        else None
                    ),
                    "usage": {
                        "total_tokens": 30,
                        "input_token_details": {"audio_tokens": 10, "text_tokens": 5},
                        "output_token_details": {"audio_tokens": 15},
                    },
                },
            },
        )


class LivePeer:
    """Fake GPT-Live server: speaks over the user from the second append; echo
    overrides fields of the session that session.started echoes."""

    def __init__(self, echo: dict[str, JsonValue] | None = None) -> None:
        self.received: list[dict[str, JsonValue]] = []
        self.echo = echo or {}

    async def handler(self, websocket: ServerConnection) -> None:
        async for frame in websocket:
            event = json.loads(frame)
            self.received.append(event)
            outgoing = []
            if event["type"] == "session.start":
                outgoing = [
                    {
                        "type": "session.started",
                        "client_event_id": event["event_id"],
                        "session": {
                            "id": "live_1",
                            "status": "active",
                            "input": [],
                            **event["session"],
                            **self.echo,
                        },
                    }
                ]
            elif event["type"] == "session.input_audio.append":
                appends = sum(
                    received["type"] == "session.input_audio.append"
                    for received in self.received
                )
                outgoing = (
                    [
                        {
                            "type": "session.output_audio.delta",
                            "delta": encode(bytes(2 * 4000)),
                        },
                        {
                            "type": "session.output_transcript.delta",
                            "delta": "hi",
                            "start_ms": 160,
                            "end_ms": 400,
                        },
                        {"type": "response.event", "event": {"type": "tool.call"}},
                        {"type": "session.usage.updated", "usage": {"seconds": 0.2}},
                    ]
                    if appends in (2, 3)
                    else []
                )
            elif event["type"] == "session.close":
                outgoing = [
                    {
                        "type": "session.closed",
                        "reason": "close_requested",
                        "usage": {"seconds": 1.5},
                    }
                ]
            else:
                pass
            for index, server_event in enumerate(outgoing):
                await websocket.send(
                    json.dumps(
                        {
                            "event_id": f"live_{len(self.received)}_{index}",
                            **server_event,
                        }
                    )
                )
            if event["type"] == "session.close":
                return
            else:
                pass


async def record_session(
    handler, variant_dir: Path, profile: str, **session_args
) -> list[dict[str, JsonValue]]:
    variant_dir.mkdir(parents=True, exist_ok=True)
    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        await run_session(
            f"ws://127.0.0.1:{port}/v1/session",
            PCM,
            trace_path=variant_dir / "continuous.jsonl",
            timeout_s=10.0,
            profile=profile,
            tail_silence_s=TAIL_SILENCE_S,
            **session_args,
        )
    return read_jsonl(variant_dir / "continuous.jsonl")


def read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def realtime(peer: RealtimePeer, variant_dir: Path, **session_args):
    return asyncio.run(
        record_session(
            peer.handler,
            variant_dir,
            "openai-realtime",
            turn_detection=SERVER_VAD,
            **session_args,
        )
    )


def live(peer: LivePeer, variant_dir: Path):
    return asyncio.run(record_session(peer.handler, variant_dir, "openai-live"))


class GeminiPeer:
    """Fake Gemini Live server: speech from the second append, a reply at the fourth
    that the user interrupts at the sixth, and a second reply at the eighth. Like
    the real server it answers in binary frames and closes with 1007 on an unknown
    field."""

    def __init__(self) -> None:
        self.received: list[dict[str, JsonValue]] = []
        self.api_key: str | None = None
        self.authorization: str | None = None

    async def handler(self, websocket: ServerConnection) -> None:
        self.api_key = websocket.request.headers.get("x-goog-api-key")
        self.authorization = websocket.request.headers.get("Authorization")
        reply = {"inlineData": {"mimeType": "audio/pcm;rate=24000"}}
        reply["inlineData"]["data"] = encode(bytes(2 * round(24000 * REPLY_SECONDS)))
        usage = {"usageMetadata": {"promptTokenCount": 7, "responseTokenCount": 3}}
        async for frame in websocket:
            event = json.loads(frame)
            self.received.append(event)
            audio = (event.get("realtimeInput") or {}).get("audio") or {}
            if set(event) == {"setup"}:
                outgoing = [{"setupComplete": {}}]
            elif set(event) == {"realtimeInput"} and set(audio) == {"data", "mimeType"}:
                appends = len(self.received) - 1
                outgoing = {
                    2: [{"voiceActivity": {"type": "ACTIVITY_START"}}],
                    4: [
                        {"voiceActivity": {"type": "ACTIVITY_END"}},
                        {"serverContent": {"outputTranscription": {"text": "hi"}}},
                        {"serverContent": {"modelTurn": {"parts": [reply]}}},
                        {"serverContent": {"generationComplete": True}},
                    ],
                    6: [
                        {"voiceActivity": {"type": "ACTIVITY_START"}},
                        {"serverContent": {"interrupted": True}},
                        {"serverContent": {"turnComplete": True}, **usage},
                    ],
                    8: [
                        {"voiceActivity": {"type": "ACTIVITY_END"}},
                        {"serverContent": {"modelTurn": {"parts": [reply]}}},
                        {"serverContent": {"turnComplete": True}, **usage},
                    ],
                }.get(appends, [])
            else:
                await websocket.close(1007, "Invalid JSON payload received.")
                return
            for server_event in outgoing:
                await websocket.send(json.dumps(server_event).encode())


def gemini(peer: GeminiPeer, variant_dir: Path, **session_args):
    return asyncio.run(
        record_session(
            peer.handler,
            variant_dir,
            "gemini-live",
            turn_detection=SERVER_VAD,
            **session_args,
        )
    )


def sent(records, event_type: str) -> list[dict[str, JsonValue]]:
    return [
        record["event"]
        for record in records
        if record["direction"] == "send" and record["event"]["type"] == event_type
    ]


def test_realtime_session_echoing_another_turn_detector_fails(tmp_path: Path) -> None:
    records = realtime(RealtimePeer("mismatch"), tmp_path)

    errors = [record for record in records if record["direction"] == "error"]
    assert len(errors) == 1
    assert "semantic_vad" in errors[0]["event"]["message"]
    assert sent(records, "input_audio_buffer.append") == []
    assert evaluate_trace(records, profile="openai-realtime")["status"] == "fail"


def test_realtime_canonical_appends_are_16k_and_the_wire_is_24k(
    tmp_path: Path,
) -> None:
    peer = RealtimePeer()
    records = realtime(peer, tmp_path)

    stream = PCM + bytes(round(TAIL_SILENCE_S * INPUT_RATE) * 2)
    canonical = sent(records, "input_audio_buffer.append")
    assert b"".join(base64.b64decode(event["audio"]) for event in canonical) == stream
    assert [event["sglang"]["seq"] for event in canonical] == list(range(10))
    assert [event["sglang"]["t_start_ms"] for event in canonical] == [
        80.0 * index for index in range(10)
    ]

    on_wire = [
        event for event in peer.received if event["type"] == "input_audio_buffer.append"
    ]
    assert all(set(event) == {"type", "event_id", "audio"} for event in on_wire)
    assert [event["event_id"] for event in on_wire] == [
        event["event_id"] for event in canonical
    ]
    wire_pcm = [base64.b64decode(event["audio"]) for event in on_wire]
    assert {len(chunk) for chunk in wire_pcm} == {1920 * 2}
    expected = resample_poly(np.frombuffer(stream, dtype="<i2"), 3, 2)
    received = np.frombuffer(b"".join(wire_pcm), dtype="<i2")
    assert received.shape == expected.shape
    assert np.max(np.abs(received - expected)) <= 1

    wire_records = read_jsonl(tmp_path / "wire.jsonl")
    assert [
        record["event"]
        for record in wire_records
        if record["direction"] == "send"
        and record["event"]["type"] == "input_audio_buffer.append"
    ] == on_wire
    session = sent(records, "session.update")[0]["session"]
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["voice"] == "marin"
    rtt = json.loads((tmp_path / "rtt.json").read_text())
    assert rtt["samples_s"] and rtt["p50_s"] <= rtt["p90_s"]


def test_realtime_cancel_truncates_at_the_played_audio(tmp_path: Path) -> None:
    peer = RealtimePeer("cancel")
    records = realtime(peer, tmp_path, report_playback=True)

    (truncate,) = [
        event
        for event in peer.received
        if event["type"] == "conversation.item.truncate"
    ]
    received_at = {
        record["event"]["type"]: record["time_s"]
        for record in records
        if record["direction"] == "receive"
    }
    played_s = received_at["response.done"] - received_at["response.output_audio.delta"]
    assert truncate["item_id"] == "a1"
    assert truncate["content_index"] == 0
    assert truncate["audio_end_ms"] == int(played_s * 1000)
    assert 150 <= truncate["audio_end_ms"] < REPLY_SECONDS * 1000
    assert evaluate_trace(records, profile="openai-realtime")["status"] == "pass"


def test_a_cut_truncates_each_audio_item_of_a_reply_at_its_own_played_audio() -> None:
    # note (wenyao): gpt-realtime-2.1 speaks a commentary item, then its final answer,
    # in one response.
    player = ZeroBufferPlayer(
        output_rate=24000, cuts=PROFILES["openai-realtime"].playout_cuts
    )

    def delta(item_id: str, seconds: float) -> dict[str, JsonValue]:
        return {
            "type": "response.output_audio.delta",
            "response_id": "r1",
            "item_id": item_id,
            "delta": encode(bytes(int(2 * 24000 * seconds))),
        }

    assert player.played(delta("commentary", 3.0), 10.0) == []
    assert player.played(delta("final_answer", 1.0), 12.25) == []
    speech = {"type": "input_audio_buffer.speech_started", "item_id": "u2"}
    assert player.played(speech, 12.5) == [
        {"item_id": "commentary", "content_index": 0, "audio_end_ms": 2500},
        {"item_id": "final_answer", "content_index": 0, "audio_end_ms": 0},
    ]
    cancelled = {
        "type": "response.done",
        "response": {"id": "r1", "status": "cancelled"},
    }
    assert player.played(cancelled, 12.6) == []


def test_realtime_closes_only_after_the_open_reply_finishes(tmp_path: Path) -> None:
    records = realtime(RealtimePeer("late"), tmp_path)

    first_append_s = next(
        record["time_s"]
        for record in records
        if record["direction"] == "send"
        and record["event"]["type"] == "input_audio_buffer.append"
    )
    input_end_s = first_append_s + len(PCM) / 2 / INPUT_RATE + TAIL_SILENCE_S
    (done_s,) = [
        record["time_s"]
        for record in records
        if record["direction"] == "receive"
        and record["event"]["type"] == "response.done"
    ]
    assert done_s > input_end_s
    assert [record for record in records if record["direction"] == "error"] == []
    assert evaluate_trace(records, profile="openai-realtime")["status"] == "pass"


@pytest.mark.parametrize(
    ("echo", "mismatch"),
    [
        (
            {"delegation": {"type": "responses"}},
            "delegation.type echoed 'responses', requested 'client'",
        ),
        (
            {"audio": {"format": {"type": "audio/pcm", "rate": 24000}}},
            "audio.format.rate echoed 24000, requested 16000",
        ),
    ],
)
def test_live_session_fails_when_the_start_is_not_echoed(
    tmp_path: Path, echo: dict[str, JsonValue], mismatch: str
) -> None:
    records = live(LivePeer(echo), tmp_path)

    (error,) = [record for record in records if record["direction"] == "error"]
    assert error["event"]["message"] == f"RuntimeError: session.updated: {mismatch}"
    assert sent(records, "input_audio_buffer.append") == []
    assert f"session.updated: {mismatch}" in (
        evaluate_trace(records, profile="openai-live")["violations"]
    )


def test_echo_checks_only_the_settings_the_request_set() -> None:
    requested = {"model": "gpt-live-1", "audio": {"format": {"type": "audio/pcm"}}}
    echoed = {
        "audio": {"format": {"type": "audio/pcm", "rate": 24000}},
        "delegation": {"type": "client"},
    }

    assert echo_mismatches(DIALECTS["openai-live"], requested, echoed) == []
    assert echo_mismatches(
        DIALECTS["openai-live"], {**requested, "delegation": {"type": "client"}}, {}
    ) == [
        "audio.format.type echoed None, requested 'audio/pcm'",
        "delegation.type echoed None, requested 'client'",
    ]


def test_live_audio_is_recorded_as_canonical_deltas(tmp_path: Path) -> None:
    peer = LivePeer()
    records = live(peer, tmp_path)

    start = next(event for event in peer.received if event["type"] == "session.start")
    assert start["session"]["audio"]["format"] == {"type": "audio/pcm", "rate": 16000}
    assert start["session"]["delegation"] == {"type": "client"}
    on_wire = [
        event
        for event in peer.received
        if event["type"] == "session.input_audio.append"
    ]
    assert b"".join(base64.b64decode(event["audio"]) for event in on_wire) == (
        PCM + bytes(round(TAIL_SILENCE_S * INPUT_RATE) * 2)
    )
    assert peer.received[-1]["type"] == "session.close"

    received = [
        record["event"] for record in records if record["direction"] == "receive"
    ]
    audio = [
        event for event in received if event["type"] == "response.output_audio.delta"
    ]
    assert len(audio) == 2
    assert all(
        event["response_id"] is None
        and event["item_id"] is None
        and event["wire_type"] == "session.output_audio.delta"
        and base64.b64decode(event["delta"]) == bytes(2 * 4000)
        for event in audio
    )
    transcripts = [
        event
        for event in received
        if event["type"] == "response.output_audio_transcript.delta"
    ]
    assert [event["delta"] for event in transcripts] == ["hi", "hi"]
    assert any(event["type"] == "response.event" for event in received)
    wire_types = {
        record["event"]["type"] for record in read_jsonl(tmp_path / "wire.jsonl")
    }
    assert {
        "session.output_audio.delta",
        "session.output_transcript.delta",
    } <= wire_types
    assert "response.output_audio.delta" not in wire_types

    playout = reconstruct_output(tmp_path, profile="openai-live")
    assert playout["errors"] == []
    assert playout["media_samples"] == 2 * 4000


def test_oracle_passes_clean_sessions_with_unknown_server_events(
    tmp_path: Path,
) -> None:
    realtime_records = realtime(RealtimePeer(), tmp_path / "realtime")
    live_records = live(LivePeer(), tmp_path / "live")

    realtime_report = evaluate_trace(realtime_records, profile="openai-realtime")
    live_report = evaluate_trace(live_records, profile="openai-live")

    assert realtime_report["status"] == "pass", realtime_report["violations"]
    assert realtime_report["metrics"]["server_events"]["provider.future_event"] == 1
    assert realtime_report["metrics"]["usage"] == {
        "total_tokens": 30,
        "input_token_details.audio_tokens": 10,
        "input_token_details.text_tokens": 5,
        "output_token_details.audio_tokens": 15,
    }
    assert realtime_report["metrics"]["admission_denials"] == 0
    assert live_report["status"] == "pass", live_report["violations"]
    assert live_report["metrics"]["server_events"]["response.event"] == 2
    assert live_report["metrics"]["usage"] == {"seconds": 1.5}


def test_oracle_fails_a_session_with_a_server_error(tmp_path: Path) -> None:
    records = realtime(RealtimePeer(), tmp_path)
    error_at = next(
        index
        for index, record in enumerate(records)
        if record["direction"] == "receive"
        and record["event"]["type"] == "rate_limits.updated"
    )
    records[error_at]["event"] = {
        "type": "error",
        "event_id": "e1",
        "error": {"code": "invalid_value", "message": "bad"},
    }

    report = evaluate_trace(records, profile="openai-realtime")

    assert report["status"] == "fail"
    assert report["violations"] == ["server error: code=invalid_value: bad"]


def test_the_api_key_reaches_only_the_socket(tmp_path: Path, monkeypatch) -> None:
    instructions = tmp_path / "instructions.txt"
    instructions.write_text("Answer briefly.", encoding="utf-8")
    monkeypatch.setenv("OPENAI_DUPLEX_INSTRUCTIONS_FILE", str(instructions))
    realtime_peer = RealtimePeer("cancel")
    variant_dir = tmp_path / "variant"
    realtime(realtime_peer, variant_dir / "realtime", report_playback=True)
    live(LivePeer(), variant_dir / "live")
    manifest = provider_manifest("openai-realtime", "wss://api.openai.com/v1/realtime")
    (variant_dir / "manifest.json").write_text(json.dumps(manifest))

    assert realtime_peer.authorization == f"Bearer {API_KEY}"
    assert sent(
        read_jsonl(variant_dir / "realtime" / "continuous.jsonl"), "session.update"
    )[0]["session"]["instructions"] == ("Answer briefly.")
    assert manifest["model"] == "gpt-realtime-2.1"
    assert manifest["url"] == "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1"
    assert len(manifest["instructions_sha256"]) == 64
    written = [path for path in variant_dir.rglob("*") if path.is_file()]
    assert {path.name for path in written} >= {
        "continuous.jsonl",
        "wire.jsonl",
        "rtt.json",
        "input-send-receipts.json",
        "manifest.json",
    }
    assert not [path for path in written if API_KEY.encode() in path.read_bytes()]


def test_serving_scores_realtime_and_refuses_live(tmp_path: Path) -> None:
    realtime(RealtimePeer(), tmp_path / "realtime")
    live(LivePeer(), tmp_path / "live")
    for variant_dir in (tmp_path / "realtime", tmp_path / "live"):
        (variant_dir / "input.pcm").write_bytes(PCM)

    scored = score_session(tmp_path / "realtime", "openai-realtime", 0.0)

    (reply,) = scored["replies"]
    assert reply["response_id"] == "r1"
    assert reply["audio_s"] == REPLY_SECONDS
    with pytest.raises(ValueError, match="openai-live"):
        score_session(tmp_path / "live", "openai-live", 0.0)


@pytest.mark.parametrize(
    ("profile", "path", "url"),
    [
        (
            "openai-realtime",
            "/v1/realtime",
            "ws://127.0.0.1:{port}/v1/realtime?model=gpt-realtime-2.1",
        ),
        ("openai-live", "/v1/live/sessions", "ws://127.0.0.1:{port}/v1/live/sessions"),
    ],
)
@pytest.mark.parametrize("served", [True, False])
def test_provider_identity_probes_the_model_with_the_key_and_records_no_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    path: str,
    url: str,
    served: bool,
) -> None:
    model = DEFAULT_MODELS[profile]
    requests: list[tuple[str, str | None]] = []

    class Models(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append((self.path, self.headers.get("Authorization")))
            body = json.dumps(
                {"id": model, "object": "model", "created": 1, "owned_by": "system"}
                if served
                else {"error": {"message": f"The model '{model}' does not exist"}}
            ).encode()
            self.send_response(200 if served else 404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    instructions = tmp_path / "instructions.txt"
    instructions.write_text("Answer briefly.", encoding="utf-8")
    monkeypatch.setenv("OPENAI_DUPLEX_INSTRUCTIONS_FILE", str(instructions))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Models)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        identity = server_identity(
            f"ws://127.0.0.1:{port}{path}",
            revision=None,
            model=model,
            runtime="openai-api region=test",
            profile=profile,
        )
    finally:
        server.shutdown()

    assert requests == [(f"/v1/models/{model}", f"Bearer {API_KEY}")]
    assert identity == {
        "identity_source": "provider_api",
        "wire": profile,
        "url": url.format(port=port),
        "model": model,
        "voice": "marin",
        "instructions_sha256": hashlib.sha256(b"Answer briefly.").hexdigest(),
        "runtime": "openai-api region=test",
        "recorded_date": datetime.now(timezone.utc).date().isoformat(),
        "served_model": (
            {"status": 200, "id": model} if served else {"status": 404, "id": None}
        ),
    }
    assert API_KEY not in json.dumps(identity)


def test_provider_identity_records_an_unreachable_probe() -> None:
    served = server_identity(
        "wss://127.0.0.1:1/v1/live/sessions",
        revision=None,
        model="gpt-live-1",
        profile="openai-live",
    )["served_model"]

    assert served["status"] is None and served["id"] is None
    assert served["error"].startswith("URLError")


def test_provider_identity_refuses_a_model_the_provider_will_not_serve() -> None:
    with pytest.raises(ValueError, match="differs from the provider model"):
        server_identity(
            "wss://api.openai.com/v1/realtime",
            revision=None,
            model="gpt-4o-realtime-preview",
            profile="openai-realtime",
        )


@pytest.mark.parametrize("module", [benchmark_duplex_v10, benchmark_duplex_v15])
@pytest.mark.parametrize(
    ("profile", "profile_args"),
    [
        ("openai-realtime", ["--turn-detection", '{"type": "server_vad"}']),
        ("openai-live", []),
        ("turn-based", ["--turn-detection", '{"type": "server_vad"}']),
    ],
)
def test_record_needs_a_server_revision_only_for_the_sglang_wire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    profile: str,
    profile_args: list[str],
) -> None:
    class Reached(Exception):
        pass

    recorded: list[dict[str, JsonValue]] = []

    async def stop_before_recording(*args, server, **kwargs):
        recorded.append(server)
        raise Reached

    monkeypatch.setattr(module, "run_samples", stop_before_recording)
    argv = [
        "record",
        "--profile",
        profile,
        *profile_args,
        "--url",
        "ws://127.0.0.1:1/v1/realtime",
        "--model",
        DEFAULT_MODELS.get(profile, "served-model"),
        "--dataset-root",
        str(tmp_path / "missing"),
        "--dataset-revision",
        "fixture",
        "--output",
        str(tmp_path / "run"),
    ]

    if profile == "turn-based":
        with pytest.raises(ValueError, match="server revision"):
            module.main(argv)
        assert recorded == []
    else:
        with pytest.raises(Reached):
            module.main(argv)
        (server,) = recorded
        assert server["identity_source"] == "provider_api"
        assert server["wire"] == profile


@pytest.mark.parametrize(
    ("profile", "id_field"),
    [("openai-realtime", "event_id"), ("openai-live", "client_event_id")],
)
def test_server_errors_quote_the_client_event_they_answer(
    tmp_path: Path, profile: str, id_field: str
) -> None:
    records = (
        realtime(RealtimePeer(), tmp_path)
        if profile == "openai-realtime"
        else live(LivePeer(), tmp_path)
    )
    culprit = sent(records, "input_audio_buffer.append")[3]["event_id"]
    error = {
        "type": "error",
        "event_id": "server_error_1",
        "error": {
            "type": "invalid_request_error",
            "code": "invalid_value",
            "message": "bad audio",
            "param": "audio",
            id_field: culprit,
        },
    }
    records.append(
        {"direction": "receive", "time_s": records[-1]["time_s"], "event": error}
    )

    report = evaluate_trace(records, profile=profile)

    assert report["status"] == "fail"
    assert report["violations"] == [
        f"server error: code=invalid_value: bad audio (client event {culprit})"
    ]


def test_a_live_session_without_output_audio_does_not_reconstruct(
    tmp_path: Path,
) -> None:
    records = live(LivePeer(), tmp_path)
    (tmp_path / "continuous.jsonl").write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in records
            if record["event"].get("type") != "response.output_audio.delta"
        )
    )

    assert (
        "no model output audio"
        in reconstruct_output(tmp_path, profile="openai-live")["errors"]
    )


def test_gemini_session_records_model_turns_and_a_barge_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", API_KEY)
    peer = GeminiPeer()
    records = gemini(peer, tmp_path)

    report = evaluate_trace(records, profile="gemini-live")
    assert report["status"] == "pass", report["violations"]
    assert report["metrics"]["responses_created"] == 2
    assert report["metrics"]["usage"] == {
        "promptTokenCount": 14,
        "responseTokenCount": 6,
    }
    assert peer.api_key == API_KEY and peer.authorization is None
    setup = peer.received[0]["setup"]
    assert setup["model"] == "models/gemini-3.8-live"
    assert setup["realtimeInputConfig"]["automaticActivityDetection"] == {
        "silenceDurationMs": 800
    }
    assert b"".join(
        base64.b64decode(event["realtimeInput"]["audio"]["data"])
        for event in peer.received[1:]
    ) == PCM + bytes(round(TAIL_SILENCE_S * INPUT_RATE) * 2)

    received = [
        record["event"] for record in records if record["direction"] == "receive"
    ]
    created = [event for event in received if event["type"] == "response.created"]
    done = [event["response"] for event in received if event["type"] == "response.done"]
    first_id, second_id = (event["response"]["id"] for event in created)
    assert [(response["id"], response["status"]) for response in done] == [
        (first_id, "cancelled"),
        (first_id, "cancelled"),
        (second_id, "completed"),
    ]
    transcript = next(
        event
        for event in received
        if event["type"] == "response.output_audio_transcript.delta"
    )
    assert transcript["response_id"] is None
    assert done[0]["output"][0]["content"] == [
        {"type": "output_audio", "transcript": "hi"}
    ]
    assert "gemini.generationComplete" in report["metrics"]["server_events"]

    playout = reconstruct_output(tmp_path, profile="gemini-live")
    reply_samples = round(24000 * REPLY_SECONDS)
    assert playout["errors"] == []
    assert playout["media_samples"] == 2 * reply_samples
    (cut,) = json.loads((tmp_path / "playout.json").read_text())["cuts"]
    assert cut["type"] == "response.done"
    assert 0 < cut["dropped_samples"] < reply_samples
    written = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert not [path for path in written if API_KEY.encode() in path.read_bytes()]
    (tmp_path / "input.pcm").write_bytes(PCM)
    assert score_session(tmp_path, "gemini-live", 0.0)["replies"]


def test_gemini_turn_detection_maps_only_what_activity_detection_has(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", API_KEY)
    provider = openai_client.Provider.from_env("gemini-live")
    setup = gemini_configure(
        provider,
        TurnDetection(
            type="server_vad",
            silence_duration_ms=500,
            prefix_padding_ms=300,
            interrupt_response=False,
        ),
    )
    assert setup["realtimeInputConfig"] == {
        "automaticActivityDetection": {
            "silenceDurationMs": 500,
            "prefixPaddingMs": 300,
        },
        "activityHandling": "NO_INTERRUPTION",
    }
    for turn_detection in (
        TurnDetection(type="semantic_vad"),
        TurnDetection(type="server_vad", threshold=0.5),
    ):
        with pytest.raises(ValueError):
            gemini_configure(provider, turn_detection)
    with pytest.raises(ValueError, match="no truncate"):
        asyncio.run(
            run_session(
                "ws://127.0.0.1:9/unused",
                PCM,
                trace_path=Path("unused.jsonl"),
                profile="gemini-live",
                turn_detection=SERVER_VAD,
                report_playback=True,
            )
        )
