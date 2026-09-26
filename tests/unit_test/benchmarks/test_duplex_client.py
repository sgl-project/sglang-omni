# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import hashlib
import itertools
import json
import shutil
import sys
import time
import wave
from http import HTTPStatus
from pathlib import Path
from typing import Literal

import pytest
import websockets
from websockets.asyncio.server import ServerConnection

from benchmarks.duplex.client import (
    ADMISSION_RETRIES,
    PACKET_MS,
    POST_CLOSE_SECONDS,
    run_session,
)
from benchmarks.duplex.oracle import evaluate_trace

FIXTURE_PCM = b"\x02\x00" * 10240
FIXTURE_PACKETS = 8
UNIT_BYTES = 2560
SAMPLES_PER_FRAME = 1764
TAIL_HOLDBACK_SAMPLES = 256
# Note (wenyao): Adapter teardown can delay close beyond the observation window.
FATAL_CLOSE_DELAY_S = POST_CLOSE_SECONDS * 3
PeerMode = Literal[
    "healthy",
    "server_error",
    "delayed_fatal",
    "nonfatal_error",
    "timeout",
    "malformed_json",
    "disconnect",
    "nonfinite",
    "linger",
    "close_without_update",
]

GRANTED = {
    "interaction": "native",
    "native_full_duplex": True,
    "proactive_output": False,
    "turn_control": [None],
    "client_commit": False,
    "input_modalities": ["audio"],
    "output_modalities": ["audio"],
    "input_audio_format": {"type": "audio/pcm", "rate": 16000},
    "output_audio_format": {"type": "audio/pcm", "rate": 22050},
    "native_unit_ms": 80,
    "first_unit_ms": 80,
    "microturn_ms": None,
    "tail_policy": "pad",
    "supports_server_interrupt": False,
    "supports_truncate": False,
    "supports_resume": False,
    "partial_style": "append_only",
    "pressure_policy": "reject",
    "strict_order": True,
    "rejections": [],
    "limits": {
        "max_input_bytes": 1920000,
        "max_input_chunks": 128,
        "max_output_bytes": 4194304,
        "max_output_events": 256,
        "max_responses": 128,
        "max_segments": 256,
        "max_history_chars": 65536,
        "session_timeout_s": 240,
        "cleanup_timeout_s": 30,
    },
}


class DuplexPeer:
    def __init__(self, mode: PeerMode = "healthy") -> None:
        self.mode = mode
        self.received: list[dict] = []
        self.sent: list[dict] = []
        self.pcm = bytearray()
        self.chunk_seq = 0
        self.frames = 0
        self.emitted = 0
        self.response_open = False
        self.release = asyncio.Event()

    def codec_fresh(self, frames: int, *, final: bool) -> int:
        available = frames * SAMPLES_PER_FRAME - (0 if final else TAIL_HOLDBACK_SAMPLES)
        fresh = available - self.emitted
        self.frames = frames
        self.emitted = available
        return fresh

    async def open_response(self, websocket: ServerConnection) -> None:
        if not self.response_open:
            await self.send(
                websocket,
                {
                    "type": "response.created",
                    "response": {
                        "id": "response_0",
                        "object": "realtime.response",
                        "status": "in_progress",
                        "output": [],
                    },
                },
            )
            self.response_open = True

    async def emit_audio(
        self, websocket: ServerConnection, samples: int, unit: int
    ) -> None:
        await self.send(
            websocket,
            {
                "type": "response.output_audio.delta",
                "response_id": "response_0",
                "item_id": "item_0",
                "output_index": 0,
                "content_index": 0,
                "delta": base64.b64encode(b"\x01\x00" * samples).decode(),
                "sglang": {
                    "unit_id": f"unit_{unit}",
                    "chunk_seq": self.chunk_seq,
                    "media_time": {
                        "t_start_ms": unit * PACKET_MS,
                        "duration_ms": PACKET_MS,
                    },
                },
            },
        )
        self.chunk_seq += 1

    async def emit_error(
        self, websocket: ServerConnection, event_id: str, *, fatal: bool
    ) -> None:
        await self.send(
            websocket,
            {
                "type": "error",
                "sglang": {"fatal": fatal},
                "error": {
                    "type": "server_error" if fatal else "invalid_request_error",
                    "code": "fixture_failure",
                    "message": "deliberate peer failure",
                    "event_id": event_id,
                    "param": None,
                },
            },
        )

    async def close_session(
        self, websocket: ServerConnection, reason: str, client_event_id: str | None
    ) -> None:
        await self.send(
            websocket,
            {
                "type": "session.closed",
                "reason": reason,
                "client_event_id": client_event_id,
            },
        )

    async def emit_unit_done(
        self, websocket: ServerConnection, unit: int, duration_ms: float
    ) -> None:
        await self.send(
            websocket,
            {
                "type": "sglang.unit.done",
                "unit_id": f"unit_{unit}",
                "sglang": {
                    "unit_id": f"unit_{unit}",
                    "media_time": {
                        "t_start_ms": unit * PACKET_MS,
                        "duration_ms": duration_ms,
                    },
                },
            },
        )

    async def send(self, websocket: ServerConnection, event: dict) -> None:
        event["event_id"] = f"server_{len(self.sent)}"
        self.sent.append(event)
        await websocket.send(json.dumps(event))

    async def finish_response(
        self, websocket: ServerConnection, status: str, reason: str
    ) -> None:
        await self.send(
            websocket,
            {
                "type": "response.output_audio.done",
                "response_id": "response_0",
                "item_id": "item_0",
                "output_index": 0,
                "content_index": 0,
            },
        )
        await self.send(
            websocket,
            {
                "type": "response.done",
                "response": {
                    "id": "response_0",
                    "object": "realtime.response",
                    "status": status,
                    "status_details": {"reason": reason},
                    "output": [
                        {
                            "id": "item_0",
                            "object": "realtime.item",
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_audio", "transcript": "fixture"}
                            ],
                        }
                    ],
                    "usage": {},
                },
            },
        )
        self.response_open = False

    async def handler(self, websocket: ServerConnection) -> None:
        await self.send(
            websocket,
            {
                "type": "session.created",
                "session": {
                    "id": "session_fixture",
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
                if self.mode == "close_without_update":
                    await self.close_session(
                        websocket, "client_closed", event["event_id"]
                    )
                    return
                await self.send(
                    websocket,
                    {
                        "type": "session.updated",
                        "client_event_id": event["event_id"],
                        "session": {
                            "id": "session_fixture",
                            "object": "realtime.session",
                            "type": "realtime",
                            "model": "fixture",
                            "output_modalities": ["audio"],
                            "sglang": {"granted": GRANTED},
                        },
                    },
                )
            elif kind == "input_audio_buffer.append":
                if self.mode == "server_error":
                    await self.emit_error(websocket, event["event_id"], fatal=True)
                    await self.close_session(websocket, "server_error", None)
                    return
                elif self.mode == "delayed_fatal":
                    await self.emit_error(websocket, event["event_id"], fatal=True)
                    await asyncio.sleep(FATAL_CLOSE_DELAY_S)
                    await self.close_session(websocket, "server_error", None)
                    return
                elif self.mode == "nonfatal_error":
                    await self.emit_error(websocket, event["event_id"], fatal=False)
                elif self.mode == "timeout":
                    await websocket.wait_closed()
                    return
                elif self.mode == "malformed_json":
                    await websocket.send("{not valid json")
                    await websocket.wait_closed()
                    return
                elif self.mode == "nonfinite":
                    await websocket.send(
                        '{"type":"sglang.unit.done","event_id":"server_nan",'
                        '"unit_id":"unit_0","sglang":{"backlog_ms":NaN}}'
                    )
                    await websocket.wait_closed()
                    return
                elif self.mode == "disconnect":
                    await websocket.close(code=1011, reason="deliberate disconnect")
                    return
                else:
                    self.pcm.extend(base64.b64decode(event["audio"], validate=True))
                    await self.send(
                        websocket,
                        {
                            "type": "sglang.input_audio.accepted",
                            "seq": event["sglang"]["seq"],
                            "accepted_end_ms": len(self.pcm) / 32,
                            "client_event_id": event["event_id"],
                        },
                    )
                    while (self.frames + 1) * UNIT_BYTES <= len(self.pcm):
                        unit = self.frames
                        await self.open_response(websocket)
                        await self.emit_audio(
                            websocket,
                            self.codec_fresh(unit + 1, final=False),
                            unit,
                        )
                        await self.emit_unit_done(websocket, unit, PACKET_MS)
            elif kind == "sglang.input_audio.end":
                accepted_ms = len(self.pcm) / 32
                units = -(-len(self.pcm) // UNIT_BYTES)
                await self.send(
                    websocket,
                    {
                        "type": "sglang.input_audio.ended",
                        "accepted_end_ms": accepted_ms,
                        "tail_policy": "pad",
                        "client_event_id": event["event_id"],
                    },
                )
                await self.open_response(websocket)
                tail_unit = self.frames if units > self.frames else units
                await self.emit_audio(
                    websocket, self.codec_fresh(units, final=True), tail_unit
                )
                await self.finish_response(websocket, "completed", "stop")
                await self.emit_unit_done(
                    websocket, tail_unit, accepted_ms - tail_unit * PACKET_MS
                )
                await self.send(
                    websocket,
                    {
                        "type": "sglang.input_audio.drained",
                        "accepted_end_ms": accepted_ms,
                        "consumed_ms": accepted_ms,
                        "discarded_ms": 0,
                        "padding_ms": units * PACKET_MS - accepted_ms,
                        "client_event_id": event["event_id"],
                    },
                )
            elif kind == "session.close":
                await self.close_session(websocket, "client_closed", event["event_id"])
                if self.mode == "linger":
                    await self.release.wait()
                return
            else:
                raise AssertionError(f"Unexpected client event: {kind}")


async def capture(
    peer: DuplexPeer, scenario: str, trace_path: Path, timeout_s: float = 2.0
) -> list[dict]:
    async with websockets.serve(peer.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        await run_session(
            f"ws://127.0.0.1:{port}/v1/realtime",
            FIXTURE_PCM,
            scenario=scenario,
            trace_path=trace_path,
            timeout_s=timeout_s,
        )
        peer.release.set()
        assert not [
            task
            for task in asyncio.all_tasks()
            if getattr(task.get_coro(), "__name__", "")
            in ("exchange", "receive", "drive")
        ]
    return [json.loads(line) for line in trace_path.read_text().splitlines()]


def appended(records: list[dict]) -> list[dict]:
    return [
        record
        for record in records
        if record["direction"] == "send"
        and record["event"]["type"] == "input_audio_buffer.append"
    ]


def output_samples(records: list[dict]) -> int:
    return sum(
        len(base64.b64decode(record["event"]["delta"], validate=True)) // 2
        for record in records
        if record["direction"] == "receive"
        and record["event"]["type"] == "response.output_audio.delta"
    )


def test_client_records_complete_native_session(tmp_path: Path) -> None:
    peer = DuplexPeer()
    records = asyncio.run(capture(peer, "continuous", tmp_path / "trace.jsonl"))
    result = evaluate_trace(records, scenario="continuous")

    assert result["status"] == "pass", result
    assert bytes(peer.pcm) == FIXTURE_PCM
    assert [r["event"] for r in records if r["direction"] == "send"] == peer.received
    assert [r["event"] for r in records if r["direction"] == "receive"] == peer.sent
    assert [r["time_s"] for r in records] == sorted(r["time_s"] for r in records)
    event_ids = [event["event_id"] for event in peer.received]
    assert len(set(event_ids)) == len(event_ids)
    assert peer.received[0]["session"]["output_modalities"] == ["audio"]
    appends = appended(records)
    assert [r["event"]["sglang"]["seq"] for r in appends] == list(
        range(FIXTURE_PACKETS)
    )
    assert result["metrics"]["input_output_overlap"] is True

    # Note (wenyao): Absolute pacing permits catch-up packets, so measure total span.
    paced_s = (FIXTURE_PACKETS - 1) * PACKET_MS / 1000
    span_s = appends[-1]["time_s"] - appends[0]["time_s"]
    assert paced_s * 0.8 <= span_s <= paced_s + 1.0, span_s

    units = -(-len(FIXTURE_PCM) // UNIT_BYTES)
    assert output_samples(records) == units * SAMPLES_PER_FRAME
    assert result["metrics"]["output_audio_s"] == pytest.approx(
        units * SAMPLES_PER_FRAME / 22050
    )

    assert all("epoch" not in event.get("sglang", {}) for event in peer.sent)
    assert "held" not in peer.sent[-1]
    assert "cancel_is_noop" not in GRANTED
    assert "response.cancel" not in [event["type"] for event in peer.received]


def test_removed_cancel_scenario_is_rejected_before_connection(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    with pytest.raises(ValueError, match="unsupported scenario: cancel_resume"):
        asyncio.run(
            run_session(
                "ws://127.0.0.1:1/v1/realtime",
                FIXTURE_PCM,
                scenario="cancel_resume",
                trace_path=path,
            )
        )
    assert not path.exists()


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    [
        ("malformed_json", "{not valid json"),
        ("nonfinite", "backlog_ms"),
        ("disconnect", "ConnectionClosedError"),
    ],
)
def test_client_retains_failed_attempt(
    tmp_path: Path, mode: str, expected_error: str
) -> None:
    peer = DuplexPeer(mode)
    records = asyncio.run(capture(peer, "continuous", tmp_path / "trace.jsonl"))
    result = evaluate_trace(records, scenario="continuous")

    assert result["status"] == "fail", result
    assert result["violations"]
    assert any(r["event"]["type"] == "session.created" for r in records)
    assert appended(records)
    assert any(
        r["direction"] == "error" and expected_error in r["event"]["message"]
        for r in records
    )


def test_client_keeps_the_close_receipt_after_a_fatal_server_error(
    tmp_path: Path,
) -> None:
    peer = DuplexPeer("server_error")
    records = asyncio.run(capture(peer, "continuous", tmp_path / "trace.jsonl"))
    result = evaluate_trace(records, scenario="continuous")

    assert result["status"] == "fail", result
    assert any(
        r["direction"] == "receive"
        and r["event"].get("error", {}).get("code") == "fixture_failure"
        for r in records
    )
    assert records[-1]["direction"] == "receive"
    assert records[-1]["event"]["type"] == "session.closed"
    assert any("server error" in violation for violation in result["violations"])
    assert len(appended(records)) < FIXTURE_PACKETS


def test_client_waits_out_a_late_close_after_a_fatal_error(tmp_path: Path) -> None:
    peer = DuplexPeer("delayed_fatal")
    started_s = time.perf_counter()
    records = asyncio.run(
        capture(peer, "continuous", tmp_path / "trace.jsonl", timeout_s=10.0)
    )
    elapsed_s = time.perf_counter() - started_s
    result = evaluate_trace(records, scenario="continuous")

    assert FATAL_CLOSE_DELAY_S < elapsed_s < 5.0, elapsed_s
    assert result["status"] == "fail", result
    assert [(r["direction"], r["event"].get("type")) for r in records][-2:] == [
        ("receive", "error"),
        ("receive", "session.closed"),
    ]
    assert not [
        r
        for r in records
        if r["direction"] == "send" and r["event"]["type"] == "session.close"
    ]


def test_client_requests_close_after_a_nonfatal_error(tmp_path: Path) -> None:
    peer = DuplexPeer("nonfatal_error")
    started_s = time.perf_counter()
    records = asyncio.run(
        capture(peer, "continuous", tmp_path / "trace.jsonl", timeout_s=10.0)
    )
    elapsed_s = time.perf_counter() - started_s
    result = evaluate_trace(records, scenario="continuous")

    assert elapsed_s < 5.0, elapsed_s
    assert result["status"] == "fail", result
    assert any(
        r["direction"] == "receive"
        and r["event"]["type"] == "error"
        and r["event"]["sglang"]["fatal"] is False
        for r in records
    )
    closes = [
        r
        for r in records
        if r["direction"] == "send" and r["event"]["type"] == "session.close"
    ]
    assert len(closes) == 1
    assert records[-1]["direction"] == "receive"
    assert records[-1]["event"]["type"] == "session.closed"
    assert len(appended(records)) < FIXTURE_PACKETS


async def capture_with_denials(
    peer: DuplexPeer, denials: int, trace_path: Path
) -> list[dict]:
    remaining = itertools.count()

    def process_request(connection: ServerConnection, request) -> object | None:
        if next(remaining) < denials:
            return connection.respond(
                HTTPStatus.SERVICE_UNAVAILABLE, "connection capacity exhausted\n"
            )
        else:
            return None

    async with websockets.serve(
        peer.handler, "127.0.0.1", 0, process_request=process_request
    ) as server:
        port = server.sockets[0].getsockname()[1]
        await run_session(
            f"ws://127.0.0.1:{port}/v1/realtime",
            FIXTURE_PCM,
            scenario="continuous",
            trace_path=trace_path,
            timeout_s=10.0,
        )
    return [json.loads(line) for line in trace_path.read_text().splitlines()]


def test_client_retries_a_denied_handshake_and_still_records_a_clean_session(
    tmp_path: Path,
) -> None:
    peer = DuplexPeer()
    records = asyncio.run(
        capture_with_denials(peer, ADMISSION_RETRIES, tmp_path / "trace.jsonl")
    )
    admissions = [r for r in records if r["direction"] == "admission"]

    assert [r["event"] for r in admissions] == [
        {"type": "connection_denied", "http_status": 503, "attempt": attempt}
        for attempt in range(1, ADMISSION_RETRIES + 1)
    ]
    assert records[: len(admissions)] == admissions
    assert not [r for r in records if r["direction"] == "error"]
    assert bytes(peer.pcm) == FIXTURE_PCM
    result = evaluate_trace(records, scenario="continuous")
    assert result["status"] == "pass", result
    assert result["metrics"]["admission_denials"] == ADMISSION_RETRIES


def test_client_fails_once_the_admission_budget_is_exhausted(tmp_path: Path) -> None:
    peer = DuplexPeer()
    records = asyncio.run(
        capture_with_denials(peer, ADMISSION_RETRIES + 1, tmp_path / "trace.jsonl")
    )

    assert [
        r["event"]["attempt"] for r in records if r["direction"] == "admission"
    ] == (list(range(1, ADMISSION_RETRIES + 1)))
    assert records[-1]["direction"] == "error"
    assert (
        records[-1]["event"]["message"]
        == f"RuntimeError: admission denied {ADMISSION_RETRIES + 1} times with HTTP 503"
    )
    assert not peer.sent
    result = evaluate_trace(records, scenario="continuous")
    assert result["status"] == "fail", result
    assert result["metrics"]["admission_denials"] == ADMISSION_RETRIES


def test_client_enforces_the_session_deadline(tmp_path: Path) -> None:
    peer = DuplexPeer("timeout")
    started_s = time.perf_counter()
    records = asyncio.run(
        capture(peer, "continuous", tmp_path / "trace.jsonl", timeout_s=0.5)
    )
    elapsed_s = time.perf_counter() - started_s

    assert 0.5 <= elapsed_s < 5.0, elapsed_s
    assert evaluate_trace(records, scenario="continuous")["status"] == "fail"
    assert any(r["event"]["type"] == "session.created" for r in records)
    assert appended(records)
    assert records[-1]["direction"] == "error"
    assert "Session timeout after 0.5s" in records[-1]["event"]["message"]


def test_client_bounds_observation_after_the_session_closes(tmp_path: Path) -> None:
    peer = DuplexPeer("linger")
    started_s = time.perf_counter()
    records = asyncio.run(
        capture(peer, "continuous", tmp_path / "trace.jsonl", timeout_s=10.0)
    )
    elapsed_s = time.perf_counter() - started_s
    streamed_s = FIXTURE_PACKETS * PACKET_MS / 1000

    assert elapsed_s < streamed_s + POST_CLOSE_SECONDS + 2.0, elapsed_s
    assert evaluate_trace(records, scenario="continuous")["status"] == "pass"
    assert records[-1]["event"]["type"] == "session.closed"


def test_client_abandons_a_driver_the_receive_loop_can_no_longer_serve(
    tmp_path: Path,
) -> None:
    peer = DuplexPeer("close_without_update")
    started_s = time.perf_counter()
    records = asyncio.run(
        capture(peer, "continuous", tmp_path / "trace.jsonl", timeout_s=10.0)
    )
    elapsed_s = time.perf_counter() - started_s

    assert elapsed_s < POST_CLOSE_SECONDS + 2.0, elapsed_s
    assert [(r["direction"], r["event"].get("type")) for r in records] == [
        ("receive", "session.created"),
        ("send", "session.update"),
        ("receive", "session.closed"),
        ("error", None),
    ]
    assert records[-1]["event"]["message"] == "driver stalled after the receive loop"
    assert evaluate_trace(records, scenario="continuous")["status"] == "fail"


async def run_cli(module: str, *arguments: str) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        module,
        *arguments,
        cwd=Path(__file__).resolve().parents[3],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    assert process.returncode is not None
    return process.returncode, stdout.decode(), stderr.decode()


def write_fixture_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(FIXTURE_PCM)


async def serve_cli(handler, run_dir: Path, wav_path: Path, timeout: str) -> tuple:
    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        return await run_cli(
            "benchmarks.eval.benchmark_duplex",
            "--url",
            f"ws://127.0.0.1:{port}/v1/realtime",
            "--audio",
            str(wav_path),
            "--output",
            str(run_dir),
            "--server-revision",
            "e1b9c9c674b1187918593257906ee6e8cc6a13da",
            "--model",
            "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
            "--timeout",
            timeout,
        )


def test_benchmark_cli_records_replays_and_rejects_tampered_audio(
    tmp_path: Path,
) -> None:
    wav_path = tmp_path / "input.wav"
    run_dir = tmp_path / "run"
    write_fixture_wav(wav_path)

    async def handler(websocket: ServerConnection) -> None:
        await DuplexPeer().handler(websocket)

    code, stdout, stderr = asyncio.run(serve_cli(handler, run_dir, wav_path, "5"))
    assert code == 0, (stdout, stderr)
    report = json.loads((run_dir / "report.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert json.loads(stdout) == report["summary"]
    assert report["summary"]["selected"] == 1
    assert report["summary"]["passed"] == 1
    assert report["summary"]["failed"] == 0
    assert report["summary"]["not_exercised"] == 0
    assert report["summary"]["qualified_metrics"]
    assert report["summary"]["diagnostic_metrics"] == {}
    assert (run_dir / manifest["input"]["file"]).read_bytes() == FIXTURE_PCM
    assert manifest["input"]["sha256"] == hashlib.sha256(FIXTURE_PCM).hexdigest()
    assert manifest["config"]["transport"]["keepalive_ping"] is False
    assert manifest["server"]["model"] == "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
    assert manifest["server"]["model_revision"] is None
    assert "error" in manifest["server"]["served_models"]
    assert manifest["source"]["packages"]["websockets"] == websockets.__version__
    assert manifest["config"]["input_duration_s"] == pytest.approx(
        FIXTURE_PACKETS * PACKET_MS / 1000
    )
    assert len(manifest["cases"]) == 1
    assert "response_cancel" in manifest["config"]["unsupported"]
    assert "server_resource_release" in manifest["config"]["unmeasured"]
    for case in manifest["cases"]:
        assert (run_dir / case["trace_file"]).stat().st_size > 0

    code, stdout, stderr = asyncio.run(
        run_cli("benchmarks.duplex.artifacts", str(run_dir))
    )
    assert code == 0, (stdout, stderr)
    replay = json.loads(stdout)
    assert replay["cases"] == report["cases"]
    assert replay["summary"] == report["summary"]
    assert replay["recorded_source"] == report["recorded_source"]
    assert replay["server"] == report["server"]

    corrupted_run = tmp_path / "corrupted-run"
    shutil.copytree(run_dir, corrupted_run)
    (corrupted_run / manifest["input"]["file"]).write_bytes(b"\x03" + FIXTURE_PCM[1:])
    code, stdout, stderr = asyncio.run(
        run_cli("benchmarks.duplex.artifacts", str(corrupted_run))
    )
    assert code == 1, (stdout, stderr)
    corrupted_report = json.loads(stdout)
    assert corrupted_report["summary"]["selected"] == 1
    assert corrupted_report["summary"]["passed"] == 0
    assert corrupted_report["summary"]["failed"] == 1
    assert corrupted_report["summary"]["qualified_metrics"] == {}
    assert all(
        "input PCM SHA256 mismatch" in case["artifact_errors"]
        for case in corrupted_report["cases"]
    )


def test_benchmark_cli_retains_failed_attempt(tmp_path: Path) -> None:
    wav_path = tmp_path / "input.wav"
    run_dir = tmp_path / "run"
    write_fixture_wav(wav_path)

    async def handler(websocket: ServerConnection) -> None:
        await DuplexPeer("disconnect").handler(websocket)

    code, stdout, stderr = asyncio.run(serve_cli(handler, run_dir, wav_path, "5"))
    assert code == 1, (stdout, stderr)
    report = json.loads((run_dir / "report.json").read_text())
    assert json.loads(stdout) == report["summary"]
    assert report["summary"]["selected"] == 1
    assert report["summary"]["passed"] == 0
    assert report["summary"]["failed"] == 1
    assert report["cases"][0]["id"] == "continuous"
    assert report["cases"][0]["status"] == "fail"
    assert report["summary"]["qualified_metrics"] == {}
    assert report["summary"]["diagnostic_metrics"]
    assert (run_dir / "continuous.jsonl").stat().st_size > 0


def test_benchmark_cli_rejects_a_deadline_the_input_cannot_meet(
    tmp_path: Path,
) -> None:
    wav_path = tmp_path / "input.wav"
    run_dir = tmp_path / "run"
    write_fixture_wav(wav_path)

    code, stdout, stderr = asyncio.run(
        run_cli(
            "benchmarks.eval.benchmark_duplex",
            "--url",
            "ws://127.0.0.1:1/v1/realtime",
            "--audio",
            str(wav_path),
            "--output",
            str(run_dir),
            "--server-revision",
            "e1b9c9c674b1187918593257906ee6e8cc6a13da",
            "--model",
            "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
            "--timeout",
            "0.5",
        )
    )
    assert code == 2, (stdout, stderr)
    assert "--timeout must exceed the 0.64 second paced input duration" in stderr
    assert not run_dir.exists()
