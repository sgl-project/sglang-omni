# SPDX-License-Identifier: Apache-2.0
"""Duplex client sessions against a fake native peer, and the recording CLI."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import itertools
import json
import shutil
import sys
import threading
import time
import wave
from http import HTTPStatus
from pathlib import Path, PosixPath
from typing import Literal

import pytest
import websockets
from pydantic import JsonValue
from websockets.asyncio.server import ServerConnection
from websockets.http11 import Request, Response

import benchmarks.duplex.client as client
from benchmarks.duplex.client import (
    ADMISSION_RETRIES,
    PACKET_MS,
    POST_CLOSE_SECONDS,
    SEND_RECEIPTS_FILE,
    run_session,
)
from benchmarks.duplex.oracle import evaluate_trace
from tests.unit_test.fixtures.duplex_fakes import (
    FATAL_CLOSE_DELAY_S,
    FIXTURE_PACKETS,
    FIXTURE_PCM,
    SAMPLES_PER_FRAME,
    UNIT_BYTES,
    DuplexPeer,
    PeerMode,
)


async def capture_session(
    peer: DuplexPeer, trace_path: Path, timeout_s: float = 2.0
) -> list[dict[str, JsonValue]]:
    existing_tasks = asyncio.all_tasks()
    async with websockets.serve(peer.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        await run_session(
            f"ws://127.0.0.1:{port}/v1/realtime",
            FIXTURE_PCM,
            trace_path=trace_path,
            timeout_s=timeout_s,
        )
        peer.release.set()
    assert asyncio.all_tasks() <= existing_tasks
    return [json.loads(line) for line in trace_path.read_text().splitlines()]


def input_append_records(
    records: list[dict[str, JsonValue]]
) -> list[dict[str, JsonValue]]:
    return [
        record
        for record in records
        if record["direction"] == "send"
        and record["event"]["type"] == "input_audio_buffer.append"
    ]


def count_output_samples(records: list[dict[str, JsonValue]]) -> int:
    return sum(
        len(base64.b64decode(record["event"]["delta"], validate=True)) // 2
        for record in records
        if record["direction"] == "receive"
        and record["event"]["type"] == "response.output_audio.delta"
    )


def test_client_records_complete_native_session(tmp_path: Path) -> None:
    peer = DuplexPeer()
    records = asyncio.run(capture_session(peer, tmp_path / "trace.jsonl"))
    result = evaluate_trace(records)

    assert result["status"] == "pass", result
    assert bytes(peer.pcm) == FIXTURE_PCM
    assert [r["event"] for r in records if r["direction"] == "send"] == peer.received
    assert [r["event"] for r in records if r["direction"] == "receive"] == peer.sent
    assert [r["time_s"] for r in records] == sorted(r["time_s"] for r in records)
    event_ids = [event["event_id"] for event in peer.received]
    assert len(set(event_ids)) == len(event_ids)
    assert peer.received[0]["session"]["output_modalities"] == ["audio"]
    appends = input_append_records(records)
    assert [r["event"]["sglang"]["seq"] for r in appends] == list(
        range(FIXTURE_PACKETS)
    )
    assert result["metrics"]["input_output_overlap"] is True

    # Note (wenyao): Absolute pacing permits catch-up packets, so measure total span.
    paced_s = (FIXTURE_PACKETS - 1) * PACKET_MS / 1000
    span_s = appends[-1]["time_s"] - appends[0]["time_s"]
    assert paced_s * 0.8 <= span_s <= paced_s + 1.0, span_s

    units = -(-len(FIXTURE_PCM) // UNIT_BYTES)
    assert count_output_samples(records) == units * SAMPLES_PER_FRAME
    assert result["metrics"]["output_audio_s"] == pytest.approx(
        units * SAMPLES_PER_FRAME / 22050
    )

    assert "response.cancel" not in [event["type"] for event in peer.received]


NON_ASCII_TRANSCRIPT = "fixturé 你好"


class FrameStylePeer(DuplexPeer):
    async def send(self, websocket: ServerConnection, event: dict) -> None:
        event["event_id"] = f"server_{len(self.sent)}"
        event = json.loads(
            json.dumps(event).replace('"fixture"', json.dumps(NON_ASCII_TRANSCRIPT))
        )
        self.sent.append(event)
        style = len(self.sent) % 3
        if style == 0:
            frame = json.dumps(event, indent=1)
        elif style == 1:
            frame = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        else:
            frame = json.dumps(event)
        await websocket.send(frame)


def test_trace_lines_embed_frames_and_reencode_the_ones_they_cannot(
    tmp_path: Path,
) -> None:
    peer = FrameStylePeer()
    trace_path = tmp_path / "trace.jsonl"
    records = asyncio.run(capture_session(peer, trace_path))

    assert evaluate_trace(records)["status"] == "pass"
    assert [r["event"] for r in records if r["direction"] == "receive"] == peer.sent
    assert [r["event"] for r in records if r["direction"] == "send"] == peer.received
    trace = trace_path.read_bytes()
    assert trace.isascii()
    assert any(
        NON_ASCII_TRANSCRIPT in json.dumps(event, ensure_ascii=False)
        for event in peer.sent
    )
    lines = trace.decode("ascii").splitlines()
    assert len(lines) == len(peer.sent) + len(peer.received)
    for line, record in zip(lines, records):
        if record["direction"] == "send":
            assert line == json.dumps(record)
        else:
            pass


class StallingBase64:
    def __init__(self, clock: StalledClock) -> None:
        self.clock = clock

    def b64encode(self, data: bytes) -> bytes:
        if not self.clock.offset_s:
            self.clock.offset_s = FIRST_APPEND_STALL_S
        else:
            pass
        return base64.b64encode(data)


class StalledClock:
    sleep = staticmethod(time.sleep)

    def __init__(self) -> None:
        self.offset_s = 0.0

    def perf_counter(self) -> float:
        return time.perf_counter() + self.offset_s


FIRST_APPEND_STALL_S = 0.15


def test_a_late_first_append_shifts_the_whole_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay measures from the first append, so the deadlines must start there too."""
    clock = StalledClock()
    monkeypatch.setattr(client, "time", clock)
    monkeypatch.setattr(client, "base64", StallingBase64(clock))
    asyncio.run(capture_session(DuplexPeer(), tmp_path / "trace.jsonl"))

    assert clock.offset_s == FIRST_APPEND_STALL_S
    appends = json.loads((tmp_path / SEND_RECEIPTS_FILE).read_text())["appends"]
    assert len(appends) == FIXTURE_PACKETS
    deviations_s = [
        append["start_s"] - appends[0]["start_s"] - append["seq"] * PACKET_MS / 1000
        for append in appends
    ]
    assert max(abs(value) for value in deviations_s) < PACKET_MS / 1000 / 2


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    [
        ("malformed_json", "{not valid json"),
        ("nonfinite", "backlog_ms"),
        ("disconnect", "ConnectionClosedError"),
    ],
)
def test_client_retains_failed_attempt(
    tmp_path: Path, mode: PeerMode, expected_error: str
) -> None:
    peer = DuplexPeer(mode)
    records = asyncio.run(capture_session(peer, tmp_path / "trace.jsonl"))
    result = evaluate_trace(records)

    assert result["status"] == "fail", result
    assert result["violations"]
    assert any(r["event"]["type"] == "session.created" for r in records)
    assert input_append_records(records)
    assert any(
        r["direction"] == "error" and expected_error in r["event"]["message"]
        for r in records
    )


def test_client_keeps_the_close_receipt_after_a_fatal_server_error(
    tmp_path: Path,
) -> None:
    peer = DuplexPeer("server_error")
    records = asyncio.run(capture_session(peer, tmp_path / "trace.jsonl"))
    result = evaluate_trace(records)

    assert result["status"] == "fail", result
    assert any(
        r["direction"] == "receive"
        and r["event"].get("error", {}).get("code") == "fixture_failure"
        for r in records
    )
    assert records[-1]["direction"] == "receive"
    assert records[-1]["event"]["type"] == "session.closed"
    assert any("server error" in violation for violation in result["violations"])
    assert len(input_append_records(records)) < FIXTURE_PACKETS


def test_client_waits_out_a_late_close_after_a_fatal_error(tmp_path: Path) -> None:
    peer = DuplexPeer("delayed_fatal")
    started_s = time.perf_counter()
    records = asyncio.run(
        capture_session(peer, tmp_path / "trace.jsonl", timeout_s=10.0)
    )
    elapsed_s = time.perf_counter() - started_s
    result = evaluate_trace(records)

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
        capture_session(peer, tmp_path / "trace.jsonl", timeout_s=10.0)
    )
    elapsed_s = time.perf_counter() - started_s
    result = evaluate_trace(records)

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
    assert len(input_append_records(records)) < FIXTURE_PACKETS


async def capture_with_denials(
    peer: DuplexPeer, denials: int, trace_path: Path
) -> list[dict[str, JsonValue]]:
    remaining = itertools.count()

    def process_request(
        connection: ServerConnection, request: Request
    ) -> Response | None:
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
    result = evaluate_trace(records)
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
    result = evaluate_trace(records)
    assert result["status"] == "fail", result
    assert result["metrics"]["admission_denials"] == ADMISSION_RETRIES


def test_client_enforces_the_session_deadline(tmp_path: Path) -> None:
    peer = DuplexPeer("timeout")
    started_s = time.perf_counter()
    records = asyncio.run(
        capture_session(peer, tmp_path / "trace.jsonl", timeout_s=0.5)
    )
    elapsed_s = time.perf_counter() - started_s

    assert 0.5 <= elapsed_s < 5.0, elapsed_s
    assert evaluate_trace(records)["status"] == "fail"
    assert any(r["event"]["type"] == "session.created" for r in records)
    assert input_append_records(records)
    assert records[-1]["direction"] == "error"
    assert "Session timeout after 0.5s" in records[-1]["event"]["message"]


def test_client_bounds_observation_after_the_session_closes(tmp_path: Path) -> None:
    peer = DuplexPeer("linger")
    started_s = time.perf_counter()
    records = asyncio.run(
        capture_session(peer, tmp_path / "trace.jsonl", timeout_s=10.0)
    )
    elapsed_s = time.perf_counter() - started_s
    streamed_s = FIXTURE_PACKETS * PACKET_MS / 1000

    assert elapsed_s < streamed_s + POST_CLOSE_SECONDS + 2.0, elapsed_s
    assert evaluate_trace(records)["status"] == "pass"
    assert records[-1]["event"]["type"] == "session.closed"


def test_client_abandons_a_driver_the_receive_loop_can_no_longer_serve(
    tmp_path: Path,
) -> None:
    peer = DuplexPeer("close_without_update")
    started_s = time.perf_counter()
    records = asyncio.run(
        capture_session(peer, tmp_path / "trace.jsonl", timeout_s=10.0)
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
    assert evaluate_trace(records)["status"] == "fail"


# Bounds a write that a blocked event loop could never release.
BLOCKED_WRITE_FALLBACK_S = 10.0


class InputEndPeer(DuplexPeer):
    """Signals, across threads, that every input packet of its session has arrived."""

    def __init__(self) -> None:
        super().__init__()
        self.input_ended = threading.Event()

    async def send(
        self, websocket: ServerConnection, event: dict[str, JsonValue]
    ) -> None:
        if event["type"] == "sglang.input_audio.ended":
            self.input_ended.set()
        else:
            pass
        await super().send(websocket, event)


class BlockedWritePath(PosixPath):
    """Session path whose trace or receipts write waits until the other session streamed."""

    blocked_file: Literal["trace", "receipts"] = "trace"
    other_input_ended = threading.Event()
    streamed_while_blocked: list[bool] = []

    def wait_for_other_session(self) -> None:
        """Only the first write waits, so a stalled loop fails after one fallback."""
        if BlockedWritePath.streamed_while_blocked:
            pass
        else:
            BlockedWritePath.streamed_while_blocked.append(
                BlockedWritePath.other_input_ended.wait(BLOCKED_WRITE_FALLBACK_S)
            )

    def open(
        self,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> io.TextIOWrapper:
        opened_file = super().open(mode, buffering, encoding, errors, newline)
        if mode == "x" and BlockedWritePath.blocked_file == "trace":
            write = opened_file.write

            def blocked_write(text: str) -> int:
                self.wait_for_other_session()
                return write(text)

            opened_file.write = blocked_write
        else:
            pass
        return opened_file

    def write_text(
        self,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if (
            self.name == SEND_RECEIPTS_FILE
            and BlockedWritePath.blocked_file == "receipts"
        ):
            self.wait_for_other_session()
        else:
            pass
        return super().write_text(data, encoding, errors, newline)


@pytest.mark.parametrize("blocked_file", ["trace", "receipts"])
def test_a_blocked_disk_write_does_not_stall_the_other_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_file: Literal["trace", "receipts"],
) -> None:
    """A write stuck in the kernel stalled the event loop that paces every session."""
    streaming_peer = InputEndPeer()
    monkeypatch.setattr(BlockedWritePath, "blocked_file", blocked_file)
    monkeypatch.setattr(
        BlockedWritePath, "other_input_ended", streaming_peer.input_ended
    )
    monkeypatch.setattr(BlockedWritePath, "streamed_while_blocked", [])
    blocked_dir, streaming_dir = tmp_path / "blocked", tmp_path / "streaming"
    blocked_dir.mkdir()
    streaming_dir.mkdir()

    async def record_both() -> None:
        async with (
            websockets.serve(DuplexPeer().handler, "127.0.0.1", 0) as blocked_server,
            websockets.serve(
                streaming_peer.handler, "127.0.0.1", 0
            ) as streaming_server,
        ):
            await asyncio.gather(
                run_session(
                    f"ws://127.0.0.1:{blocked_server.sockets[0].getsockname()[1]}",
                    FIXTURE_PCM[: 2 * UNIT_BYTES],
                    trace_path=BlockedWritePath(blocked_dir / "trace.jsonl"),
                ),
                run_session(
                    f"ws://127.0.0.1:{streaming_server.sockets[0].getsockname()[1]}",
                    FIXTURE_PCM * 3,
                    trace_path=streaming_dir / "trace.jsonl",
                ),
            )

    asyncio.run(record_both())

    assert BlockedWritePath.streamed_while_blocked[0]
    for session_dir in (blocked_dir, streaming_dir):
        records = [
            json.loads(line)
            for line in (session_dir / "trace.jsonl").read_text().splitlines()
        ]
        assert evaluate_trace(records)["status"] == "pass"
        receipts = json.loads((session_dir / SEND_RECEIPTS_FILE).read_text())
        assert len(receipts["appends"]) == len(input_append_records(records))


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
