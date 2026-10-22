# SPDX-License-Identifier: Apache-2.0
"""Duplex client sessions against a fake native peer, and the recording CLI."""

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

import pytest
import websockets
from pydantic import JsonValue
from websockets.asyncio.server import ServerConnection
from websockets.http11 import Request, Response

from benchmarks.duplex.client import (
    ADMISSION_RETRIES,
    PACKET_MS,
    POST_CLOSE_SECONDS,
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


def test_client_resends_a_rejected_session_update_on_the_same_socket(
    tmp_path: Path,
) -> None:
    """An in-session admission_rejected used to abort the recording, though the session stays open for another session.update."""
    peer = DuplexPeer(rejected_updates=ADMISSION_RETRIES)
    records = asyncio.run(
        capture_session(peer, tmp_path / "trace.jsonl", timeout_s=10.0)
    )
    updates = [
        r
        for r in records
        if r["direction"] == "send" and r["event"]["type"] == "session.update"
    ]

    assert len(updates) == ADMISSION_RETRIES + 1
    assert not [r for r in records if r["direction"] == "error"]
    assert bytes(peer.pcm) == FIXTURE_PCM
    result = evaluate_trace(records)
    assert result["status"] == "pass", result
    assert result["metrics"]["admission_rejections"] == ADMISSION_RETRIES


def test_client_closes_once_the_rejection_budget_is_exhausted(tmp_path: Path) -> None:
    peer = DuplexPeer(rejected_updates=ADMISSION_RETRIES + 1)
    records = asyncio.run(
        capture_session(peer, tmp_path / "trace.jsonl", timeout_s=10.0)
    )
    sends = [r["event"]["type"] for r in records if r["direction"] == "send"]

    assert sends == ["session.update"] * (ADMISSION_RETRIES + 1) + ["session.close"]
    assert records[-2]["event"]["type"] == "session.closed"
    assert records[-1]["direction"] == "error"
    assert (
        records[-1]["event"]["message"]
        == f"RuntimeError: admission rejected {ADMISSION_RETRIES + 1} times"
    )
    assert not peer.pcm
    result = evaluate_trace(records)
    assert result["status"] == "fail", result
    assert result["metrics"]["admission_rejections"] == ADMISSION_RETRIES + 1


def test_client_does_not_resend_after_another_error(tmp_path: Path) -> None:
    peer = DuplexPeer(rejected_updates=1, rejection_code="fixture_failure")
    records = asyncio.run(
        capture_session(peer, tmp_path / "trace.jsonl", timeout_s=10.0)
    )
    sends = [r["event"]["type"] for r in records if r["direction"] == "send"]

    assert sends == ["session.update", "session.close"]
    result = evaluate_trace(records)
    assert result["status"] == "fail", result
    assert any("code=fixture_failure" in item for item in result["violations"])


def test_client_stops_retrying_when_the_session_fails_during_the_backoff(
    tmp_path: Path,
) -> None:
    """A fatal error during the backoff used to leave the recorder asleep after the server closed, so the trace recorded a stalled driver."""
    peer = DuplexPeer("fatal_after_rejection", rejected_updates=1)
    records = asyncio.run(
        capture_session(peer, tmp_path / "trace.jsonl", timeout_s=10.0)
    )
    sends = [r["event"]["type"] for r in records if r["direction"] == "send"]

    assert sends == ["session.update"]
    assert not [r for r in records if r["direction"] == "error"]
    assert records[-1]["event"]["type"] == "session.closed"
    assert evaluate_trace(records)["status"] == "fail"


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
