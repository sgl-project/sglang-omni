# SPDX-License-Identifier: Apache-2.0
"""CPU-only serving timing and coordinated session tests."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
import wave
from pathlib import Path

import pytest
import websockets
from pydantic import JsonValue
from websockets.asyncio.server import ServerConnection

from benchmarks.duplex.client import (
    PACKET_BYTES,
    PACKET_MS,
    SEND_RECEIPTS_FILE,
    run_session,
    scheduled_send_s,
)
from benchmarks.duplex.profiles import DEFAULT_PROFILE
from benchmarks.duplex.serving import run_concurrency
from benchmarks.duplex.serving_metrics import (
    aggregate_sessions,
    distribution,
    session_metrics,
)
from benchmarks.duplex.serving_unit_metrics import (
    aggregate_unit_sessions,
    unit_session_metrics,
)
from tests.unit_test.benchmarks.test_duplex_client import DuplexPeer

INPUT_DURATION_S = 0.32


def recorded_session(
    trace_path: Path,
    output_times_s: list[float],
    *,
    send_delay_s: float = 0.0,
    failed: bool = False,
    output_samples: list[int] | None = None,
) -> dict[str, JsonValue]:
    trace_path.parent.mkdir(parents=True)
    receipts = [
        {
            "event_id": f"append-{index}",
            "seq": index,
            "scheduled_s": 10 + index * 0.08,
            "start_s": 10 + index * 0.08 + send_delay_s,
            "completed_s": 10 + index * 0.08 + send_delay_s + 0.001,
        }
        for index in range(4)
    ]
    packet_samples = output_samples or [1764] * len(output_times_s)
    assert len(packet_samples) == len(output_times_s)
    records = [
        {
            "direction": "receive",
            "time_s": timestamp_s,
            "event": {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(b"\x00\x00" * samples).decode("ascii"),
            },
        }
        for timestamp_s, samples in zip(output_times_s, packet_samples)
    ]
    records.extend(
        [
            {
                "direction": "receive",
                "time_s": 10.34,
                "event": {"type": "response.done", "response": {"status": "completed"}},
            },
            {
                "direction": "receive",
                "time_s": 10.35,
                "event": {"type": "sglang.input_audio.drained"},
            },
            {
                "direction": "receive",
                "time_s": 10.36,
                "event": {"type": "session.closed"},
            },
        ]
    )
    if failed:
        records.append(
            {
                "direction": "error",
                "time_s": 10.37,
                "event": {"message": "server disconnected"},
            }
        )
    trace_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    trace_path.with_name(SEND_RECEIPTS_FILE).write_text(
        json.dumps({"session_start_s": 10.0, "appends": receipts})
    )
    return session_metrics(
        trace_path,
        session_id=trace_path.parent.name,
        input_duration_s=INPUT_DURATION_S,
        profile=DEFAULT_PROFILE,
        reserve_s=0.08,
    )


def test_synthetic_serving_metrics(tmp_path: Path) -> None:
    assert distribution([0, 1, 2, 3])["p75"] == pytest.approx(2.25)
    assert distribution([])["p75"] is None
    perfect = recorded_session(
        tmp_path / "perfect" / "trace.jsonl", [10, 10.08, 10.16, 10.24]
    )
    assert perfect["success"] is True
    assert perfect["ttfa_s"] == pytest.approx(0)
    assert perfect["output_gap_s"]["p99"] == pytest.approx(0.08)
    assert perfect["output_gap_s"]["p75"] == pytest.approx(0.08)
    assert perfect["output_gap_excess_s"]["max"] == pytest.approx(0)
    assert perfect["output_drift_s"]["max"] == pytest.approx(0)
    assert perfect["final_output_drift_s"] == pytest.approx(0)
    assert perfect["late_send_rate"] == 0
    assert perfect["output_coverage"] == pytest.approx(1)
    assert perfect["underrun_count"] == 0
    assert perfect["underrun_ratio"] == 0

    stalled = recorded_session(
        tmp_path / "stalled" / "trace.jsonl", [10, 10.08, 10.28, 10.36]
    )
    assert stalled["output_gap_s"]["max"] == pytest.approx(0.2)
    assert stalled["output_gap_excess_s"]["max"] == pytest.approx(0.12)
    assert stalled["output_drift_s"]["p75"] == pytest.approx(0.12)
    assert stalled["final_output_drift_s"] == pytest.approx(0.12)
    assert stalled["underrun_count"] == 1
    assert stalled["underrun_total_s"] == pytest.approx(0.04)
    assert stalled["underrun_ratio"] == pytest.approx(0.125)

    sparse = recorded_session(tmp_path / "sparse" / "trace.jsonl", [10, 10.08])
    assert sparse["output_coverage"] == pytest.approx(0.5)
    assert sparse["underrun_total_s"] == pytest.approx(0.16)
    assert sparse["underrun_ratio"] == pytest.approx(0.5)

    late = recorded_session(
        tmp_path / "late" / "trace.jsonl",
        [10, 10.08, 10.16, 10.24],
        send_delay_s=0.05,
    )
    assert late["send_lateness_s"]["p99"] == pytest.approx(0.05)
    assert late["late_send_count"] == 4
    assert late["late_send_rate"] == 1

    batched = recorded_session(
        tmp_path / "batched" / "trace.jsonl",
        [10, 10.1, 10.19],
        output_samples=[3528, 1764, 1764],
    )
    assert batched["output_gap_excess_s"]["max"] == pytest.approx(0.01)
    assert batched["final_output_drift_s"] == pytest.approx(-0.05)

    silent = recorded_session(tmp_path / "silent" / "trace.jsonl", [])
    assert silent["success"] is False
    assert silent["ttfa_s"] is None
    assert silent["output_drift_s"]["n"] == 0
    assert silent["final_output_drift_s"] is None
    assert silent["output_coverage"] == 0
    assert silent["underrun_total_s"] == INPUT_DURATION_S
    assert silent["underrun_ratio"] == 1

    failed = recorded_session(
        tmp_path / "failed" / "trace.jsonl", [10, 10.08], failed=True
    )
    aggregate = aggregate_sessions([perfect, failed, silent])
    assert aggregate["attempted_sessions"] == 3
    assert aggregate["successful_sessions"] == 1
    assert aggregate["output_coverage"]["p50"] == pytest.approx(0.5)
    assert aggregate["ttfa_s"]["n"] == 2
    assert aggregate["final_output_drift_s"]["n"] == 2
    assert aggregate["late_send_rate"] == 0
    assert aggregate["underrun_ratio"] == pytest.approx(0.5)


def test_scheduled_deadlines_do_not_follow_late_sends() -> None:
    assert [scheduled_send_s(10, index) for index in range(4)] == pytest.approx(
        [10, 10.08, 10.16, 10.24]
    )
    assert scheduled_send_s(10, 3) == pytest.approx(10.24)


def test_late_start_returns_to_the_schedule(tmp_path: Path) -> None:
    async def run() -> list[dict[str, JsonValue]]:
        peer = DuplexPeer()
        async with websockets.serve(peer.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            gate = asyncio.get_running_loop().create_future()
            gate.set_result(time.perf_counter() - 0.25)
            trace_path = tmp_path / "trace.jsonl"
            await run_session(
                f"ws://127.0.0.1:{port}/v1/realtime",
                b"\x00\x00" * (8 * PACKET_BYTES // 2),
                scenario="continuous",
                trace_path=trace_path,
                start_gate=gate,
            )
            return json.loads(trace_path.with_name(SEND_RECEIPTS_FILE).read_text())[
                "appends"
            ]

    appends = asyncio.run(run())
    assert len(appends) == 8
    assert appends[0]["start_s"] - appends[0]["scheduled_s"] >= 0.25
    assert [r["scheduled_s"] - appends[0]["scheduled_s"] for r in appends] == (
        pytest.approx([index * PACKET_MS / 1000 for index in range(8)])
    )
    # note (Junnan Li): Packets due before the late start go out at once; later ones are on time.
    assert appends[-1]["start_s"] - appends[-1]["scheduled_s"] < 0.05


def test_coordinator_keeps_failed_session(tmp_path: Path) -> None:
    async def run() -> dict[str, JsonValue]:
        peers: list[DuplexPeer] = []

        async def handler(websocket: ServerConnection) -> None:
            peer = DuplexPeer("close_without_update" if len(peers) == 0 else "healthy")
            peers.append(peer)
            await peer.handler(websocket)

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            result = await run_concurrency(
                f"ws://127.0.0.1:{port}/v1/realtime",
                [b"\x00\x00" * (4 * PACKET_BYTES // 2)],
                concurrency=3,
                profile=DEFAULT_PROFILE,
                output_dir=tmp_path / "c3",
                timeout_s=5,
                reserve_s=0.08,
            )
        assert len(peers) == 3
        return result

    summary = asyncio.run(run())
    assert summary["configured_sessions"] == 2
    assert summary["aggregate"]["attempted_sessions"] == 3
    assert summary["aggregate"]["successful_sessions"] == 2
    assert summary["aggregate"]["underrun_ratio"] >= 1 / 3
    assert [session["success"] for session in summary["sessions"]] == [
        False,
        True,
        True,
    ]
    for session in summary["sessions"]:
        assert Path(session["trace_file"]).exists()
        receipts = json.loads(Path(session["receipts_file"]).read_text())
        if session["success"]:
            assert receipts["session_start_s"] == summary["common_start_s"]
        else:
            assert receipts["session_start_s"] is None


def test_serving_cli_sweep_against_fake_server(tmp_path: Path) -> None:
    audio_path = tmp_path / "input.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(b"\x00\x00" * (4 * PACKET_BYTES // 2))

    async def run() -> tuple[int, str]:
        async def handler(websocket: ServerConnection) -> None:
            await DuplexPeer().handler(websocket)

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "benchmarks.duplex.serving",
                "--url",
                f"ws://127.0.0.1:{port}/v1/realtime",
                "--audio",
                str(audio_path),
                "--profile",
                "nemotron",
                "--concurrencies",
                "1,2",
                "--output-dir",
                str(tmp_path / "results"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            assert process.returncode == 0, stderr.decode()
            return process.returncode, stdout.decode()

    _, output = asyncio.run(run())
    assert "1/1" in output
    assert "2/2" in output
    assert "late sends" in output
    assert "underrun sessions" in output
    assert "underrun ratio" in output
    summaries = json.loads((tmp_path / "results" / "summary.json").read_text())["runs"]
    assert [run["aggregate"]["attempted_sessions"] for run in summaries] == [1, 2]
    assert [run["aggregate"]["successful_sessions"] for run in summaries] == [1, 2]


def unit_trace(
    trace_path: Path, events: list[tuple[float, dict[str, JsonValue]]]
) -> dict[str, JsonValue]:
    """Record a 3 s MiniCPM-o session whose packets are sent on schedule."""
    trace_path.parent.mkdir(parents=True)
    packets = 38
    receipts = [
        {
            "event_id": f"append-{index}",
            "seq": index,
            "scheduled_s": 10 + index * 0.08,
            "start_s": 10 + index * 0.08,
            "completed_s": 10 + index * 0.08,
        }
        for index in range(packets)
    ]
    trace_path.with_name(SEND_RECEIPTS_FILE).write_text(
        json.dumps({"session_start_s": 10.0, "appends": receipts})
    )
    trace_path.write_text(
        "".join(
            json.dumps({"direction": "receive", "time_s": time_s, "event": event})
            + "\n"
            for time_s, event in events
        )
    )
    return unit_session_metrics(
        trace_path,
        session_id="session-000",
        input_duration_s=3.0,
        profile="minicpmo-native-pr2377",
        reserve_s=0.0,
    )


def unit_audio(seconds: float) -> dict[str, JsonValue]:
    return {
        "type": "response.output_audio.delta",
        "response_id": "reply",
        "delta": base64.b64encode(b"\x00\x00" * int(24000 * seconds)).decode(),
    }


def test_unit_metrics_separate_listening_from_late_speech(tmp_path: Path) -> None:
    # note (Junnan Li): Units become ready when packets 12, 24 and 37 are sent.
    ready = [10 + 12 * 0.08, 10 + 24 * 0.08, 10 + 37 * 0.08]
    done = {"type": "sglang.unit.done"}
    session = unit_trace(
        tmp_path / "session" / "trace.jsonl",
        [
            (ready[0] + 0.05, done),
            (ready[1] + 0.2, {"type": "response.created", "response": {"id": "reply"}}),
            (ready[1] + 0.4, unit_audio(1.0)),
            (ready[1] + 0.4, done),
            (ready[2] + 1.5, unit_audio(0.5)),
            (
                ready[2] + 1.5,
                {"type": "response.done", "response": {"status": "completed"}},
            ),
            (ready[2] + 1.5, done),
            (ready[2] + 1.5, {"type": "sglang.input_audio.drained"}),
            (ready[2] + 1.6, {"type": "session.closed"}),
        ],
    )
    assert session["success"]
    assert (session["completed_units"], session["speak_units"]) == (3, 2)
    assert session["listen_unit_lag_s"]["max"] == pytest.approx(0.05)
    assert session["speak_unit_lag_s"]["max"] == pytest.approx(1.5)
    assert (session["missed_units"], session["unit_miss_rate"]) == (1, 1 / 3)
    assert session["response_audio_lag_s"]["p50"] == pytest.approx(0.4)
    # note (Junnan Li): The second chunk arrives 1.14 s after a 1 s chunk began playing.
    assert session["underrun_count"] == 1
    assert session["underrun_total_s"] == pytest.approx(
        ready[2] + 1.5 - (ready[1] + 0.4) - 1.0
    )
    assert session["underrun_ratio"] == pytest.approx(session["underrun_total_s"] / 1.5)


def test_silent_unit_session_succeeds_and_incomplete_one_fails(tmp_path: Path) -> None:
    done = {"type": "sglang.unit.done"}
    closing = [
        (14.0, {"type": "sglang.input_audio.drained"}),
        (14.1, {"type": "session.closed"}),
    ]
    silent = unit_trace(
        tmp_path / "silent" / "trace.jsonl",
        [(11.0, done), (12.0, done), (13.0, done), *closing],
    )
    assert silent["success"] and silent["speak_units"] == 0
    assert silent["underrun_ratio"] is None and silent["missed_units"] == 0
    stalled = unit_trace(tmp_path / "stalled" / "trace.jsonl", [(11.0, done), *closing])
    assert not stalled["success"]
    assert stalled["errors"] == ["completed 1/3 units"]
    assert stalled["missed_units"] == 2
    aggregate = aggregate_unit_sessions([silent, stalled])
    assert aggregate["successful_sessions"] == 1
    assert aggregate["unit_miss_rate"] == pytest.approx(2 / 6)


def test_lockstep_sends_each_unit_when_the_previous_one_is_done(tmp_path: Path) -> None:
    async def run() -> tuple[float, list[dict[str, JsonValue]]]:
        peer = DuplexPeer()
        async with websockets.serve(peer.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            trace_path = tmp_path / "trace.jsonl"
            started_s = time.perf_counter()
            await run_session(
                f"ws://127.0.0.1:{port}/v1/realtime",
                b"\x00\x00" * (25 * PACKET_BYTES // 2),
                scenario="continuous",
                trace_path=trace_path,
                pacing="lockstep",
            )
            elapsed_s = time.perf_counter() - started_s
            return elapsed_s, [
                json.loads(line) for line in trace_path.read_text().splitlines()
            ]

    elapsed_s, records = asyncio.run(run())
    order = [
        record["event"]["type"]
        for record in records
        if record["event"].get("type")
        in ("input_audio_buffer.append", "sglang.unit.done")
    ]
    # note (Junnan Li): 25 packets are 2 s of input, and each waits for the unit before it.
    assert elapsed_s < 1.0
    assert order[:4] == [
        "input_audio_buffer.append",
        "sglang.unit.done",
        "input_audio_buffer.append",
        "sglang.unit.done",
    ]
    assert order.count("input_audio_buffer.append") == 25
