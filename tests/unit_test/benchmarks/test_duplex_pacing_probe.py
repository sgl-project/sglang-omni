# SPDX-License-Identifier: Apache-2.0
"""Pacing probe: deviation per session, pooled run processes, and the mock protocol."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import websockets

from benchmarks.duplex.client import PACKET_MS, SEND_RECEIPTS_FILE
from benchmarks.duplex.pacing_probe import (
    SUMMARY_FILE,
    TRACE_FILE,
    LoopLag,
    MockServer,
    ProbeSamples,
    run_probe,
    summarize_session,
    write_summary,
)

SYNTHETIC_PACKETS = 4


def write_session(session_dir: Path, start_s: float, delays_ms: list[float]) -> None:
    """Receipts of a closed session whose packet N started delays_ms[N] after its slot."""
    session_dir.mkdir(parents=True)
    appends = []
    for sequence, delay_ms in enumerate(delays_ms):
        send_start_s = start_s + sequence * PACKET_MS / 1000 + delay_ms / 1000
        appends.append(
            {
                "event_id": f"append-{sequence}",
                "seq": sequence,
                "start_s": send_start_s,
                "completed_s": send_start_s + 0.001,
            }
        )
    (session_dir / SEND_RECEIPTS_FILE).write_text(json.dumps({"appends": appends}))
    closed = {
        "direction": "receive",
        "time_s": start_s + 1,
        "event": {"type": "session.closed"},
    }
    (session_dir / TRACE_FILE).write_text(json.dumps(closed) + "\n")


def probe_samples(session_dir: Path) -> ProbeSamples:
    return ProbeSamples(
        concurrency=1,
        duration_s=SYNTHETIC_PACKETS * PACKET_MS / 1000,
        expected_packets_per_session=SYNTHETIC_PACKETS,
        inject_stall_ms=0.0,
        sessions=[summarize_session(session_dir, SYNTHETIC_PACKETS)],
        websocket_send_spans_s=[0.001],
        loop_lags=[LoopLag(wake_s=0.0, lag_s=0.0)],
        wall_s=1.0,
        cpu_s=0.1,
        peak_rss_megabytes=1.0,
    )


def test_merged_processes_judge_each_session_from_its_own_first_packet(
    tmp_path: Path,
) -> None:
    """A later session must not look late; 79 ms is in tolerance and 81 ms is not."""
    write_session(tmp_path / "early", start_s=100.0, delays_ms=[0, 0, 79, 0])
    write_session(tmp_path / "late", start_s=100.5, delays_ms=[0, 81, 0, 0])

    write_summary(
        [probe_samples(tmp_path / "early"), probe_samples(tmp_path / "late")],
        tmp_path,
    )

    summary = json.loads((tmp_path / SUMMARY_FILE).read_text())
    assert summary["processes"] == 2
    assert summary["packets"] == 2 * SYNTHETIC_PACKETS
    assert summary["packets_over_tolerance"] == 1
    assert summary["sessions_with_pacing_violations"] == 1
    assert summary["sessions_incomplete"] == 0
    assert summary["peak_overlapping_input_streams"] == 1


def test_mock_serves_every_packet_of_concurrent_client_sessions(
    tmp_path: Path,
) -> None:
    """The mock must keep up with the protocol the client speaks, or no level is valid."""
    concurrency, duration_s = 2, 0.8
    mock = MockServer(reply_audio_bytes=64, replies_per_append=3)

    async def probe() -> None:
        async with websockets.serve(
            mock.handle, "127.0.0.1", 0, compression=None, ping_interval=None
        ) as server:
            await run_probe(
                f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                concurrency,
                duration_s,
                tmp_path / "run",
                inject_stall_ms=0.0,
            )

    asyncio.run(probe())

    summary = json.loads((tmp_path / "run" / SUMMARY_FILE).read_text())
    packets = concurrency * summary["expected_packets_per_session"]
    assert summary["sessions_incomplete"] == 0
    assert summary["packets"] == packets
    assert mock.connections == concurrency
    assert len(mock.arrival_deviations_s) == packets
    assert summary["peak_overlapping_input_streams"] == concurrency
