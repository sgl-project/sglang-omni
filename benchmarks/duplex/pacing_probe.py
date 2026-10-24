# SPDX-License-Identifier: Apache-2.0
"""Check that the duplex load generator holds its 80 ms input pacing at a concurrency.

Subcommands: mock serves the native session protocol without a model, run streams
concurrent run_session sessions of silence against it, merge pools run processes.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import platform
import resource
import shutil
import signal
import time
import uuid
from pathlib import Path

import numpy as np
import websockets
from pydantic import BaseModel, ConfigDict, JsonValue

from benchmarks.duplex.client import (
    MAX_TIMEOUT_S,
    PACKET_BYTES,
    PACKET_MS,
    SAMPLE_RATE,
    SEND_RECEIPTS_FILE,
    run_session,
)
from benchmarks.duplex.oracle_models import TraceRecord

PACING_TOLERANCE_S = PACKET_MS / 1000
LOOP_LAG_PROBE_S = 0.01
PERCENTILES = (50, 95, 99)
SAMPLES_FILE = "samples.json"
SUMMARY_FILE = "summary.json"
TRACE_FILE = "trace.jsonl"
APPEND_EVENT = "input_audio_buffer.append"
# One 80 ms packet of 22.05 kHz PCM16, the reply size of the default profile.
DEFAULT_REPLY_AUDIO_BYTES = 3528
DEFAULT_REPLIES_PER_APPEND = 3
# Late packets past this are saved with their time, to line up processes of one host.
LATE_PACKET_RECORD_S = 0.02
# ru_maxrss is in kilobytes on Linux.
KILOBYTES_PER_MEGABYTE = 1024
# Room for session setup and the close handshake after the last packet.
SESSION_TIMEOUT_MARGIN_S = 30.0


class SequenceMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    seq: int


class ClientEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str
    sglang: SequenceMetadata | None = None


class AppendReceipt(BaseModel):
    model_config = ConfigDict(extra="ignore")
    seq: int
    start_s: float
    completed_s: float


class SendReceipts(BaseModel):
    appends: list[AppendReceipt]


class LoopLag(BaseModel):
    wake_s: float
    lag_s: float


class LatePacket(BaseModel):
    time_s: float
    deviation_s: float


class SessionSamples(BaseModel):
    deviations_s: list[float]
    late_packets: list[LatePacket]
    append_spans_s: list[float]
    first_start_s: float
    last_completed_s: float
    is_complete: bool


class ProbeSamples(BaseModel):
    """Raw samples of one run process; perf_counter is shared by processes of one host."""

    concurrency: int
    duration_s: float
    expected_packets_per_session: int
    inject_stall_ms: float
    sessions: list[SessionSamples]
    websocket_send_spans_s: list[float]
    loop_lags: list[LoopLag]
    wall_s: float
    cpu_s: float
    peak_rss_megabytes: float


def percentiles_ms(values_s: list[float], prefix: str) -> dict[str, JsonValue]:
    assert values_s, f"no samples for {prefix}"
    return {
        **{
            f"{prefix}_p{percent}_ms": float(np.percentile(values_s, percent)) * 1000
            for percent in PERCENTILES
        },
        f"{prefix}_max_ms": max(values_s) * 1000,
    }


async def measure_loop_lag(loop_lags: list[LoopLag]) -> None:
    """Record how late each fixed-interval sleep wakes up."""
    while True:
        sleep_started_s = time.perf_counter()
        await asyncio.sleep(LOOP_LAG_PROBE_S)
        wake_s = time.perf_counter()
        loop_lags.append(
            LoopLag(wake_s=wake_s, lag_s=wake_s - sleep_started_s - LOOP_LAG_PROBE_S)
        )


class MockServer:
    """Native session protocol without a model; times when input packets arrive."""

    def __init__(self, reply_audio_bytes: int, replies_per_append: int) -> None:
        self.reply_fields: dict[str, dict[str, str]] = {
            "response.output_audio.delta": {
                "delta": base64.b64encode(bytes(reply_audio_bytes)).decode("ascii")
            },
            "response.output_text.delta": {"delta": "mock reply text"},
            "sglang.unit.done": {},
        }
        # One audio delta first and one unit.done last, text deltas in between.
        self.reply_types: list[str] = (
            ["response.output_audio.delta"]
            + ["response.output_text.delta"] * (replies_per_append - 2)
            + ["sglang.unit.done"]
        )[:replies_per_append]
        self.connections: int = 0
        self.arrival_deviations_s: list[float] = []
        self.late_arrivals: list[LatePacket] = []
        self.loop_lags: list[LoopLag] = []
        self.first_arrival_s: float | None = None
        self.first_arrival_cpu_s: float = 0.0
        self.last_arrival_s: float = 0.0
        self.last_arrival_cpu_s: float = 0.0

    async def reply(self, websocket: websockets.ServerConnection) -> None:
        for reply_type in self.reply_types:
            await websocket.send(
                json.dumps(
                    {
                        "type": reply_type,
                        "event_id": uuid.uuid4().hex,
                        **self.reply_fields[reply_type],
                    }
                )
            )

    async def handle(self, websocket: websockets.ServerConnection) -> None:
        self.connections += 1
        first_append_s: float | None = None
        await websocket.send(json.dumps({"type": "session.created"}))
        async for frame in websocket:
            arrival_s = time.perf_counter()
            event = ClientEvent.model_validate_json(frame)
            if event.type == "session.update":
                await websocket.send(json.dumps({"type": "session.updated"}))
            elif event.type == APPEND_EVENT:
                assert event.sglang is not None
                if first_append_s is None:
                    first_append_s = arrival_s
                else:
                    pass
                if self.first_arrival_s is None:
                    self.first_arrival_s = arrival_s
                    self.first_arrival_cpu_s = time.process_time()
                else:
                    pass
                self.last_arrival_s = arrival_s
                self.last_arrival_cpu_s = time.process_time()
                deviation_s = (
                    arrival_s - first_append_s - event.sglang.seq * PACKET_MS / 1000
                )
                self.arrival_deviations_s.append(deviation_s)
                if abs(deviation_s) > LATE_PACKET_RECORD_S:
                    self.late_arrivals.append(
                        LatePacket(time_s=arrival_s, deviation_s=deviation_s)
                    )
                else:
                    pass
                await self.reply(websocket)
            elif event.type == "sglang.input_audio.end":
                await websocket.send(json.dumps({"type": "sglang.input_audio.drained"}))
            elif event.type == "session.close":
                await websocket.send(json.dumps({"type": "session.closed"}))
                await websocket.close()
            else:
                pass

    def summarize(self) -> dict[str, JsonValue]:
        assert self.first_arrival_s is not None, "no input packet arrived"
        window_s = self.last_arrival_s - self.first_arrival_s
        return {
            "connections": self.connections,
            "packets": len(self.arrival_deviations_s),
            "arrival_window_s": window_s,
            "mock_cpu_cores": (self.last_arrival_cpu_s - self.first_arrival_cpu_s)
            / window_s,
            **percentiles_ms(
                [abs(deviation_s) for deviation_s in self.arrival_deviations_s],
                "arrival_pacing_abs",
            ),
            **percentiles_ms(
                [
                    loop_lag.lag_s
                    for loop_lag in self.loop_lags
                    if self.first_arrival_s <= loop_lag.wake_s <= self.last_arrival_s
                ],
                "mock_loop_lag",
            ),
            "late_arrivals": [
                late_arrival.model_dump() for late_arrival in self.late_arrivals
            ],
        }


async def serve_mock(
    host: str,
    port: int,
    reply_audio_bytes: int,
    replies_per_append: int,
    stats_output: Path,
) -> None:
    mock = MockServer(reply_audio_bytes, replies_per_append)
    stop = asyncio.Event()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(signal_number, stop.set)
    lag_task = asyncio.create_task(measure_loop_lag(mock.loop_lags))
    async with websockets.serve(
        mock.handle,
        host,
        port,
        compression=None,
        ping_interval=None,
        max_size=None,
    ):
        print(f"Mock server listening on ws://{host}:{port}", flush=True)
        await stop.wait()
    lag_task.cancel()
    stats_output.write_text(json.dumps(mock.summarize(), indent=2) + "\n")


def summarize_session(session_dir: Path, expected_packets: int) -> SessionSamples:
    """Per-append pacing deviation and spans of one recorded session."""
    appends = SendReceipts.model_validate_json(
        (session_dir / SEND_RECEIPTS_FILE).read_text()
    ).appends
    records = [
        TraceRecord.model_validate_json(line)
        for line in (session_dir / TRACE_FILE).read_text().splitlines()
    ]
    has_error = any(record.direction == "error" for record in records)
    has_closed = any(
        record.direction == "receive" and record.event["type"] == "session.closed"
        for record in records
    )
    origin_s = appends[0].start_s if appends else 0.0
    deviations_s = [
        append.start_s - origin_s - append.seq * PACKET_MS / 1000 for append in appends
    ]
    return SessionSamples(
        deviations_s=deviations_s,
        late_packets=[
            LatePacket(time_s=append.start_s, deviation_s=deviation_s)
            for append, deviation_s in zip(appends, deviations_s)
            if abs(deviation_s) > LATE_PACKET_RECORD_S
        ],
        append_spans_s=[append.completed_s - append.start_s for append in appends],
        first_start_s=origin_s,
        last_completed_s=appends[-1].completed_s if appends else 0.0,
        is_complete=len(appends) == expected_packets and has_closed and not has_error,
    )


async def stall_event_loop(stall_ms: float, delay_s: float) -> None:
    """Fault injection: block the whole loop once, as a synchronous call would."""
    await asyncio.sleep(delay_s)
    time.sleep(stall_ms / 1000)


async def run_probe(
    url: str,
    concurrency: int,
    duration_s: float,
    output: Path,
    inject_stall_ms: float,
) -> None:
    timeout_s = duration_s + SESSION_TIMEOUT_MARGIN_S
    if timeout_s > MAX_TIMEOUT_S:
        raise ValueError(
            f"duration {duration_s}s needs a session timeout above {MAX_TIMEOUT_S}s"
        )
    else:
        pass
    pcm = bytes(round(duration_s * SAMPLE_RATE) * 2 // PACKET_BYTES * PACKET_BYTES)
    expected_packets = len(pcm) // PACKET_BYTES
    session_dirs = [output / f"session-{index:03d}" for index in range(concurrency)]
    for session_dir in session_dirs:
        session_dir.mkdir(parents=True)

    # note (kangwenqiao): Receipt spans also hold the trace record of the event, so
    # back-pressure is read from the socket send alone.
    websocket_send_spans_s: list[float] = []
    original_send = websockets.ClientConnection.send

    async def timed_send(self: websockets.ClientConnection, message: str) -> None:
        send_started_s = time.perf_counter()
        await original_send(self, message)
        websocket_send_spans_s.append(time.perf_counter() - send_started_s)

    loop_lags: list[LoopLag] = []
    lag_task = asyncio.create_task(measure_loop_lag(loop_lags))
    stall_task = asyncio.create_task(
        stall_event_loop(inject_stall_ms, duration_s / 2)
        if inject_stall_ms > 0
        else asyncio.sleep(0)
    )
    websockets.ClientConnection.send = timed_send
    wall_started_s = time.perf_counter()
    cpu_started_s = time.process_time()
    try:
        await asyncio.gather(
            *(
                run_session(
                    url, pcm, trace_path=session_dir / TRACE_FILE, timeout_s=timeout_s
                )
                for session_dir in session_dirs
            )
        )
    finally:
        websockets.ClientConnection.send = original_send
    wall_s = time.perf_counter() - wall_started_s
    cpu_s = time.process_time() - cpu_started_s
    lag_task.cancel()
    await stall_task

    sessions = [
        summarize_session(session_dir, expected_packets) for session_dir in session_dirs
    ]
    streamed = [session for session in sessions if session.deviations_s]
    window_start_s = min(session.first_start_s for session in streamed)
    window_end_s = max(session.last_completed_s for session in streamed)
    samples = ProbeSamples(
        concurrency=concurrency,
        duration_s=duration_s,
        expected_packets_per_session=expected_packets,
        inject_stall_ms=inject_stall_ms,
        sessions=sessions,
        websocket_send_spans_s=websocket_send_spans_s,
        loop_lags=[
            loop_lag
            for loop_lag in loop_lags
            if window_start_s <= loop_lag.wake_s <= window_end_s
        ],
        wall_s=wall_s,
        cpu_s=cpu_s,
        peak_rss_megabytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / KILOBYTES_PER_MEGABYTE,
    )
    (output / SAMPLES_FILE).write_text(samples.model_dump_json())
    # A 90 s session leaves about 10 MB of trace, mostly base64 audio.
    for session_dir in session_dirs:
        shutil.rmtree(session_dir)
    write_summary([samples], output)


def write_summary(processes: list[ProbeSamples], output: Path) -> None:
    sessions = [session for process in processes for session in process.sessions]
    streamed = [session for session in sessions if session.deviations_s]
    deviations_s = [
        deviation_s for session in streamed for deviation_s in session.deviations_s
    ]
    stream_edges = sorted(
        [(session.first_start_s, 1) for session in streamed]
        + [(session.last_completed_s, -1) for session in streamed]
    )
    open_streams, peak_streams = 0, 0
    for _, stream_change in stream_edges:
        open_streams += stream_change
        peak_streams = max(peak_streams, open_streams)
    summary: dict[str, JsonValue] = {
        "processes": len(processes),
        "concurrency": sum(process.concurrency for process in processes),
        "duration_s": processes[0].duration_s,
        "expected_packets_per_session": processes[0].expected_packets_per_session,
        "inject_stall_ms": processes[0].inject_stall_ms,
        "sessions_incomplete": sum(not session.is_complete for session in sessions),
        "peak_overlapping_input_streams": peak_streams,
        "all_sessions_overlap_s": max(
            0.0,
            min(session.last_completed_s for session in streamed)
            - max(session.first_start_s for session in streamed),
        ),
        "packets": len(deviations_s),
        "packets_over_tolerance": sum(
            abs(deviation_s) > PACING_TOLERANCE_S for deviation_s in deviations_s
        ),
        "sessions_with_pacing_violations": sum(
            any(
                abs(deviation_s) > PACING_TOLERANCE_S
                for deviation_s in session.deviations_s
            )
            for session in streamed
        ),
        **percentiles_ms(
            [abs(deviation_s) for deviation_s in deviations_s], "pacing_abs"
        ),
        **percentiles_ms(
            [span_s for session in streamed for span_s in session.append_spans_s],
            "append_span",
        ),
        **percentiles_ms(
            [
                span_s
                for process in processes
                for span_s in process.websocket_send_spans_s
            ],
            "websocket_send",
        ),
        **percentiles_ms(
            [loop_lag.lag_s for process in processes for loop_lag in process.loop_lags],
            "loop_lag",
        ),
        "wall_s": max(process.wall_s for process in processes),
        "process_cpu_cores_max": max(
            process.cpu_s / process.wall_s for process in processes
        ),
        "process_cpu_cores_total": sum(
            process.cpu_s / process.wall_s for process in processes
        ),
        "peak_rss_megabytes_max": max(
            process.peak_rss_megabytes for process in processes
        ),
        "python": platform.python_version(),
        "websockets": websockets.__version__,
    }
    (output / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    mock_parser = subparsers.add_parser("mock")
    mock_parser.add_argument("--host", default="127.0.0.1")
    mock_parser.add_argument("--port", type=int, default=8765)
    mock_parser.add_argument(
        "--reply-audio-bytes", type=int, default=DEFAULT_REPLY_AUDIO_BYTES
    )
    mock_parser.add_argument(
        "--replies-per-append", type=int, default=DEFAULT_REPLIES_PER_APPEND
    )
    mock_parser.add_argument("--stats-output", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--url", default="ws://127.0.0.1:8765")
    run_parser.add_argument("--concurrency", type=int, required=True)
    run_parser.add_argument("--duration", type=float, default=10.0)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--inject-stall-ms", type=float, default=0.0)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("run_dirs", type=Path, nargs="+")
    merge_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "mock":
        asyncio.run(
            serve_mock(
                arguments.host,
                arguments.port,
                arguments.reply_audio_bytes,
                arguments.replies_per_append,
                arguments.stats_output,
            )
        )
    elif arguments.command == "run":
        asyncio.run(
            run_probe(
                arguments.url,
                arguments.concurrency,
                arguments.duration,
                arguments.output,
                arguments.inject_stall_ms,
            )
        )
    else:
        arguments.output.mkdir(parents=True)
        write_summary(
            [
                ProbeSamples.model_validate_json((run_dir / SAMPLES_FILE).read_text())
                for run_dir in arguments.run_dirs
            ],
            arguments.output,
        )


if __name__ == "__main__":
    main()
