# SPDX-License-Identifier: Apache-2.0
"""Check that the duplex load generator holds its 80 ms input pacing at a concurrency.

Subcommands: mock serves the native session protocol without a model, run streams
concurrent run_session sessions of silence against it, merge pools run processes.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import json
import platform
import resource
import shutil
import signal
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile
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
from benchmarks.duplex.v15_runner import run_samples

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


# note (edwingao28): Bursts stand for a talker that runs ahead of real time and a pacer
# that flushes its staged audio, both of which a native full-duplex server has.
NATIVE_GRANTED: dict[str, JsonValue] = {
    "interaction": "native",
    "native_full_duplex": True,
    "proactive_output": False,
    "turn_control": [None],
    "client_commit": False,
    "input_modalities": ["audio"],
    "output_modalities": ["audio"],
    "input_audio_format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
    "output_audio_format": {"type": "audio/pcm", "rate": 22050},
    "native_unit_ms": PACKET_MS,
    "first_unit_ms": PACKET_MS,
    "microturn_ms": None,
    "tail_policy": "pad",
    "supports_server_interrupt": False,
    "supports_truncate": False,
    "supports_resume": False,
    "partial_style": "append_only",
    "pressure_policy": "reject",
    "strict_order": True,
    "limits": {
        "max_input_bytes": 16 * 1024 * 1024,
        "max_input_chunks": 4096,
        "max_output_bytes": 64 * 1024 * 1024,
        "max_output_events": 65536,
        "max_responses": 128,
        "max_segments": 4096,
        "max_history_chars": 65536,
        "session_timeout_s": 240,
        "cleanup_timeout_s": 30,
    },
    "rejections": [],
}
DEFAULT_OUTPUT_SAMPLE_RATE = 22050
DEFAULT_BURST_UNITS = 1


class NativeSession:
    def __init__(
        self, mock: NativeMock, websocket: websockets.ServerConnection
    ) -> None:
        self.mock = mock
        self.websocket = websocket
        self.sent = 0
        self.input_bytes = 0
        self.units_emitted = 0
        self.response_open = False
        self.first_append_s: float | None = None

    async def send(self, event: dict[str, JsonValue]) -> None:
        event["event_id"] = f"server_{self.sent}"
        self.sent += 1
        await self.websocket.send(json.dumps(event))

    async def emit_unit(self, unit: int, duration_ms: float) -> None:
        if not self.response_open:
            await self.send(
                {
                    "type": "response.created",
                    "response": {
                        "id": "response_0",
                        "object": "realtime.response",
                        "status": "in_progress",
                        "output": [],
                    },
                }
            )
            self.response_open = True
        else:
            pass
        media_time = {"t_start_ms": unit * PACKET_MS, "duration_ms": duration_ms}
        await self.send(
            {
                "type": "response.output_audio.delta",
                "response_id": "response_0",
                "item_id": "item_0",
                "output_index": 0,
                "content_index": 0,
                "delta": self.mock.unit_audio,
                "sglang": {
                    "unit_id": f"unit_{unit}",
                    "chunk_seq": unit,
                    "media_time": media_time,
                },
            }
        )
        if self.mock.transcript_deltas:
            await self.send(
                {
                    "type": "response.output_audio_transcript.delta",
                    "response_id": "response_0",
                    "item_id": "item_0",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": " mock",
                }
            )
        else:
            pass
        await self.send(
            {
                "type": "sglang.unit.done",
                "unit_id": f"unit_{unit}",
                "sglang": {"unit_id": f"unit_{unit}", "media_time": media_time},
            }
        )
        self.units_emitted = unit + 1

    async def handle(self) -> None:
        await self.send(
            {
                "type": "session.created",
                "session": {
                    "id": "session_mock",
                    "object": "realtime.session",
                    "type": "realtime",
                    "model": "mock",
                    "sglang": {"granted": None},
                },
            }
        )
        async for frame in self.websocket:
            arrival_s = time.perf_counter()
            event = json.loads(frame)
            kind = event["type"]
            if kind == "session.update":
                await self.send(
                    {
                        "type": "session.updated",
                        "client_event_id": event["event_id"],
                        "session": {
                            "id": "session_mock",
                            "object": "realtime.session",
                            "type": "realtime",
                            "model": "mock",
                            "output_modalities": ["audio"],
                            "sglang": {"granted": self.mock.granted},
                        },
                    }
                )
            elif kind == APPEND_EVENT:
                sequence = event["sglang"]["seq"]
                if self.first_append_s is None:
                    self.first_append_s = arrival_s
                else:
                    pass
                self.mock.record_arrival(
                    arrival_s - self.first_append_s - sequence * PACKET_MS / 1000,
                    arrival_s,
                )
                self.input_bytes += len(base64.b64decode(event["audio"]))
                await self.send(
                    {
                        "type": "sglang.input_audio.accepted",
                        "seq": sequence,
                        "accepted_end_ms": self.input_bytes / 32,
                        "client_event_id": event["event_id"],
                    }
                )
                complete_units = self.input_bytes // PACKET_BYTES
                if complete_units - self.units_emitted >= self.mock.burst_units:
                    for unit in range(self.units_emitted, complete_units):
                        await self.emit_unit(unit, PACKET_MS)
                else:
                    pass
            elif kind == "sglang.input_audio.end":
                accepted_ms = self.input_bytes / 32
                units = -(-self.input_bytes // PACKET_BYTES)
                await self.send(
                    {
                        "type": "sglang.input_audio.ended",
                        "accepted_end_ms": accepted_ms,
                        "tail_policy": "pad",
                        "client_event_id": event["event_id"],
                    }
                )
                for unit in range(self.units_emitted, units):
                    await self.emit_unit(
                        unit, min(PACKET_MS, accepted_ms - unit * PACKET_MS)
                    )
                await self.send(
                    {
                        "type": "response.output_audio.done",
                        "response_id": "response_0",
                        "item_id": "item_0",
                        "output_index": 0,
                        "content_index": 0,
                    }
                )
                await self.send(
                    {
                        "type": "response.done",
                        "response": {
                            "id": "response_0",
                            "object": "realtime.response",
                            "status": "completed",
                            "status_details": {"reason": "stop"},
                            "output": [],
                            "usage": {},
                        },
                    }
                )
                await self.send(
                    {
                        "type": "sglang.input_audio.drained",
                        "accepted_end_ms": accepted_ms,
                        "consumed_ms": accepted_ms,
                        "discarded_ms": 0,
                        "padding_ms": units * PACKET_MS - accepted_ms,
                        "client_event_id": event["event_id"],
                    }
                )
            elif kind == "session.close":
                await self.send(
                    {
                        "type": "session.closed",
                        "reason": "client_closed",
                        "client_event_id": event["event_id"],
                    }
                )
                return
            else:
                pass


class NativeMock:
    """Native session protocol that streams output audio back at a realistic rate."""

    def __init__(
        self,
        output_sample_rate: int,
        burst_units: int,
        transcript_deltas: bool,
    ) -> None:
        unit_samples = output_sample_rate * PACKET_MS // 1000
        # note (edwingao28): Noise, so the base64 text is not a run of one character.
        rng = np.random.default_rng(0)
        self.unit_audio: str = base64.b64encode(
            rng.integers(-64, 64, unit_samples, dtype="<i2").tobytes()
        ).decode("ascii")
        self.burst_units = burst_units
        self.transcript_deltas = transcript_deltas
        self.granted: dict[str, JsonValue] = {
            **NATIVE_GRANTED,
            "output_audio_format": {"type": "audio/pcm", "rate": output_sample_rate},
        }
        self.connections = 0
        self.arrival_deviations_s: list[float] = []
        self.late_arrivals: list[LatePacket] = []
        self.loop_lags: list[LoopLag] = []
        self.first_arrival_s: float | None = None
        self.first_arrival_cpu_s = 0.0
        self.last_arrival_s = 0.0
        self.last_arrival_cpu_s = 0.0

    def record_arrival(self, deviation_s: float, arrival_s: float) -> None:
        if self.first_arrival_s is None:
            self.first_arrival_s = arrival_s
            self.first_arrival_cpu_s = time.process_time()
        else:
            pass
        self.last_arrival_s = arrival_s
        self.last_arrival_cpu_s = time.process_time()
        self.arrival_deviations_s.append(deviation_s)
        if abs(deviation_s) > LATE_PACKET_RECORD_S:
            self.late_arrivals.append(
                LatePacket(time_s=arrival_s, deviation_s=deviation_s)
            )
        else:
            pass

    async def handle(self, websocket: websockets.ServerConnection) -> None:
        self.connections += 1
        await NativeSession(self, websocket).handle()

    summarize = MockServer.summarize


async def serve_native_mock(
    host: str,
    port: int,
    output_sample_rate: int,
    burst_units: int,
    transcript_deltas: bool,
    stats_output: Path,
) -> None:
    mock = NativeMock(output_sample_rate, burst_units, transcript_deltas)
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
        print(f"Native mock server listening on ws://{host}:{port}", flush=True)
        await stop.wait()
    lag_task.cancel()
    stats_output.write_text(json.dumps(mock.summarize(), indent=2) + "\n")


PROBE_SERVER: dict[str, JsonValue] = {
    "revision": "0" * 40,
    "revision_source": "operator_supplied",
    "model": "pacing-probe-mock",
}
RECORD_SUBSETS = (
    "user_interruption",
    "user_backchannel",
    "talking_to_other",
    "background_speech",
)
RECORD_METADATA: dict[str, dict[str, str]] = {
    "user_interruption": {"current_turn_text": "wait"},
    "user_backchannel": {"backchannel_text": "yeah"},
    "talking_to_other": {"current_turn_text": "one second"},
    "background_speech": {"background_text": "rain"},
}


class GcPauses:
    def __init__(self) -> None:
        self.started_s = 0.0
        self.pauses_s: list[float] = []

    def callback(self, phase: str, info: dict[str, int]) -> None:
        if phase == "start":
            self.started_s = time.perf_counter()
        else:
            self.pauses_s.append(time.perf_counter() - self.started_s)


def write_probe_dataset(
    root: Path, per_subset: int, min_s: float, max_s: float, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    for subset in RECORD_SUBSETS:
        for index in range(per_subset):
            sample_dir = root / subset / str(index + 1)
            sample_dir.mkdir(parents=True)
            duration_s = float(rng.uniform(min_s, max_s))
            samples = rng.integers(
                -2000, 2000, round(duration_s * SAMPLE_RATE), dtype="<i2"
            )
            for filename in ("input.wav", "clean_input.wav"):
                soundfile.write(
                    str(sample_dir / filename), samples, SAMPLE_RATE, subtype="PCM_16"
                )
            (sample_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "context_text": "hello",
                        **RECORD_METADATA[subset],
                        "timestamps": [0.1, 0.3],
                    }
                )
            )


async def record_probe(
    url: str,
    concurrency: int,
    per_subset: int,
    min_s: float,
    max_s: float,
    seed: int,
    output: Path,
    keep_sessions: bool,
) -> None:
    dataset = output / "dataset"
    write_probe_dataset(dataset, per_subset, min_s, max_s, seed)
    run_dir = output / "run"
    loop_lags: list[LoopLag] = []
    gc_pauses = GcPauses()
    gc.callbacks.append(gc_pauses.callback)
    lag_task = asyncio.create_task(measure_loop_lag(loop_lags))
    wall_started_s = time.perf_counter()
    cpu_started_s = time.process_time()
    result = await run_samples(
        dataset,
        url=url,
        output=run_dir,
        server=PROBE_SERVER,
        dataset_revision="pacing-probe",
        timeout_s=max_s + SESSION_TIMEOUT_MARGIN_S,
        concurrency=concurrency,
    )
    wall_s = time.perf_counter() - wall_started_s
    cpu_s = time.process_time() - cpu_started_s
    lag_task.cancel()
    gc.callbacks.remove(gc_pauses.callback)
    deviations_s: list[float] = []
    variants = []
    for sample in result["samples"]:
        for variant in sample["variants"].values():
            receipts_path = run_dir / variant["directory"] / SEND_RECEIPTS_FILE
            appends = (
                SendReceipts.model_validate_json(receipts_path.read_text()).appends
                if receipts_path.is_file()
                else []
            )
            origin_s = appends[0].start_s if appends else 0.0
            session_deviations_s = [
                append.start_s - origin_s - append.seq * PACKET_MS / 1000
                for append in appends
            ]
            deviations_s.extend(session_deviations_s)
            variants.append(
                {
                    "status": variant["status"],
                    "within_tolerance": (variant["input_timing"] or {}).get(
                        "within_tolerance"
                    ),
                    "max_abs_deviation_s": max(
                        (abs(value) for value in session_deviations_s), default=None
                    ),
                }
            )
    load = result["load"]
    summary: dict[str, JsonValue] = {
        "concurrency": concurrency,
        "variants": len(variants),
        "variant_status": {
            status: sum(variant["status"] == status for variant in variants)
            for status in sorted({variant["status"] for variant in variants})
        },
        "input_pacing_violations": load["input_pacing_violations"],
        "unqualified": load["input_pacing_violations"] > 0,
        "max_input_pacing_deviation_s": load["max_input_pacing_deviation_s"],
        "packets": len(deviations_s),
        "packets_over_tolerance": sum(
            abs(value) > PACING_TOLERANCE_S for value in deviations_s
        ),
        **percentiles_ms([abs(value) for value in deviations_s], "pacing_abs"),
        **percentiles_ms([lag.lag_s for lag in loop_lags], "loop_lag"),
        "loop_lags_over_20_ms": sum(lag.lag_s > 0.02 for lag in loop_lags),
        "gc_collections": len(gc_pauses.pauses_s),
        "gc_pause_max_ms": max(gc_pauses.pauses_s, default=0.0) * 1000,
        "gc_pause_total_ms": sum(gc_pauses.pauses_s) * 1000,
        "recording_wall_s": load["recording_wall_s"],
        "wall_s": wall_s,
        "process_cpu_cores": cpu_s / wall_s,
        "peak_rss_megabytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / KILOBYTES_PER_MEGABYTE,
        "python": platform.python_version(),
        "websockets": websockets.__version__,
    }
    (output / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n")
    (output / "variants.json").write_text(json.dumps(variants) + "\n")
    if not keep_sessions:
        shutil.rmtree(run_dir / "samples")
        shutil.rmtree(dataset)
    else:
        pass
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
    native_parser = subparsers.add_parser("native-mock")
    native_parser.add_argument("--host", default="127.0.0.1")
    native_parser.add_argument("--port", type=int, default=8765)
    native_parser.add_argument(
        "--output-sample-rate", type=int, default=DEFAULT_OUTPUT_SAMPLE_RATE
    )
    native_parser.add_argument("--burst-units", type=int, default=DEFAULT_BURST_UNITS)
    native_parser.add_argument("--no-transcript-deltas", action="store_true")
    native_parser.add_argument("--stats-output", type=Path, required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--url", default="ws://127.0.0.1:8765")
    record_parser.add_argument("--concurrency", type=int, required=True)
    record_parser.add_argument("--per-subset", type=int, required=True)
    record_parser.add_argument("--min-duration", type=float, default=20.0)
    record_parser.add_argument("--max-duration", type=float, default=60.0)
    record_parser.add_argument("--seed", type=int, default=0)
    record_parser.add_argument("--output", type=Path, required=True)
    record_parser.add_argument("--keep-sessions", action="store_true")
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
    elif arguments.command == "native-mock":
        asyncio.run(
            serve_native_mock(
                arguments.host,
                arguments.port,
                arguments.output_sample_rate,
                arguments.burst_units,
                not arguments.no_transcript_deltas,
                arguments.stats_output,
            )
        )
    elif arguments.command == "record":
        arguments.output.mkdir(parents=True)
        asyncio.run(
            record_probe(
                arguments.url,
                arguments.concurrency,
                arguments.per_subset,
                arguments.min_duration,
                arguments.max_duration,
                arguments.seed,
                arguments.output,
                arguments.keep_sessions,
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
