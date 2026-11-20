# SPDX-License-Identifier: Apache-2.0
"""Record per-chunk stream observations and aggregate TTFA, RTF, and stream health."""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import statistics
from dataclasses import dataclass
from typing import Any

from benchmarks.tts_serving.stream_stress.config_parser import StressConfig

LOGGER = logging.getLogger(__name__)
WAV_HEADER_BYTES = 44


def percentile(values: list[float], percentile_rank: float) -> float | None:
    """Return a linearly interpolated percentile matching numpy defaults."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_rank / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "p50": round(percentile(values, 50), 6),
        "p90": round(percentile(values, 90), 6),
        "p95": round(percentile(values, 95), 6),
        "p99": round(percentile(values, 99), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def compute_playback_underrun(
    chunk_records: list[ChunkRecord],
) -> tuple[float | None, int]:
    """Simulate a real-time player and report the worst playback starvation."""
    audio_chunks = [chunk for chunk in chunk_records if chunk.audio_bytes > 0]
    if len(audio_chunks) < 2:
        return None, 0
    deadline = audio_chunks[0].arrival_s + audio_chunks[0].audio_duration_s
    maximum = 0.0
    count = 0
    for chunk in audio_chunks[1:]:
        underrun = max(0.0, chunk.arrival_s - deadline)
        if underrun > 1e-9:
            count += 1
        maximum = max(maximum, underrun)
        deadline = max(deadline, chunk.arrival_s) + chunk.audio_duration_s
    return maximum, count


@dataclass(kw_only=True)
class ChunkRecord:
    """One observed HTTP chunk or SSE frame."""

    index: int
    arrival_s: float
    raw_bytes: int
    audio_bytes: int
    audio_duration_s: float
    inter_chunk_s: float | None = None
    kind: str = "audio"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "arrival_s": round(self.arrival_s, 6),
            "inter_chunk_s": (
                round(self.inter_chunk_s, 6) if self.inter_chunk_s is not None else None
            ),
            "raw_bytes": self.raw_bytes,
            "audio_bytes": self.audio_bytes,
            "audio_duration_s": round(self.audio_duration_s, 6),
            "kind": self.kind,
        }


@dataclass(kw_only=True)
class RequestRecord:
    """One complete request observation, serialized as one jsonl row."""

    run_id: str
    request_id: str
    sequence: int
    warmup: bool
    labels: dict[str, str]
    sample_id: str
    text: str

    planned_at_s: float
    dispatch_lag_s: float
    sent_at_s: float
    http_status: int | None
    content_type: str | None
    server_request_id: str | None
    speech_id: str | None
    worker: str | None
    sample_rate: int
    channels: int
    bit_depth: int

    chunks: list[ChunkRecord]
    chunk_count: int
    audio_chunk_count: int
    first_chunk_s: float | None
    ttfa_s: float | None
    first_audio_payload_bytes: int
    latency_s: float
    audio_bytes: int
    audio_duration_s: float
    rtf: float | None

    inter_chunk_s: list[float]
    max_inter_chunk_s: float | None
    max_playback_underrun_s: float | None
    underrun_count: int
    continuity_pass: dict[str, bool]

    finish_reason: str | None
    server_duration_s: float | None
    usage: dict[str, Any] | None
    outcome: dict[str, Any] | None

    is_success: bool
    is_broken_stream: bool
    broken_reasons: list[str]
    error: str | None

    def to_dict(self, *, omit_chunks: bool = False) -> dict[str, Any]:
        return {
            "record_type": "request",
            "run_id": self.run_id,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "warmup": self.warmup,
            "labels": self.labels,
            "sample_id": self.sample_id,
            "text": self.text,
            "timing": {
                "planned_at_s": round(self.planned_at_s, 6),
                "sent_at_s": round(self.sent_at_s, 6),
                "dispatch_lag_s": round(self.dispatch_lag_s, 6),
                "first_chunk_s": (
                    round(self.first_chunk_s, 6)
                    if self.first_chunk_s is not None
                    else None
                ),
                "ttfa_s": round(self.ttfa_s, 6) if self.ttfa_s is not None else None,
                "latency_s": round(self.latency_s, 6),
            },
            "response": {
                "http_status": self.http_status,
                "content_type": self.content_type,
                "server_request_id": self.server_request_id,
                "speech_id": self.speech_id,
                "worker": self.worker,
                "finish_reason": self.finish_reason,
                "server_duration_s": self.server_duration_s,
                "usage": self.usage,
                "outcome": self.outcome,
            },
            "audio": {
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "bit_depth": self.bit_depth,
                "audio_bytes": self.audio_bytes,
                "audio_duration_s": round(self.audio_duration_s, 6),
                "rtf": round(self.rtf, 6) if self.rtf is not None else None,
                "chunk_count": self.chunk_count,
                "audio_chunk_count": self.audio_chunk_count,
                "first_audio_payload_bytes": self.first_audio_payload_bytes,
            },
            "stream_health": {
                "inter_chunk_s": [round(value, 6) for value in self.inter_chunk_s],
                "max_inter_chunk_s": (
                    round(self.max_inter_chunk_s, 6)
                    if self.max_inter_chunk_s is not None
                    else None
                ),
                "max_playback_underrun_s": (
                    round(self.max_playback_underrun_s, 6)
                    if self.max_playback_underrun_s is not None
                    else None
                ),
                "underrun_count": self.underrun_count,
                "continuity_pass": self.continuity_pass,
                "is_broken_stream": self.is_broken_stream,
                "broken_reasons": self.broken_reasons,
            },
            "is_success": self.is_success,
            "error": self.error,
            "chunks": (
                [] if omit_chunks else [chunk.to_dict() for chunk in self.chunks]
            ),
        }


class StreamRecorder:
    """Accumulate one request's chunk observations into a RequestRecord."""

    def __init__(
        self,
        config: StressConfig,
        *,
        sequence: int,
        request_id: str,
        sample_id: str,
        text: str,
        warmup: bool,
        planned_at_s: float = 0.0,
    ) -> None:
        self.config = config
        self.sequence = sequence
        self.request_id = request_id
        self.sample_id = sample_id
        self.text = text
        self.warmup = warmup
        self.planned_at_s = planned_at_s

        self.sent_at: float | None = None
        self.completed_at: float | None = None
        self.http_status: int | None = None
        self.content_type: str | None = None
        self.headers: dict[str, str] = {}
        self.sample_rate = config.sample_rate or config.fallback_sample_rate
        self.channels = config.channels
        self.bit_depth = config.bit_depth

        self.chunks: list[ChunkRecord] = []
        self.raw_bytes_total = 0
        self.audio_bytes_total = 0
        self.first_chunk_s: float | None = None
        self.first_audio_s: float | None = None
        self.first_audio_payload_bytes = 0

        self.error: str | None = None
        self.error_kind: str | None = None
        self.sse_done: dict[str, Any] | None = None
        self.sse_error: dict[str, Any] | None = None
        self.outcome: dict[str, Any] | None = None

        self.wav_header_buffer = bytearray()
        self.expect_wav_header = config.response_format == "wav"
        self.health_failures: set[str] = set()

    @property
    def bytes_per_second(self) -> float:
        return float(self.sample_rate * self.channels * self.bit_depth // 8)

    def mark_sent(self, observed_at: float) -> None:
        self.sent_at = observed_at

    def start_response(self, status: int, headers: dict[str, str]) -> None:
        self.http_status = status
        self.headers = {name.lower(): value for name, value in headers.items()}
        self.content_type = self.headers.get("content-type")
        if self.config.sample_rate is None:
            self.sample_rate = positive_int(
                self.headers.get("x-sample-rate"), self.config.fallback_sample_rate
            )
        self.channels = positive_int(
            self.headers.get("x-channels"), self.config.channels
        )
        self.bit_depth = positive_int(
            self.headers.get("x-bit-depth"), self.config.bit_depth
        )

    def add_raw_bytes(
        self, network_bytes: bytes, observed_at: float, *, kind: str = "audio"
    ) -> None:
        """Record one HTTP chunk and strip the 44-byte streaming WAV header."""
        if not network_bytes:
            return
        self.raw_bytes_total += len(network_bytes)
        audio_payload = network_bytes
        if self.expect_wav_header:
            needed = WAV_HEADER_BYTES - len(self.wav_header_buffer)
            self.wav_header_buffer.extend(audio_payload[:needed])
            audio_payload = audio_payload[needed:]
            if len(self.wav_header_buffer) >= WAV_HEADER_BYTES:
                self.expect_wav_header = False
                if bytes(self.wav_header_buffer[:4]) != b"RIFF":
                    self.health_failures.add("invalid_wav_header")
        self.append_chunk(
            len(network_bytes), len(audio_payload), observed_at, kind=kind
        )

    def add_sse_audio(self, encoded: str, frame_bytes: int, observed_at: float) -> None:
        """Decode one base64 audio delta and strip a leading WAV header."""
        try:
            audio_payload = base64.b64decode(encoded, validate=False)
        except ValueError:
            self.health_failures.add("invalid_base64_audio")
            audio_payload = b""
        if self.expect_wav_header and audio_payload.startswith(b"RIFF"):
            audio_payload = audio_payload[WAV_HEADER_BYTES:]
            self.expect_wav_header = False
        elif self.expect_wav_header and audio_payload:
            self.expect_wav_header = False
        self.raw_bytes_total += frame_bytes
        self.append_chunk(
            frame_bytes, len(audio_payload), observed_at, kind="sse_delta"
        )

    def add_sse_done(
        self, event: dict[str, Any], frame_bytes: int, observed_at: float
    ) -> None:
        self.sse_done = event
        self.raw_bytes_total += frame_bytes
        self.append_chunk(frame_bytes, 0, observed_at, kind="sse_done")

    def add_sse_error(
        self, event: dict[str, Any], frame_bytes: int, observed_at: float
    ) -> None:
        self.sse_error = event
        self.raw_bytes_total += frame_bytes
        self.append_chunk(frame_bytes, 0, observed_at, kind="sse_error")
        self.fail("sse_error", f"SSE error event: {event.get('error', event)}")

    def append_chunk(
        self,
        raw_bytes: int,
        audio_bytes: int,
        observed_at: float,
        *,
        kind: str,
    ) -> None:
        if self.sent_at is None:
            self.sent_at = observed_at
        arrival_s = max(0.0, observed_at - self.sent_at)
        previous = self.chunks[-1].arrival_s if self.chunks else None
        record = ChunkRecord(
            index=len(self.chunks),
            arrival_s=arrival_s,
            raw_bytes=raw_bytes,
            audio_bytes=audio_bytes,
            audio_duration_s=(
                audio_bytes / self.bytes_per_second if audio_bytes else 0.0
            ),
            inter_chunk_s=(arrival_s - previous) if previous is not None else None,
            kind=kind,
        )
        self.chunks.append(record)
        if self.first_chunk_s is None:
            self.first_chunk_s = arrival_s
        if audio_bytes > 0:
            self.audio_bytes_total += audio_bytes
            if self.first_audio_s is None:
                self.first_audio_s = arrival_s
                self.first_audio_payload_bytes = audio_bytes

    def fail(
        self, kind: str, message: str, *, observed_at: float | None = None
    ) -> None:
        if self.error is None:
            self.error_kind = kind
            self.error = message[:1000]
        self.health_failures.add(kind)
        if observed_at is not None and self.completed_at is None:
            self.completed_at = observed_at

    def mark_completed(self, observed_at: float) -> None:
        if self.completed_at is None:
            self.completed_at = observed_at

    def set_outcome(self, outcome: dict[str, Any]) -> None:
        self.outcome = outcome

    def finalize(self, *, load_start: float) -> RequestRecord:
        config = self.config
        sent_at = self.sent_at or load_start
        completed_at = self.completed_at or sent_at
        latency_s = max(0.0, completed_at - sent_at)
        audio_duration_s = self.audio_bytes_total / self.bytes_per_second

        inter_chunk = [
            chunk.inter_chunk_s
            for chunk in self.chunks
            if chunk.inter_chunk_s is not None
        ]
        max_inter_chunk = max(inter_chunk) if inter_chunk else None
        max_underrun, underrun_count = compute_playback_underrun(self.chunks)
        continuity_pass = {
            f"c{threshold}": (
                max_underrun is not None and max_underrun <= threshold / 1000.0
            )
            for threshold in config.underrun_thresholds_ms
        }

        self.health_failures.update(
            self.health_reasons(
                audio_duration_s=audio_duration_s,
                max_inter_chunk=max_inter_chunk,
                max_underrun=max_underrun,
            )
        )
        broken_reasons = sorted(self.health_failures)
        fatal = {
            "http_error",
            "transport_error",
            "timeout",
            "chunk_timeout",
            "cancelled",
            "no_audio",
            "partial_audio_frame",
            "audio_too_short",
            "invalid_wav_header",
            "invalid_base64_audio",
            "sse_error",
            "missing_done_event",
            "bad_finish_reason",
            "malformed_sse_frame",
            "unknown_sse_event",
            "truncated_sse_frame",
        }
        is_success = (
            self.http_status == 200
            and self.error is None
            and self.audio_bytes_total > 0
            and not (self.health_failures & fatal)
        )

        finish_reason = None
        server_duration_s = None
        usage = None
        if self.sse_done:
            finish_reason = self.sse_done.get("finish_reason")
            server_duration_s = self.sse_done.get("duration")
            usage = self.sse_done.get("usage")
        if self.outcome:
            finish_reason = finish_reason or self.outcome.get("finish_reason")
            usage = usage or self.outcome.get("usage")

        return RequestRecord(
            run_id=config.run_id,
            request_id=self.request_id,
            sequence=self.sequence,
            warmup=self.warmup,
            labels=dict(config.labels),
            sample_id=self.sample_id,
            text=self.text,
            planned_at_s=self.planned_at_s,
            dispatch_lag_s=max(0.0, sent_at - load_start - self.planned_at_s),
            sent_at_s=max(0.0, sent_at - load_start),
            http_status=self.http_status,
            content_type=self.content_type,
            server_request_id=self.headers.get("x-request-id"),
            speech_id=self.headers.get("x-sglang-omni-speech-id"),
            worker=self.headers.get("x-sglang-omni-worker"),
            sample_rate=self.sample_rate,
            channels=self.channels,
            bit_depth=self.bit_depth,
            chunks=self.chunks,
            chunk_count=len(self.chunks),
            audio_chunk_count=sum(1 for chunk in self.chunks if chunk.audio_bytes > 0),
            first_chunk_s=self.first_chunk_s,
            ttfa_s=self.first_audio_s,
            first_audio_payload_bytes=self.first_audio_payload_bytes,
            latency_s=latency_s,
            audio_bytes=self.audio_bytes_total,
            audio_duration_s=audio_duration_s,
            rtf=(latency_s / audio_duration_s) if audio_duration_s > 0 else None,
            inter_chunk_s=inter_chunk,
            max_inter_chunk_s=max_inter_chunk,
            max_playback_underrun_s=max_underrun,
            underrun_count=underrun_count,
            continuity_pass=continuity_pass,
            finish_reason=finish_reason,
            server_duration_s=server_duration_s,
            usage=usage,
            outcome=self.outcome,
            is_success=is_success,
            is_broken_stream=bool(broken_reasons),
            broken_reasons=broken_reasons,
            error=self.error,
        )

    def health_reasons(
        self,
        *,
        audio_duration_s: float,
        max_inter_chunk: float | None,
        max_underrun: float | None,
    ) -> set[str]:
        config = self.config
        reasons: set[str] = set()
        if self.http_status is not None and self.http_status != 200:
            reasons.add("http_error")
        if not self.chunks or self.audio_bytes_total == 0:
            reasons.add("no_audio")
        if self.audio_bytes_total % config.block_align:
            reasons.add("partial_audio_frame")
        if self.audio_bytes_total and audio_duration_s < config.min_audio_duration_s:
            reasons.add("audio_too_short")
        if max_inter_chunk is not None and max_inter_chunk > config.stall_broken_s:
            reasons.add("chunk_stall")
        if (
            max_underrun is not None
            and max_underrun > config.broken_underrun_ms / 1000.0
        ):
            reasons.add("playback_underrun")
        if (
            config.stream_format == "sse"
            and self.sse_done is None
            and self.sse_error is None
        ):
            reasons.add("missing_done_event")
        if self.sse_done:
            reason = self.sse_done.get("finish_reason")
            if isinstance(reason, str) and reason.lower() in (
                "error",
                "abort",
                "aborted",
                "cancelled",
                "canceled",
                "timeout",
                "failed",
                "content_filter",
            ):
                reasons.add("bad_finish_reason")
        return reasons


def positive_int(raw: str | None, fallback: int) -> int:
    try:
        value = int(raw) if raw is not None else fallback
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


@dataclass(kw_only=True)
class AggregateReport:
    """Aggregated metrics consumed by the builder."""

    run_id: str
    labels: dict[str, str]
    wall_clock_s: float
    total_requests: int
    warmup_requests: int
    dispatched: int
    succeeded: int
    failed: int
    broken_streams: int
    broken_rate: float
    success_rate: float
    achieved_qpm: float
    throughput_qps: float
    audio_seconds_total: float
    audio_throughput_s_per_s: float
    ttfa_s: dict[str, Any]
    latency_s: dict[str, Any]
    rtf: dict[str, Any]
    audio_duration_s: dict[str, Any]
    chunk_count: dict[str, Any]
    inter_chunk_s: dict[str, Any]
    max_playback_underrun_s: dict[str, Any]
    continuity_pass_rate_pct: dict[str, float | None]
    broken_reasons: dict[str, int]
    http_status_counts: dict[str, int]
    load: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "summary",
            "run_id": self.run_id,
            "labels": self.labels,
            "wall_clock_s": round(self.wall_clock_s, 6),
            "requests": {
                "total": self.total_requests,
                "warmup": self.warmup_requests,
                "dispatched": self.dispatched,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "success_rate": round(self.success_rate, 6),
                "broken_streams": self.broken_streams,
                "broken_rate": round(self.broken_rate, 6),
                "achieved_qpm": round(self.achieved_qpm, 3),
                "throughput_qps": round(self.throughput_qps, 6),
                "audio_seconds_total": round(self.audio_seconds_total, 3),
                "audio_throughput_s_per_s": round(self.audio_throughput_s_per_s, 6),
            },
            "metrics": {
                "ttfa_s": self.ttfa_s,
                "latency_s": self.latency_s,
                "rtf": self.rtf,
                "audio_duration_s": self.audio_duration_s,
                "chunk_count": self.chunk_count,
                "inter_chunk_s": self.inter_chunk_s,
                "max_playback_underrun_s": self.max_playback_underrun_s,
                "continuity_pass_rate_pct": self.continuity_pass_rate_pct,
            },
            "health": {
                "broken_reasons": self.broken_reasons,
                "http_status_counts": self.http_status_counts,
            },
            "load": self.load,
        }


class Responder:
    """Collect finished request records and build the aggregate report."""

    def __init__(self, config: StressConfig) -> None:
        self.config = config
        self.request_records: list[RequestRecord] = []
        self.submit_lock = asyncio.Lock()

    @property
    def records(self) -> list[RequestRecord]:
        return self.request_records

    async def submit(self, record: RequestRecord) -> None:
        async with self.submit_lock:
            self.request_records.append(record)
            completed = len(self.request_records)
        if (
            not self.config.quiet
            and self.config.log_every
            and completed % self.config.log_every == 0
        ):
            LOGGER.info(
                f"completed={completed} success={record.is_success} "
                f"ttfa_s={format_metric(record.ttfa_s)} "
                f"rtf={format_metric(record.rtf)} "
                f"broken={record.is_broken_stream}"
            )

    def build_report(
        self,
        *,
        wall_clock_s: float,
        load_stats: dict[str, Any],
    ) -> AggregateReport:
        scored = [record for record in self.request_records if not record.warmup]
        warmups = [record for record in self.request_records if record.warmup]
        successes = [record for record in scored if record.is_success]
        broken = [record for record in scored if record.is_broken_stream]

        underruns = [
            record.max_playback_underrun_s
            for record in scored
            if record.max_playback_underrun_s is not None
        ]
        continuity: dict[str, float | None] = {}
        for threshold in self.config.underrun_thresholds_ms:
            continuity[f"c{threshold}"] = (
                round(
                    100.0
                    * sum(1 for value in underruns if value <= threshold / 1000.0)
                    / len(underruns),
                    2,
                )
                if underruns
                else None
            )

        reasons: dict[str, int] = {}
        statuses: dict[str, int] = {}
        for record in scored:
            for reason in record.broken_reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
            status_key = (
                str(record.http_status) if record.http_status is not None else "none"
            )
            statuses[status_key] = statuses.get(status_key, 0) + 1

        total = len(scored)
        audio_seconds = sum(record.audio_duration_s for record in successes)
        window = wall_clock_s if wall_clock_s > 0 else 1e-9
        load = dict(load_stats)
        load["planned_requests"] = self.config.total_requests
        load["qpm"] = self.config.qpm
        load["request_rate"] = round(self.config.request_rate, 6)
        load["arrival_mode"] = self.config.arrival_mode
        load["max_concurrency"] = self.config.max_concurrency
        load["warmup_completed"] = len(warmups)

        return AggregateReport(
            run_id=self.config.run_id,
            labels=dict(self.config.labels),
            wall_clock_s=wall_clock_s,
            total_requests=total,
            warmup_requests=len(warmups),
            dispatched=total,
            succeeded=len(successes),
            failed=total - len(successes),
            broken_streams=len(broken),
            broken_rate=(len(broken) / total) if total else 0.0,
            success_rate=(len(successes) / total) if total else 0.0,
            achieved_qpm=total / window * 60.0,
            throughput_qps=total / window,
            audio_seconds_total=audio_seconds,
            audio_throughput_s_per_s=audio_seconds / window,
            ttfa_s=summarize(
                [record.ttfa_s for record in successes if record.ttfa_s is not None]
            ),
            latency_s=summarize([record.latency_s for record in successes]),
            rtf=summarize(
                [record.rtf for record in successes if record.rtf is not None]
            ),
            audio_duration_s=summarize(
                [record.audio_duration_s for record in successes]
            ),
            chunk_count=summarize([float(record.chunk_count) for record in successes]),
            inter_chunk_s=summarize(
                [value for record in successes for value in record.inter_chunk_s]
            ),
            max_playback_underrun_s=summarize(underruns),
            continuity_pass_rate_pct=continuity,
            broken_reasons=reasons,
            http_status_counts=statuses,
            load=load,
        )


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
