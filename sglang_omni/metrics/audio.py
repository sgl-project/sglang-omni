# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics for generated audio responses."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Histogram

from sglang_omni.metrics.types import (
    CONTINUITY_THRESHOLDS_MS,
    FAST_BUCKETS_S,
    LATENCY_BUCKETS_S,
    ORPHANED_STATE_TTL_NS,
    RTF_BUCKETS,
    MetricDetails,
    RuntimeMetricEvent,
)


@dataclass(kw_only=True)
class AudioState:
    last_ns: int
    duration_s: float
    buffered_s: float
    max_underrun_s: float = 0.0


class AudioMetrics:
    """Aggregate audio response timing and continuity events."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.ttfp = Histogram(
            "sglang_omni:audio_ttfp_s",
            "Time from HTTP request arrival to first streamed audio payload.",
            buckets=LATENCY_BUCKETS_S,
            registry=registry,
        )
        self.rtf = Histogram(
            "sglang_omni:audio_rtf",
            "Engine-reported generation time divided by output audio duration.",
            buckets=RTF_BUCKETS,
            registry=registry,
        )
        self.e2e = Histogram(
            "sglang_omni:audio_e2e_latency_s",
            "Time from HTTP request arrival to final audio response.",
            buckets=LATENCY_BUCKETS_S,
            registry=registry,
        )
        self.duration = Histogram(
            "sglang_omni:audio_duration_s",
            "Generated audio duration.",
            buckets=LATENCY_BUCKETS_S,
            registry=registry,
        )
        self.interval = Histogram(
            "sglang_omni:audio_chunk_interval_s",
            "Wall time between consecutive audio chunks.",
            buckets=FAST_BUCKETS_S,
            registry=registry,
        )
        self.underrun = Histogram(
            "sglang_omni:audio_underrun_s",
            "Largest playback buffer underrun per audio response.",
            buckets=FAST_BUCKETS_S,
            registry=registry,
        )
        self.continuity = Counter(
            "sglang_omni:audio_continuity_ok_total",
            "Audio streams without an underrun beyond the threshold.",
            ["threshold_ms"],
            registry=registry,
        )
        self.response_started_ns: dict[str, int] = {}
        self.streams: dict[str, AudioState] = {}

    def record(
        self,
        event_name: RuntimeMetricEvent,
        request_id: str,
        timestamp_ns: int,
        details: MetricDetails,
    ) -> None:
        if event_name == "audio_response_start":
            self.response_started_ns[request_id] = timestamp_ns
        elif event_name == "audio_response_aborted":
            self.response_started_ns.pop(request_id, None)
            self.streams.pop(request_id, None)
        elif event_name == "audio_response_done":
            self.record_done(request_id, timestamp_ns, details)
        elif event_name == "audio_chunk":
            self.record_chunk(request_id, timestamp_ns, details)

    def record_done(
        self, request_id: str, timestamp_ns: int, details: MetricDetails
    ) -> None:
        start_ns = self.response_started_ns.pop(request_id, None)
        audio = self.streams.pop(request_id, None)
        duration_s = (
            audio.duration_s if audio is not None else details.get("audio_duration_s")
        )
        if (
            start_ns is not None
            and isinstance(duration_s, (int, float))
            and duration_s > 0
        ):
            elapsed_s = max(0.0, (timestamp_ns - start_ns) / 1e9)
            self.e2e.observe(elapsed_s)
            self.duration.observe(duration_s)
            generation_s = details.get("audio_generation_s")
            if isinstance(generation_s, (int, float)) and generation_s >= 0:
                self.rtf.observe(generation_s / duration_s)
        if audio is not None and audio.duration_s > 0:
            self.underrun.observe(audio.max_underrun_s)
            for threshold_ms in CONTINUITY_THRESHOLDS_MS:
                if audio.max_underrun_s <= threshold_ms / 1000:
                    self.continuity.labels(str(threshold_ms)).inc()

    def record_chunk(
        self, request_id: str, timestamp_ns: int, details: MetricDetails
    ) -> None:
        duration_s = details.get("duration_s")
        if isinstance(duration_s, (int, float)) and duration_s > 0:
            audio = self.streams.get(request_id)
            if audio is None:
                self.streams[request_id] = AudioState(
                    last_ns=timestamp_ns,
                    duration_s=float(duration_s),
                    buffered_s=float(duration_s),
                )
                start_ns = self.response_started_ns.get(request_id)
                if start_ns is not None:
                    self.ttfp.observe(max(0.0, (timestamp_ns - start_ns) / 1e9))
            else:
                interval_s = max(0.0, (timestamp_ns - audio.last_ns) / 1e9)
                self.interval.observe(interval_s)
                underrun_s = max(0.0, interval_s - audio.buffered_s)
                audio.max_underrun_s = max(audio.max_underrun_s, underrun_s)
                audio.buffered_s = max(0.0, audio.buffered_s - interval_s) + duration_s
                audio.duration_s += duration_s
                audio.last_ns = timestamp_ns

    def prune(self, now_ns: int) -> None:
        expired = {
            request_id
            for request_id, started_ns in self.response_started_ns.items()
            if now_ns - started_ns > ORPHANED_STATE_TTL_NS
        }
        expired.update(
            request_id
            for request_id, state in self.streams.items()
            if now_ns - state.last_ns > ORPHANED_STATE_TTL_NS
        )
        for request_id in expired:
            self.response_started_ns.pop(request_id, None)
            self.streams.pop(request_id, None)
