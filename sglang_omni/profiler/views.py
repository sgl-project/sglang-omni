# SPDX-License-Identifier: Apache-2.0
"""Profiler views: per-request timeline, stage breakdown, hop breakdown.

Reads JSONL events emitted by :mod:`sglang_omni.profiler.event_recorder`.
Input may be a single file, a directory of ``events_*_<pid>.jsonl``, or a
list of either. Framework-free so reports can be regenerated locally.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------


def iter_events(source: str | Path | Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    """Yield every JSON event from a file, directory, or list of either."""
    paths: list[Path] = []
    if isinstance(source, (str, Path)):
        sources: list[str | Path] = [source]
    else:
        sources = list(source)
    for raw in sources:
        p = Path(raw).expanduser()
        if p.is_dir():
            paths.extend(sorted(p.glob("events_*.jsonl")))
        elif p.is_file():
            paths.append(p)
        else:
            logger.warning("iter_events: skipping non-existent path %s", p)
    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                line = line.strip()
                if not line:
                    continue
                else:
                    pass
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "iter_events: skipping malformed line %d in %s",
                        line_no,
                        path,
                    )


def load_events(source: str | Path | Iterable[str | Path]) -> list[dict[str, Any]]:
    """Return all events sorted by ``timestamp_ns`` (stable)."""
    events = list(iter_events(source))
    events.sort(key=lambda e: e.get("timestamp_ns", 0))
    return events


# ---------------------------------------------------------------------------
# View 1: per-request timeline
# ---------------------------------------------------------------------------


@dataclass
class RequestTimeline:
    """All events for a single request, sorted by time."""

    request_id: str
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def t0_ns(self) -> int | None:
        if not self.events:
            return None
        else:
            pass
        return self.events[0]["timestamp_ns"]

    @property
    def t_end_ns(self) -> int | None:
        if not self.events:
            return None
        else:
            pass
        return self.events[-1]["timestamp_ns"]

    @property
    def total_ms(self) -> float:
        if not self.events:
            return 0.0
        else:
            pass
        t0 = self.t0_ns
        t1 = self.t_end_ns
        assert t0 is not None and t1 is not None
        return (t1 - t0) / 1e6

    def to_relative(self) -> list[dict[str, Any]]:
        """Return events with an added ``t_rel_ms`` field anchored at t0."""
        if not self.events:
            return []
        else:
            pass
        t0 = self.t0_ns
        assert t0 is not None
        result = []
        for ev in self.events:
            out = dict(ev)
            out["t_rel_ms"] = (ev["timestamp_ns"] - t0) / 1e6
            result.append(out)
        return result


def reconstruct_timelines(
    source: str | Path | Iterable[str | Path],
) -> dict[str, RequestTimeline]:
    """Group every event by ``request_id`` into a per-request timeline."""
    events = load_events(source)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        rid = ev.get("request_id")
        if not rid:
            continue
        else:
            pass
        grouped[rid].append(ev)
    return {
        rid: RequestTimeline(request_id=rid, events=evts)
        for rid, evts in grouped.items()
    }


# ---------------------------------------------------------------------------
# View 2: stage breakdown
# ---------------------------------------------------------------------------


# (opener, closer) pairs framing a stage-local interval. Opener can
# appear in multiple pairs (e.g. prefill_start closes against both
# first_emit and first_stream_chunk_sent → thinker TTFT / talker TTFCC).
_STAGE_INTERVAL_EVENTS = (
    ("stage_input_received", "stage_complete"),
    ("encoder_start", "encoder_end"),
    ("preprocess_start", "preprocess_end"),
    ("scheduler_request_build_start", "scheduler_request_build_end"),
    ("scheduler_request_build_end", "scheduler_queue_enter"),
    ("scheduler_queue_enter", "scheduler_prefill_start"),
    ("scheduler_prefill_start", "scheduler_prefill_end"),
    ("scheduler_prefill_end", "stage_complete"),
    ("scheduler_prefill_start", "stage_first_stream_chunk_sent"),
    ("scheduler_prefill_start", "scheduler_first_emit"),
    ("scheduler_first_emit", "stage_complete"),
)


@dataclass
class StageInterval:
    """One occurrence of (open_event, close_event) inside a stage."""

    request_id: str
    stage: str
    open_event: str
    close_event: str
    open_ns: int
    close_ns: int

    @property
    def duration_ms(self) -> float:
        return (self.close_ns - self.open_ns) / 1e6


@dataclass
class StageBreakdownRow:
    stage: str
    interval_name: str
    count: int
    total_ms: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "interval": self.interval_name,
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(self.avg_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile. q in [0, 1]. ``values`` is sorted."""
    if not values:
        return 0.0
    else:
        pass
    if len(values) == 1:
        return values[0]
    else:
        pass
    k = (len(values) - 1) * q
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    else:
        pass
    return values[f] + (values[c] - values[f]) * (k - f)


def compute_stage_intervals(
    timelines: dict[str, RequestTimeline] | None = None,
    *,
    source: str | Path | Iterable[str | Path] | None = None,
    interval_events: tuple[tuple[str, str], ...] = _STAGE_INTERVAL_EVENTS,
) -> list[StageInterval]:
    """Pair open/close events into (stage, request_id)-scoped intervals."""
    if timelines is None:
        if source is None:
            raise ValueError("compute_stage_intervals requires timelines or source")
        else:
            pass
        timelines = reconstruct_timelines(source)
    else:
        pass

    # Per-pair lookup: one opener event seeds every pair it's in; each
    # closer consumes only its own pair's stack.
    opens_to_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    closes_to_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for opener, closer in interval_events:
        opens_to_pairs[opener].append((opener, closer))
        closes_to_pairs[closer].append((opener, closer))

    out: list[StageInterval] = []
    for rid, tl in timelines.items():
        pending: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for ev in tl.events:
            name = ev.get("event_name")
            stage = ev.get("stage", "unknown")
            ts = int(ev.get("timestamp_ns", 0))
            # An event may be both opener and closer (chained intervals).
            if name in opens_to_pairs:
                for opener, closer in opens_to_pairs[name]:
                    pending[(stage, opener, closer)].append(ts)
            else:
                pass
            if name in closes_to_pairs:
                for opener, closer in closes_to_pairs[name]:
                    stack = pending.get((stage, opener, closer))
                    if not stack:
                        continue
                    else:
                        pass
                    open_ns = stack.pop(0)
                    out.append(
                        StageInterval(
                            request_id=rid,
                            stage=stage,
                            open_event=opener,
                            close_event=closer,
                            open_ns=open_ns,
                            close_ns=ts,
                        )
                    )
            else:
                pass
    return out


def stage_breakdown(
    timelines: dict[str, RequestTimeline] | None = None,
    *,
    source: str | Path | Iterable[str | Path] | None = None,
) -> list[StageBreakdownRow]:
    """Aggregate interval durations by (stage, open/close pair)."""
    intervals = compute_stage_intervals(timelines, source=source)
    bucket: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for it in intervals:
        key = (it.stage, it.open_event, it.close_event)
        bucket[key].append(it.duration_ms)
    rows: list[StageBreakdownRow] = []
    for (stage, open_ev, close_ev), durations in bucket.items():
        durations.sort()
        rows.append(
            StageBreakdownRow(
                stage=stage,
                interval_name=f"{open_ev}->{close_ev}",
                count=len(durations),
                total_ms=sum(durations),
                avg_ms=sum(durations) / len(durations),
                p50_ms=percentile(durations, 0.50),
                p95_ms=percentile(durations, 0.95),
                max_ms=durations[-1],
            )
        )
    rows.sort(key=lambda r: (-r.total_ms, r.stage))
    return rows


# ---------------------------------------------------------------------------
# View 3: hop breakdown (stage_a -> stage_b)
# ---------------------------------------------------------------------------


@dataclass
class HopBreakdownRow:
    src_stage: str
    dst_stage: str
    kind: str  # "payload" or "stream_chunk"
    count: int
    total_ms: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src_stage,
            "dst": self.dst_stage,
            "kind": self.kind,
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(self.avg_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


def hop_breakdown(
    timelines: dict[str, RequestTimeline] | None = None,
    *,
    source: str | Path | Iterable[str | Path] | None = None,
) -> list[HopBreakdownRow]:
    """Pair `stage_hop_sent` / `stage_input_received` and stream-chunk variants.

    Matching is keyed by ``(request_id, src_stage, dst_stage, chunk_id?)``.
    ``chunk_id`` distinguishes stream-chunk hops; payload hops have none.
    """
    if timelines is None:
        if source is None:
            raise ValueError("hop_breakdown requires timelines or source")
        else:
            pass
        timelines = reconstruct_timelines(source)
    else:
        pass

    # Pending sends keyed by (rid, src, dst, kind, chunk_id_or_None) -> list of ts.
    pending: dict[tuple[str, str, str, str, int | None], list[int]] = defaultdict(list)
    durations: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for rid, tl in timelines.items():
        for ev in tl.events:
            name = ev.get("event_name")
            md = ev.get("metadata") or {}
            ts = int(ev.get("timestamp_ns", 0))
            stage = ev.get("stage", "unknown")
            if name == "stage_hop_sent":
                dst = md.get("to_stage", "?")
                pending[(rid, stage, dst, "payload", None)].append(ts)
            elif name == "stage_stream_chunk_sent":
                dst = md.get("to_stage", "?")
                chunk_id = md.get("chunk_id")
                pending[(rid, stage, dst, "stream_chunk", chunk_id)].append(ts)
            elif name == "stage_input_received":
                src = md.get("from_stage")
                if not src or src == "coordinator":
                    continue
                else:
                    pass
                key = (rid, src, stage, "payload", None)
                stack = pending.get(key)
                if stack:
                    open_ns = stack.pop(0)
                    durations[(src, stage, "payload")].append((ts - open_ns) / 1e6)
                else:
                    pass
            elif name == "stage_stream_chunk_received":
                src = md.get("from_stage")
                chunk_id = md.get("chunk_id")
                if not src:
                    continue
                else:
                    pass
                key = (rid, src, stage, "stream_chunk", chunk_id)
                stack = pending.get(key)
                if stack:
                    open_ns = stack.pop(0)
                    durations[(src, stage, "stream_chunk")].append((ts - open_ns) / 1e6)
                else:
                    pass
            else:
                pass

    rows: list[HopBreakdownRow] = []
    for (src, dst, kind), values in durations.items():
        values.sort()
        rows.append(
            HopBreakdownRow(
                src_stage=src,
                dst_stage=dst,
                kind=kind,
                count=len(values),
                total_ms=sum(values),
                avg_ms=sum(values) / len(values),
                p50_ms=percentile(values, 0.50),
                p95_ms=percentile(values, 0.95),
                max_ms=values[-1],
            )
        )
    rows.sort(key=lambda r: (-r.total_ms, r.src_stage, r.dst_stage))
    return rows


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


MetricStats = dict[str, float | int]
ServingSummary = dict[str, dict[str, MetricStats | float | int | None]]


def serving_summary(timelines: dict[str, RequestTimeline]) -> ServingSummary:
    """Aggregate request intervals and unweighted execution samples by stage.

    Graph attempt success uses hits plus explicit fallbacks as its denominator.
    Missing observations are omitted; an unobserved attempt success rate is null.
    """
    samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    modes: dict[str, Counter[str]] = defaultdict(Counter)
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    retractions: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    interval_names = {
        ("scheduler_queue_enter", "scheduler_prefill_start"): "queue_wait_ms",
        ("scheduler_prefill_start", "scheduler_prefill_end"): "prefill_ms",
        (
            "scheduler_request_build_start",
            "scheduler_request_build_end",
        ): "request_build_ms",
        ("code2wav_decode_start", "code2wav_decode_end"): "decode_ms",
        ("code2wav_batch_start", "code2wav_batch_end"): "batch_ms",
    }
    for interval in compute_stage_intervals(
        timelines, interval_events=tuple(interval_names)
    ):
        samples[interval.stage][
            interval_names[(interval.open_event, interval.close_event)]
        ].append(interval.duration_ms)

    for timeline in timelines.values():
        for event in timeline.events:
            stage = event.get("stage", "unknown")
            name = event.get("event_name")
            metadata = event.get("metadata") or {}
            if name == "scheduler_request_retracted":
                retractions[stage] += 1
            elif name == "scheduler_batch_start":
                fields = {
                    "batch_size": f"{metadata.get('batch_type', 'unknown')}_batch_size",
                    **{
                        key: key
                        for key in (
                            "running_requests",
                            "waiting_requests",
                            "kv_usage",
                            "kv_used_tokens",
                            "kv_available_tokens",
                            "kv_evictable_tokens",
                            "request_build_pending",
                            "request_build_backlog",
                        )
                    },
                }
                for field_name, metric in fields.items():
                    value = metadata.get(field_name)
                    if isinstance(value, (int, float)):
                        samples[stage][metric].append(value)
                    else:
                        pass
            elif name in ("code2wav_batch_start", "code2wav_decode_start"):
                for field_name in (
                    "batch_size",
                    "active_request_count",
                    "inbox_depth",
                    "oldest_wait_ms",
                ):
                    value = metadata.get(field_name)
                    if isinstance(value, (int, float)):
                        samples[stage][field_name].append(value)
                    else:
                        pass
            elif name in ("code2wav_batch_end", "code2wav_decode_end"):
                # note (AkazaAkane): sub-batches, when present, are authoritative even if empty.
                executions = metadata.get("sub_batch_execution", [metadata])
                for execution in executions:
                    mode = execution.get("execution_mode")
                    if isinstance(mode, str):
                        modes[stage][mode] += 1
                    else:
                        pass
                    value = execution.get(
                        "batch_size", 1 if name == "code2wav_decode_end" else None
                    )
                    if isinstance(value, (int, float)):
                        samples[stage]["effective_batch_size"].append(value)
                    else:
                        pass
                    reason = execution.get("fallback_reason")
                    if isinstance(reason, str) and reason:
                        reasons[stage][reason] += 1
                    else:
                        pass
                    if mode == "eager" and (
                        reason or execution.get("graph_requested") is True
                    ):
                        fallback_counts[stage] += 1
                    else:
                        pass
            else:
                pass

    summary: ServingSummary = {}
    for stage in sorted(set(samples) | set(modes) | set(retractions)):
        metrics: dict[str, MetricStats | float | int | None] = {}
        for name, values in samples[stage].items():
            values.sort()
            metrics[name] = {
                "count": len(values),
                "avg": round(sum(values) / len(values), 3),
                "p50": round(percentile(values, 0.50), 3),
                "p95": round(percentile(values, 0.95), 3),
                "max": values[-1],
            }
        if stage in retractions or any(
            name in samples[stage] for name in ("running_requests", "waiting_requests")
        ):
            metrics["observed_retractions"] = retractions[stage]
        else:
            pass
        if stage in modes:
            hits = modes[stage]["cuda_graph"]
            fallbacks = fallback_counts[stage]
            metrics.update(
                execution_mode=dict(modes[stage]),
                fallback_reason=dict(reasons[stage]),
                graph_hit_count=hits,
                graph_fallback_count=fallbacks,
                graph_attempt_success_rate=(
                    hits / (hits + fallbacks) if hits + fallbacks else None
                ),
            )
        else:
            pass
        summary[stage] = metrics
    return summary


def format_serving_summary(summary: ServingSummary) -> str:
    """Render serving metrics with explicit sample counts and graph denominators."""
    lines = ["=== Serving Summary ==="]
    if not summary:
        lines.append("(empty)")
    else:
        pass
    for stage, metrics in summary.items():
        lines.append(stage)
        for name, value in metrics.items():
            if isinstance(value, dict):
                rendered = "  ".join(f"{key}={number}" for key, number in value.items())
            elif name == "graph_attempt_success_rate" and value is not None:
                rendered = f"{value:.1%} (hits / graph attempts)"
            else:
                rendered = "n/a" if value is None else str(value)
            lines.append(f"  {name:<24} {rendered}")
    return "\n".join(lines) + "\n"


def build_report(source: str | Path | Iterable[str | Path]) -> dict[str, Any]:
    """Return all profiler views as a single dict for JSON serialization."""
    timelines = reconstruct_timelines(source)
    return {
        "timelines": {rid: tl.to_relative() for rid, tl in timelines.items()},
        "stage_breakdown": [row.to_dict() for row in stage_breakdown(timelines)],
        "hop_breakdown": [row.to_dict() for row in hop_breakdown(timelines)],
        "request_count": len(timelines),
        "serving_summary": serving_summary(timelines),
    }


def format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Pretty-print a list of dicts as a fixed-width table."""
    if not rows:
        return "(empty)\n"
    else:
        pass
    widths = {
        c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in columns
    }
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    body = "\n".join(
        " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns) for r in rows
    )
    return f"{header}\n{sep}\n{body}\n"
