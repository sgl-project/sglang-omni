# SPDX-License-Identifier: Apache-2.0
"""Pure derivations over a :class:`SessionTrace`.

Three groups, each a function of the trace alone:

* :func:`check_invariants` - protocol rules that must hold for the trace to be
  worth measuring. Returns the violations instead of asserting so a sweep can
  keep going and CI can assert on an empty list.
* :func:`latency_metrics` - client-observed latencies. Definitions are fixed
  here so numbers stay comparable across runs:

  - first_partial_latency_s: per segment, from the send time of the first
    packet whose cumulative audio reached ``segment_start + decode_interval``
    (the server's first refresh point) to the first non-final
    transcription.segment for that segment. Segment start is
    speech_started.audio_start_ms under server VAD, 0 for the first
    segment without VAD, and unknown (skipped) otherwise.
  - partial_interval_s: receive-time gaps between consecutive partials of
    the same segment.
  - final_latency_s: per segment, input_audio_buffer.committed to its
    is_final segment event.
  - done_to_completed_s: transcription.done sent to
    transcription.completed received.

* :func:`wer_metrics` - transcription.completed text against a reference,
  reusing the SeedTTS ASR normalization and jiwer path.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.realtime_asr.client import SAMPLE_RATE, SessionTrace
from benchmarks.tasks.asr import apply_wer


def check_invariants(trace: SessionTrace) -> list[str]:
    """Return every protocol violation found in ``trace`` (empty means clean)."""
    violations: list[str] = []
    if trace.error:
        violations.append(f"client error: {trace.error}")

    for item in trace.events("error"):
        violations.append(f"server error event: {item.event.get('error')}")

    indexes = [
        item.event.get("event_index")
        for item in trace.received
        if "event_index" in item.event
    ]
    if any(not isinstance(index, int) for index in indexes):
        violations.append("event_index missing or non-integer on some events")
    else:
        for previous, current in zip(indexes, indexes[1:]):
            if current <= previous:
                violations.append(
                    f"event_index not strictly increasing: {previous} -> {current}"
                )
                break

    committed_ids = [
        item.event.get("segment_id")
        for item in trace.events("input_audio_buffer.committed")
    ]
    final_ids = [item.event.get("segment_id") for item in trace.segments(is_final=True)]
    if sorted(committed_ids, key=_sort_key) != sorted(final_ids, key=_sort_key):
        violations.append(
            f"committed segments {committed_ids} do not match final segments {final_ids}"
        )
    if len(set(final_ids)) != len(final_ids):
        violations.append(f"duplicate final segment ids: {final_ids}")

    finalized: set[Any] = set()
    for item in trace.events("transcription.segment"):
        segment_id = item.event.get("segment_id")
        if segment_id in finalized:
            violations.append(f"segment {segment_id} updated after its final event")
            break
        if item.event.get("is_final"):
            finalized.add(segment_id)

    if not trace.events("transcription.completed"):
        violations.append("no transcription.completed event")
    return violations


def _sort_key(value: Any) -> tuple[int, str]:
    return (0, "") if isinstance(value, int) else (1, str(value))


def _segment_starts_ms(trace: SessionTrace) -> dict[Any, float]:
    """Audio offset (ms) at which each segment started, where knowable."""
    starts: dict[Any, float] = {}
    for item in trace.events("input_audio_buffer.speech_started"):
        segment_id = item.event.get("segment_id")
        if segment_id not in starts:
            starts[segment_id] = float(item.event.get("audio_start_ms", 0))
    if not starts and trace.session.get("turn_detection") is None:
        # Manual mode: the first segment starts with the first sample.
        starts[0] = 0.0
    return starts


def _packet_reaching(trace: SessionTrace, audio_ms: float) -> float | None:
    """Send time of the first packet whose cumulative audio reaches ``audio_ms``.

    Mirrors the server's ``buffer_end >= start + interval`` check in samples.
    """
    threshold_samples = math.ceil(audio_ms * SAMPLE_RATE / 1000.0)
    for packet in trace.sent:
        if packet.audio_end_samples >= threshold_samples:
            return packet.send_s
    return None


def latency_metrics(trace: SessionTrace) -> dict[str, Any]:
    decode_interval_ms = trace.session.get("decode_interval_ms")
    partials = trace.segments(is_final=False)
    finals = trace.segments(is_final=True)

    partials_by_segment: dict[Any, list[float]] = defaultdict(list)
    for item in partials:
        partials_by_segment[item.event.get("segment_id")].append(item.recv_s)

    first_partial_latency: dict[Any, float] = {}
    if isinstance(decode_interval_ms, (int, float)):
        for segment_id, start_ms in _segment_starts_ms(trace).items():
            arrivals = partials_by_segment.get(segment_id)
            if not arrivals:
                continue
            trigger_s = _packet_reaching(trace, start_ms + decode_interval_ms)
            if trigger_s is None:
                continue
            first_partial_latency[segment_id] = arrivals[0] - trigger_s

    partial_interval: list[float] = []
    for arrivals in partials_by_segment.values():
        partial_interval.extend(b - a for a, b in zip(arrivals, arrivals[1:]))

    committed_at = {
        item.event.get("segment_id"): item.recv_s
        for item in trace.events("input_audio_buffer.committed")
    }
    final_latency: dict[Any, float] = {}
    for item in finals:
        segment_id = item.event.get("segment_id")
        if segment_id in committed_at:
            final_latency[segment_id] = item.recv_s - committed_at[segment_id]

    done_to_completed: float | None = None
    completed = trace.events("transcription.completed")
    if completed and trace.done_sent_s is not None:
        done_to_completed = completed[-1].recv_s - trace.done_sent_s

    return {
        "decode_interval_ms": decode_interval_ms,
        "segment_count": len(finals),
        "partial_count": len(partials),
        "first_partial_latency_s": list(first_partial_latency.values()),
        "partial_interval_s": partial_interval,
        "final_latency_s": list(final_latency.values()),
        "done_to_completed_s": done_to_completed,
        "audio_duration_s": trace.audio_duration_s,
        "wall_s": trace.wall_s,
    }


def wer_metrics(trace: SessionTrace, ref_text: str, *, lang: str) -> SampleOutput:
    """Score transcription.completed against ref_text.

    Returns the same :class:`SampleOutput` the HTTP ASR benchmarks use, so the
    result can be fed to calculate_wer_metrics for corpus-level numbers.
    """
    output = SampleOutput(target_text=ref_text, audio_duration_s=trace.audio_duration_s)
    if trace.wall_s is not None:
        output.latency_s = trace.wall_s
    text = trace.completed_text
    if text is None:
        output.error = trace.error or "No transcription.completed event"
        return output
    return apply_wer(output, text, lang)


def paired_corpus_wer(
    stream_outputs: list[SampleOutput],
    http_outputs: list[SampleOutput],
    *,
    lang: str,
) -> dict[str, Any]:
    """Compare two transcription paths on the samples that succeeded in both.

    Corpus WER silently drops failed samples, so subtracting two corpus WERs
    computed over different successful sets measures the set difference, not
    the transcripts. The delta here is taken over the intersection only and is
    ``None`` when the intersection is empty.
    """
    stream_ok = {o.sample_id: o for o in stream_outputs if o.is_success}
    http_ok = {o.sample_id: o for o in http_outputs if o.is_success}
    common = sorted(stream_ok.keys() & http_ok.keys())
    if not common:
        return {
            "common_evaluated": 0,
            "stream_corpus_wer_common": None,
            "http_corpus_wer_common": None,
            "corpus_wer_delta_vs_http": None,
        }
    stream_wer = calculate_wer_metrics([stream_ok[i] for i in common], lang)
    http_wer = calculate_wer_metrics([http_ok[i] for i in common], lang)
    return {
        "common_evaluated": len(common),
        "stream_corpus_wer_common": stream_wer["wer_corpus"],
        "http_corpus_wer_common": http_wer["wer_corpus"],
        "corpus_wer_delta_vs_http": stream_wer["wer_corpus"] - http_wer["wer_corpus"],
    }


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; values must be non-empty."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def summarize(values: list[float]) -> dict[str, float | int] | None:
    """mean/p50/p95/min/max/n over values; None when empty."""
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


__all__ = [
    "check_invariants",
    "latency_metrics",
    "paired_corpus_wer",
    "percentile",
    "summarize",
    "wer_metrics",
]
