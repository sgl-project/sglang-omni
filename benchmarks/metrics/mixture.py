# SPDX-License-Identifier: Apache-2.0
"""Fixed window metrics for mixed streaming and non streaming TTS load.

A load NS+S runs NS non streaming and S streaming closed loop clients on one server. A summary counts the requests whose response completed inside the measured window, 0 <= finished_at_s <= duration_s, and every throughput divides by duration_s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from benchmarks.benchmarker.conditions import MetricAggregate, aggregate_numbers
from benchmarks.benchmarker.data import MixtureWindow, RequestClass, RequestResult
from benchmarks.metrics._format import (
    SPEED_LABEL_WIDTH,
    SPEED_LINE_WIDTH,
    print_speed_metric_line,
)
from benchmarks.metrics.performance import compute_speed_metrics

MixtureCategory = Literal["non_streaming", "streaming", "mixture"]

SUMMARY_METRICS: tuple[tuple[str, str], ...] = (
    ("completed_requests", "Completed in window:"),
    ("failed_requests", "Failed in window:"),
    ("audio_throughput_s_per_s", "Total audio (s/s):"),
    ("max_token_hits", "Max token hits:"),
)
CLASS_METRICS: dict[RequestClass, tuple[tuple[str, str], ...]] = {
    "non_streaming": (
        ("completed_requests", "Completed in window:"),
        ("failed_requests", "Failed in window:"),
        ("throughput_qps", "Requests per second:"),
        ("audio_throughput_s_per_s", "Audio (s/s):"),
        ("latency_median_s", "Latency p50 (s):"),
        ("latency_p95_s", "Latency p95 (s):"),
        ("latency_p99_s", "Latency p99 (s):"),
        ("rtf_mean", "RTF mean:"),
        ("max_token_hits", "Max token hits:"),
        ("finish_reason_observed", "Finish reasons observed:"),
        ("output_tokens_total", "Output tokens (total):"),
    ),
    "streaming": (
        ("completed_requests", "Completed in window:"),
        ("failed_requests", "Failed in window:"),
        ("throughput_qps", "Streams finished per second:"),
        ("audio_throughput_s_per_s", "Audio (s/s):"),
        ("audio_ttfp_median_s", "First audio p50 (s):"),
        ("audio_ttfp_p95_s", "First audio p95 (s):"),
        ("audio_ttfp_p99_s", "First audio p99 (s):"),
        ("max_playback_underrun_p95_s", "Max underrun p95 (s):"),
        ("max_playback_underrun_p99_s", "Max underrun p99 (s):"),
        ("max_playback_underrun_max_s", "Max underrun max (s):"),
        ("c50", "C50 (%):"),
        ("c100", "C100 (%):"),
        ("c200", "C200 (%):"),
        ("inter_chunk_p95_s", "ITL p95 (s):"),
        ("stream_finish_gap_max_s", "Longest no finish gap (s):"),
        ("max_token_hits", "Max token hits:"),
        ("finish_reason_observed", "Finish reasons observed:"),
        ("output_tokens_total", "Output tokens (total):"),
    ),
}
CLASS_TITLES: dict[RequestClass, str] = {
    "non_streaming": "Non streaming",
    "streaming": "Streaming",
}


@dataclass(frozen=True, kw_only=True)
class MixtureLoad:
    """NS non streaming plus S streaming closed loop clients, written NS+S."""

    non_streaming_clients: int
    streaming_clients: int

    @property
    def name(self) -> str:
        return f"ns{self.non_streaming_clients}_s{self.streaming_clients}"

    @property
    def total_clients(self) -> int:
        return self.non_streaming_clients + self.streaming_clients

    @property
    def category(self) -> MixtureCategory:
        if self.streaming_clients == 0:
            return "non_streaming"
        elif self.non_streaming_clients == 0:
            return "streaming"
        else:
            return "mixture"

    @property
    def clients_by_class(self) -> dict[RequestClass, int]:
        """Clients per class, leaving out a class with no clients."""
        all_classes: dict[RequestClass, int] = {
            "non_streaming": self.non_streaming_clients,
            "streaming": self.streaming_clients,
        }
        return {
            request_class: clients
            for request_class, clients in all_classes.items()
            if clients > 0
        }


class MixtureSummaries(TypedDict):
    summary: dict[str, float | int | None]
    summary_by_class: dict[RequestClass, dict[str, float | int | None]]
    mixture: MixtureWindow


class MixtureRepeatResult(MixtureSummaries):
    repeat: int
    output_dir: str


class NormalizedWork(TypedDict):
    value: MetricAggregate
    references_s_per_s: dict[RequestClass, float]


class MixtureLoadAggregate(TypedDict):
    load: str
    category: MixtureCategory
    non_streaming_clients: int
    streaming_clients: int
    repeats: int
    failed_requests: int
    summary: dict[str, MetricAggregate]
    summary_by_class: dict[RequestClass, dict[str, MetricAggregate]]
    normalized_work: NormalizedWork | None
    per_repeat: list[MixtureRepeatResult]


def stream_finish_gap_max_s(finish_times_s: list[float], duration_s: float) -> float:
    """Longest interval inside the window with no stream finishing, with both window edges counted."""
    inside_s = sorted(time_s for time_s in finish_times_s if 0 <= time_s <= duration_s)
    edges_s = [0.0, *inside_s, duration_s]
    gaps_s = [later_s - earlier_s for earlier_s, later_s in zip(edges_s, edges_s[1:])]
    return round(max(gaps_s), 4)


def compute_mixture_summaries(
    results: list[RequestResult],
    *,
    load: MixtureLoad,
    settle_s: float,
    duration_s: float,
    pool_size: int,
) -> MixtureSummaries:
    """Combined and per class summaries over the window, plus what happened outside it."""
    window_results: list[RequestResult] = []
    summary_by_class: dict[RequestClass, dict[str, float | int | None]] = {}
    issued_requests: dict[RequestClass, int] = {}
    pool_cycles: dict[RequestClass, float] = {}
    outside_window_requests: dict[RequestClass, int] = {}
    outside_window_failed_requests: dict[RequestClass, int] = {}
    for request_class in load.clients_by_class:
        class_results = [
            result for result in results if result.request_class == request_class
        ]
        class_window_results = [
            result
            for result in class_results
            if 0 <= result.finished_at_s <= duration_s
        ]
        class_outside_results = [
            result
            for result in class_results
            if not 0 <= result.finished_at_s <= duration_s
        ]
        class_summary = compute_speed_metrics(
            class_window_results, wall_clock_s=duration_s
        )
        class_summary["wall_clock_s"] = duration_s
        if request_class == "streaming":
            stream_finish_times_s = [
                result.finished_at_s for result in class_results if result.is_success
            ]
            class_summary["stream_finish_gap_max_s"] = stream_finish_gap_max_s(
                stream_finish_times_s, duration_s
            )
        else:
            pass
        summary_by_class[request_class] = class_summary
        issued_requests[request_class] = len(class_results)
        pool_cycles[request_class] = round(len(class_results) / pool_size, 3)
        outside_window_requests[request_class] = len(class_outside_results)
        outside_window_failed_requests[request_class] = sum(
            1 for result in class_outside_results if not result.is_success
        )
        window_results.extend(class_window_results)
    summary = compute_speed_metrics(window_results, wall_clock_s=duration_s)
    summary["wall_clock_s"] = duration_s
    # note (db-ol): compute_speed_metrics leaves both rates out when nothing succeeded. Over a fixed window that is a rate of 0, not a missing value.
    for rate_summary in (summary, *summary_by_class.values()):
        rate_summary.setdefault("throughput_qps", 0.0)
        rate_summary.setdefault("audio_throughput_s_per_s", 0.0)
    last_finish_s = max(
        (result.finished_at_s for result in results), default=duration_s
    )
    return {
        "summary": summary,
        "summary_by_class": summary_by_class,
        "mixture": {
            "load": load.name,
            "non_streaming_clients": load.non_streaming_clients,
            "streaming_clients": load.streaming_clients,
            "settle_s": settle_s,
            "duration_s": duration_s,
            "pool_size": pool_size,
            "issued_requests": issued_requests,
            "pool_cycles": pool_cycles,
            "outside_window_requests": outside_window_requests,
            "outside_window_failed_requests": outside_window_failed_requests,
            "drain_s": round(max(0.0, last_finish_s - duration_s), 3),
        },
    }


def normalized_work(
    summary_by_class: dict[RequestClass, dict[str, float | int | None]],
    references_s_per_s: dict[RequestClass, float],
) -> float | None:
    """Each class's audio throughput over its single mode reference, summed over classes."""
    audio_s_per_s = [
        summary_by_class[request_class].get("audio_throughput_s_per_s")
        for request_class in references_s_per_s
    ]
    if None in audio_s_per_s:
        return None
    else:
        return round(
            sum(
                class_audio_s_per_s / reference_s_per_s
                for class_audio_s_per_s, reference_s_per_s in zip(
                    audio_s_per_s, references_s_per_s.values()
                )
            ),
            4,
        )


def aggregate_mixture_sweep(
    repeat_results_by_load: dict[MixtureLoad, list[MixtureRepeatResult]],
) -> list[MixtureLoadAggregate]:
    """One row per load, with normalized work on mixture rows when both single mode classes ran."""
    rows: list[MixtureLoadAggregate] = []
    for load, repeat_results in repeat_results_by_load.items():
        rows.append(
            {
                "load": load.name,
                "category": load.category,
                "non_streaming_clients": load.non_streaming_clients,
                "streaming_clients": load.streaming_clients,
                "repeats": len(repeat_results),
                "failed_requests": sum(
                    result["summary"]["failed_requests"]
                    + sum(result["mixture"]["outside_window_failed_requests"].values())
                    for result in repeat_results
                ),
                "summary": {
                    name: aggregate_numbers(
                        [result["summary"].get(name) for result in repeat_results]
                    )
                    for name, _ in SUMMARY_METRICS
                },
                "summary_by_class": {
                    request_class: {
                        name: aggregate_numbers(
                            [
                                result["summary_by_class"][request_class].get(name)
                                for result in repeat_results
                            ]
                        )
                        for name, _ in CLASS_METRICS[request_class]
                    }
                    for request_class in load.clients_by_class
                },
                "normalized_work": None,
                "per_repeat": repeat_results,
            }
        )
    references_s_per_s: dict[RequestClass, float] = {}
    for request_class in CLASS_METRICS:
        single_mode_audio_s_per_s = [
            row["summary_by_class"][request_class]["audio_throughput_s_per_s"]["mean"]
            for row in rows
            if row["category"] == request_class
        ]
        positive_audio_s_per_s = [
            audio_mean_s_per_s
            for audio_mean_s_per_s in single_mode_audio_s_per_s
            if audio_mean_s_per_s is not None and audio_mean_s_per_s > 0
        ]
        if positive_audio_s_per_s:
            references_s_per_s[request_class] = max(positive_audio_s_per_s)
        else:
            pass
    is_normalizable = len(references_s_per_s) == len(CLASS_METRICS)
    for row in rows:
        if is_normalizable and row["category"] == "mixture":
            row["normalized_work"] = {
                "value": aggregate_numbers(
                    [
                        normalized_work(result["summary_by_class"], references_s_per_s)
                        for result in row["per_repeat"]
                    ]
                ),
                "references_s_per_s": references_s_per_s,
            }
        else:
            pass
    return rows


def print_mixture_summary(summaries: MixtureSummaries, model_name: str) -> None:
    label_width = SPEED_LABEL_WIDTH
    line_width = SPEED_LINE_WIDTH
    window = summaries["mixture"]
    print(f"\n{'=' * line_width}")
    print(f"{'TTS Mixture Benchmark Result':^{line_width}}")
    print(f"{'=' * line_width}")
    print(f"  {'Model:':<{label_width}} {model_name}")
    print(
        f"  {'Clients:':<{label_width}} {window['non_streaming_clients']} "
        f"non streaming + {window['streaming_clients']} streaming"
    )
    print(
        f"  {'Window (s):':<{label_width}} {window['duration_s']} "
        f"after {window['settle_s']} settle"
    )
    print(
        f"  {'Failed outside window:':<{label_width}} "
        f"{sum(window['outside_window_failed_requests'].values())}"
    )
    for name, label in SUMMARY_METRICS:
        print_speed_metric_line(label_width, label, summaries["summary"], name)
    for request_class, class_summary in summaries["summary_by_class"].items():
        print(f"{'-' * line_width}")
        print(f"  {CLASS_TITLES[request_class]}")
        for name, label in CLASS_METRICS[request_class]:
            print_speed_metric_line(label_width, label, class_summary, name)
    print(f"{'=' * line_width}")
