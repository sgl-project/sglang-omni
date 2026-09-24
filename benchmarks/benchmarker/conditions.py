# SPDX-License-Identifier: Apache-2.0
"""Shared recording for benchmark runs that compare speed across revisions."""

from __future__ import annotations

import argparse
import logging
import statistics
from typing import Literal, TypedDict

from benchmarks.benchmarker.fingerprint import (
    BenchmarkFingerprint,
    collect_environment_fingerprint,
    collect_server_identity,
)

logger = logging.getLogger(__name__)

# note (wilsonzheng0327): below this count p99 interpolates the two slowest requests.
TAIL_PERCENTILE_MIN_SAMPLES = 100


class MetricAggregate(TypedDict):
    mean: float | None
    min: float | None
    max: float | None
    n: int


SweepMetricName = Literal[
    "throughput_qps",
    "audio_throughput_s_per_s",
    "latency_mean_s",
    "latency_median_s",
    "latency_p95_s",
    "latency_p99_s",
    "audio_ttfp_mean_s",
    "audio_ttfp_p95_s",
    "rtf_mean",
    "audio_duration_mean_s",
    "warmup",
]
SWEEP_METRIC_NAMES: tuple[SweepMetricName, ...] = (
    "throughput_qps",
    "audio_throughput_s_per_s",
    "latency_mean_s",
    "latency_median_s",
    "latency_p95_s",
    "latency_p99_s",
    "audio_ttfp_mean_s",
    "audio_ttfp_p95_s",
    "rtf_mean",
    "audio_duration_mean_s",
    "warmup",
)


class RepeatSpeedSummary(TypedDict, total=False):
    repeat: int
    output_dir: str
    completed_requests: int
    failed_requests: int
    throughput_qps: float
    audio_throughput_s_per_s: float
    latency_mean_s: float
    latency_median_s: float
    latency_p95_s: float
    latency_p99_s: float
    audio_ttfp_mean_s: float
    audio_ttfp_p95_s: float
    rtf_mean: float | None
    audio_duration_mean_s: float
    warmup: int


class ConcurrencyAggregate(TypedDict):
    concurrency: int
    repeats: int
    completed_requests: int
    failed_requests: int
    throughput_qps: MetricAggregate
    audio_throughput_s_per_s: MetricAggregate
    latency_mean_s: MetricAggregate
    latency_median_s: MetricAggregate
    latency_p95_s: MetricAggregate
    latency_p99_s: MetricAggregate
    audio_ttfp_mean_s: MetricAggregate
    audio_ttfp_p95_s: MetricAggregate
    rtf_mean: MetricAggregate
    audio_duration_mean_s: MetricAggregate
    warmup: MetricAggregate
    per_repeat: list[RepeatSpeedSummary]


def warn_if_tail_percentile_is_thin(sample_count: int) -> None:
    if sample_count >= TAIL_PERCENTILE_MIN_SAMPLES:
        return
    logger.warning(
        f"latency_p99_s interpolates the two slowest of {sample_count} requests; "
        f"use at least {TAIL_PERCENTILE_MIN_SAMPLES} samples before citing tails"
    )


def collect_run_fingerprint(base_url: str) -> BenchmarkFingerprint:
    return {
        "client": collect_environment_fingerprint(),
        "server": collect_server_identity(base_url),
    }


def fingerprint_fields(
    enabled: bool,
    base_url: str,
) -> dict[str, BenchmarkFingerprint]:
    if not enabled:
        return {}
    return {"environment_fingerprint": collect_run_fingerprint(base_url)}


def add_fingerprint_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fingerprint",
        action="store_true",
        help="Record the client environment and the server /v1/models identity.",
    )


def add_talker_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--talker-temperature",
        type=float,
        default=None,
        help="Talker sampling temperature. Unset keeps the server default.",
    )
    parser.add_argument("--talker-top-p", type=float, default=None)
    parser.add_argument("--talker-top-k", type=int, default=None)
    parser.add_argument("--talker-repetition-penalty", type=float, default=None)


def aggregate_numbers(values: list[float | int | None]) -> MetricAggregate:
    present: list[float] = []
    for value in values:
        if value is None:
            continue
        present.append(float(value))
    if not present:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {
        "mean": statistics.mean(present),
        "min": min(present),
        "max": max(present),
        "n": len(present),
    }


def aggregate_metric(
    summaries: list[RepeatSpeedSummary], metric_name: SweepMetricName
) -> MetricAggregate:
    return aggregate_numbers([summary.get(metric_name) for summary in summaries])


def aggregate_repeats(
    concurrency: int, summaries: list[RepeatSpeedSummary]
) -> ConcurrencyAggregate:
    """Aggregate one concurrency level, keeping every raw repeat row."""
    return {
        "concurrency": concurrency,
        "repeats": len(summaries),
        "completed_requests": sum(
            summary.get("completed_requests", 0) for summary in summaries
        ),
        "failed_requests": sum(
            summary.get("failed_requests", 0) for summary in summaries
        ),
        "throughput_qps": aggregate_metric(summaries, "throughput_qps"),
        "audio_throughput_s_per_s": aggregate_metric(
            summaries, "audio_throughput_s_per_s"
        ),
        "latency_mean_s": aggregate_metric(summaries, "latency_mean_s"),
        "latency_median_s": aggregate_metric(summaries, "latency_median_s"),
        "latency_p95_s": aggregate_metric(summaries, "latency_p95_s"),
        "latency_p99_s": aggregate_metric(summaries, "latency_p99_s"),
        "audio_ttfp_mean_s": aggregate_metric(summaries, "audio_ttfp_mean_s"),
        "audio_ttfp_p95_s": aggregate_metric(summaries, "audio_ttfp_p95_s"),
        "rtf_mean": aggregate_metric(summaries, "rtf_mean"),
        "audio_duration_mean_s": aggregate_metric(summaries, "audio_duration_mean_s"),
        "warmup": aggregate_metric(summaries, "warmup"),
        "per_repeat": summaries,
    }


def sampling_seed_field(seed: int | None) -> dict[str, int]:
    if seed is None:
        return {}
    return {"seed": seed}
