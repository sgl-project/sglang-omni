from __future__ import annotations

import statistics

from .types import PerRequestResult, RunSummary


def compute_run_summary(
    *,
    results: list[PerRequestResult],
    run_wall_time_s: float,
    model_profile: str,
    case_id: str,
    scenario_id: str,
    request_family: str,
) -> RunSummary:
    successes = [result for result in results if result.success]
    failures = len(results) - len(successes)

    summary = RunSummary(
        model_profile=model_profile,
        case=case_id,
        scenario=scenario_id,
        request_family=request_family,
        completed_requests=len(successes),
        failed_requests=failures,
        run_wall_time_s=run_wall_time_s,
    )

    if not successes:
        return summary

    latencies = [result.latency_s for result in successes]
    audio_durations = _non_null(result.audio_duration_s for result in successes)
    rtfs = _positive(result.rtf for result in successes)
    tok_rates = _positive(result.tok_per_s for result in successes)
    prompt_tokens = _positive_int(result.prompt_tokens for result in successes)
    completion_tokens = _positive_int(
        result.completion_tokens for result in successes
    )
    engine_times = _positive(result.engine_time_s for result in successes)

    summary.latency_mean_s = statistics.fmean(latencies)
    summary.latency_median_s = statistics.median(latencies)
    summary.latency_p95_s = _percentile(latencies, 95)
    summary.latency_p99_s = _percentile(latencies, 99)
    if run_wall_time_s > 0:
        summary.throughput_qps = len(successes) / run_wall_time_s
    if audio_durations:
        summary.audio_duration_mean_s = statistics.fmean(audio_durations)
    if rtfs:
        summary.rtf_mean = statistics.fmean(rtfs)
        summary.rtf_median = statistics.median(rtfs)
    if tok_rates:
        summary.tok_per_s_mean = statistics.fmean(tok_rates)
        summary.tok_per_s_median = statistics.median(tok_rates)
    if prompt_tokens:
        summary.prompt_tokens_mean = statistics.fmean(prompt_tokens)
        summary.prompt_tokens_total = sum(prompt_tokens)
    if completion_tokens:
        summary.completion_tokens_mean = statistics.fmean(completion_tokens)
        summary.completion_tokens_total = sum(completion_tokens)
    if completion_tokens and engine_times and sum(engine_times) > 0:
        summary.tok_per_s_agg = sum(completion_tokens) / sum(engine_times)

    return summary


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _non_null(values) -> list[float]:
    return [value for value in values if value is not None]


def _positive(values) -> list[float]:
    return [value for value in values if value is not None and value > 0]


def _positive_int(values) -> list[int]:
    return [value for value in values if value is not None and value > 0]

