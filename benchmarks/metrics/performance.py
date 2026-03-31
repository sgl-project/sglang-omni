# SPDX-License-Identifier: Apache-2.0
"""System performance metrics: latency, RTF, throughput, token throughput."""

from __future__ import annotations

import numpy as np

from benchmarks.benchmarker.data import RequestResult


def _compute_token_metrics(successes: list[RequestResult]) -> dict:
    tokens_per_sec = [o.tok_per_s for o in successes if o.tok_per_s > 0]
    gen_token_counts = [
        o.completion_tokens for o in successes if o.completion_tokens > 0
    ]
    total_tokens = sum(gen_token_counts)
    total_engine_time = sum(o.engine_time_s for o in successes if o.engine_time_s > 0)

    prompt_token_counts = [o.prompt_tokens for o in successes if o.prompt_tokens > 0]

    token_metrics: dict = {}
    if tokens_per_sec:
        token_metrics["tok_per_s_mean"] = round(float(np.mean(tokens_per_sec)), 1)
        token_metrics["tok_per_s_median"] = round(float(np.median(tokens_per_sec)), 1)
    if total_engine_time > 0 and total_tokens > 0:
        token_metrics["tok_per_s_agg"] = round(total_tokens / total_engine_time, 1)
    if gen_token_counts:
        token_metrics["gen_tokens_mean"] = round(float(np.mean(gen_token_counts)), 0)
        token_metrics["gen_tokens_total"] = total_tokens
    if prompt_token_counts:
        token_metrics["prompt_tokens_mean"] = round(
            float(np.mean(prompt_token_counts)), 0
        )
        token_metrics["prompt_tokens_total"] = sum(prompt_token_counts)
    return token_metrics


def compute_speed_metrics(
    outputs: list[RequestResult], wall_clock_s: float | None = None
) -> dict:
    """Compute system performance summary from a list of request results."""
    successes = [o for o in outputs if o.is_success]
    if not successes:
        return {"completed_requests": 0, "failed_requests": len(outputs)}

    latencies = [o.latency_s for o in successes]
    rtfs = [o.rtf for o in successes if 0 < o.rtf < float("inf")]
    audio_durations = [o.audio_duration_s for o in successes if o.audio_duration_s > 0]

    if wall_clock_s is not None and wall_clock_s > 0:
        throughput = round(len(successes) / wall_clock_s, 3)
    else:
        total_latency = sum(latencies)
        throughput = (
            round(len(successes) / total_latency, 3) if total_latency > 0 else 0
        )

    metrics_summary: dict = {
        "completed_requests": len(successes),
        "failed_requests": len(outputs) - len(successes),
        "latency_mean_s": round(float(np.mean(latencies)), 3),
        "latency_median_s": round(float(np.median(latencies)), 3),
        "latency_p95_s": round(float(np.percentile(latencies, 95)), 3),
        "latency_p99_s": round(float(np.percentile(latencies, 99)), 3),
        "audio_duration_mean_s": (
            round(float(np.mean(audio_durations)), 3) if audio_durations else 0
        ),
        "rtf_mean": round(float(np.mean(rtfs)), 4) if rtfs else None,
        "rtf_median": round(float(np.median(rtfs)), 4) if rtfs else None,
        "throughput_qps": throughput,
        **_compute_token_metrics(successes),
    }
    return metrics_summary
