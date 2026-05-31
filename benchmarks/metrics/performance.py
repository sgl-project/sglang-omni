# SPDX-License-Identifier: Apache-2.0
"""System performance metrics: latency, RTF, throughput, token throughput,
streaming UX.

Metric semantics:

``throughput_qps``
    Completed requests divided by measured benchmark wall-clock seconds.
``output_tokens_total``
    Sum of completion tokens across successful requests with completion tokens.
``output_tokens_mean``
    Mean completion tokens per successful request with completion tokens.
``output_throughput``
    Completion tokens divided by measured benchmark wall-clock seconds.
``output_tok_per_req_s``
    Completion tokens divided by summed per-request engine/request-time seconds.
``output_token_rate``
    Per-request completion tokens divided by that request's engine/request time.
``rtf_mean``
    Mean request elapsed seconds divided by generated output audio duration seconds.
``rtf_p95`` / ``rtf_p99``
    Tail percentiles of per-request RTF.
``audio_throughput_s_per_s``
    Total seconds of generated audio divided by benchmark wall-clock seconds.
    Independent of per-request audio duration; comparable across engines that
    emit different audio lengths for the same input.
``audio_ttfp_mean_s`` (TTFC)
    Mean time-to-first-audio-chunk: client-side wall time from request send
    to first decoded audio chunk arrival. Streaming only.
``audio_ttfp_median_s`` / ``audio_ttfp_p95_s`` / ``audio_ttfp_p99_s``
    Median / tail percentiles of TTFC.
``text_ttft_mean_s`` (TTFT)
    Mean time-to-first-text-token: client-side wall time from request send to
    first non-empty content delta. Streaming only; populated when the model
    emits text deltas (dual-modality text+audio output).
``text_ttft_median_s`` / ``text_ttft_p95_s`` / ``text_ttft_p99_s``
    Median / tail percentiles of TTFT.
``inter_chunk_mean_s`` (ITL)
    Mean inter-arrival latency between successive audio chunks within a
    request. Streaming smoothness metric.
``inter_chunk_p95_s`` / ``inter_chunk_p99_s``
    Tail percentiles of inter-chunk latency (streaming jitter).
"""

from __future__ import annotations

import numpy as np

from benchmarks.benchmarker.data import RequestResult
from benchmarks.metrics._format import SPEED_LABEL_WIDTH, SPEED_LINE_WIDTH


def _compute_token_metrics(
    successes: list[RequestResult],
    *,
    wall_clock_s: float | None,
) -> dict:
    output_token_counts = [
        o.completion_tokens for o in successes if o.completion_tokens > 0
    ]
    total_tokens = sum(output_token_counts)
    total_engine_time = sum(o.engine_time_s for o in successes if o.engine_time_s > 0)

    prompt_token_counts = [o.prompt_tokens for o in successes if o.prompt_tokens > 0]

    token_metrics: dict = {}
    if total_engine_time > 0 and total_tokens > 0:
        token_metrics["output_tok_per_req_s"] = round(
            total_tokens / total_engine_time,
            1,
        )
    if wall_clock_s is not None and wall_clock_s > 0 and total_tokens > 0:
        token_metrics["output_throughput"] = round(total_tokens / wall_clock_s, 1)
    if output_token_counts:
        token_metrics["output_tokens_mean"] = round(
            float(np.mean(output_token_counts)), 0
        )
        token_metrics["output_tokens_total"] = total_tokens
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
        return {
            "total_requests": len(outputs),
            "completed_requests": 0,
            "failed_requests": len(outputs),
        }

    latencies = [o.latency_s for o in successes]
    rtfs = [o.rtf for o in successes if 0 < o.rtf < float("inf")]
    audio_durations = [o.audio_duration_s for o in successes if o.audio_duration_s > 0]
    ttfps = [
        o.audio_ttfp_s
        for o in successes
        if getattr(o, "audio_ttfp_s", None) is not None
    ]
    text_ttfts = [
        o.text_ttft_s for o in successes if getattr(o, "text_ttft_s", None) is not None
    ]
    inter_chunk_deltas = [
        d for o in successes for d in getattr(o, "inter_chunk_s", []) or []
    ]

    if wall_clock_s is not None and wall_clock_s > 0:
        throughput = round(len(successes) / wall_clock_s, 3)
    else:
        total_latency = sum(latencies)
        throughput = (
            round(len(successes) / total_latency, 3) if total_latency > 0 else 0
        )

    metrics_summary: dict = {
        "total_requests": len(outputs),
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
        "rtf_p95": round(float(np.percentile(rtfs, 95)), 4) if rtfs else None,
        "rtf_p99": round(float(np.percentile(rtfs, 99)), 4) if rtfs else None,
        "throughput_qps": throughput,
        **_compute_token_metrics(successes, wall_clock_s=wall_clock_s),
    }
    if audio_durations and wall_clock_s is not None and wall_clock_s > 0:
        total_audio_s = sum(audio_durations)
        metrics_summary["audio_throughput_s_per_s"] = round(
            total_audio_s / wall_clock_s, 3
        )
    if ttfps:
        metrics_summary["audio_ttfp_mean_s"] = round(float(np.mean(ttfps)), 4)
        metrics_summary["audio_ttfp_median_s"] = round(float(np.median(ttfps)), 4)
        metrics_summary["audio_ttfp_p95_s"] = round(float(np.percentile(ttfps, 95)), 4)
        metrics_summary["audio_ttfp_p99_s"] = round(float(np.percentile(ttfps, 99)), 4)
    if text_ttfts:
        metrics_summary["text_ttft_mean_s"] = round(float(np.mean(text_ttfts)), 4)
        metrics_summary["text_ttft_median_s"] = round(float(np.median(text_ttfts)), 4)
        metrics_summary["text_ttft_p95_s"] = round(
            float(np.percentile(text_ttfts, 95)), 4
        )
        metrics_summary["text_ttft_p99_s"] = round(
            float(np.percentile(text_ttfts, 99)), 4
        )
    if inter_chunk_deltas:
        metrics_summary["inter_chunk_mean_s"] = round(
            float(np.mean(inter_chunk_deltas)), 4
        )
        metrics_summary["inter_chunk_p95_s"] = round(
            float(np.percentile(inter_chunk_deltas, 95)), 4
        )
        metrics_summary["inter_chunk_p99_s"] = round(
            float(np.percentile(inter_chunk_deltas, 99)), 4
        )
    return metrics_summary


def print_speed_summary(
    metrics: dict,
    model_name: str,
    concurrency: int | None = None,
    title: str = "Speed Benchmark Result",
) -> None:
    lw = SPEED_LABEL_WIDTH
    w = SPEED_LINE_WIDTH
    print(f"\n{'=' * w}")
    print(f"{title:^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<{lw}} {model_name}")
    if concurrency is not None:
        print(f"  {'Concurrency:':<{lw}} {concurrency}")
    print(f"  {'Completed requests:':<{lw}} {metrics['completed_requests']}")
    print(f"  {'Failed requests:':<{lw}} {metrics['failed_requests']}")
    print(f"{'-' * w}")
    print(f"  {'Latency mean (s):':<{lw}} {metrics.get('latency_mean_s', 'N/A')}")
    print(f"  {'Latency median (s):':<{lw}} {metrics.get('latency_median_s', 'N/A')}")
    print(f"  {'Latency p95 (s):':<{lw}} {metrics.get('latency_p95_s', 'N/A')}")
    print(f"  {'Latency p99 (s):':<{lw}} {metrics.get('latency_p99_s', 'N/A')}")
    if metrics.get("rtf_mean") is not None:
        print(f"  {'RTF mean:':<{lw}} {metrics['rtf_mean']}")
        print(f"  {'RTF median:':<{lw}} {metrics['rtf_median']}")
    if metrics.get("rtf_p95") is not None:
        print(f"  {'RTF p95:':<{lw}} {metrics['rtf_p95']}")
        print(f"  {'RTF p99:':<{lw}} {metrics['rtf_p99']}")
    if metrics.get("audio_duration_mean_s"):
        print(
            f"  {'Audio duration mean (s):':<{lw}} {metrics['audio_duration_mean_s']}"
        )
    if metrics.get("audio_throughput_s_per_s") is not None:
        print(
            f"  {'Audio throughput (s/s):':<{lw}} "
            f"{metrics['audio_throughput_s_per_s']}"
        )
    if metrics.get("audio_ttfp_mean_s") is not None:
        print(f"  {'TTFC mean (s):':<{lw}} {metrics['audio_ttfp_mean_s']}")
        print(f"  {'TTFC median (s):':<{lw}} {metrics['audio_ttfp_median_s']}")
        print(f"  {'TTFC p95 (s):':<{lw}} {metrics['audio_ttfp_p95_s']}")
        print(f"  {'TTFC p99 (s):':<{lw}} {metrics['audio_ttfp_p99_s']}")
    if metrics.get("text_ttft_mean_s") is not None:
        print(f"  {'TTFT mean (s):':<{lw}} {metrics['text_ttft_mean_s']}")
        print(f"  {'TTFT median (s):':<{lw}} {metrics['text_ttft_median_s']}")
        print(f"  {'TTFT p95 (s):':<{lw}} {metrics['text_ttft_p95_s']}")
        print(f"  {'TTFT p99 (s):':<{lw}} {metrics['text_ttft_p99_s']}")
    if metrics.get("inter_chunk_mean_s") is not None:
        print(f"  {'ITL mean (s):':<{lw}} {metrics['inter_chunk_mean_s']}")
        print(f"  {'ITL p95 (s):':<{lw}} {metrics['inter_chunk_p95_s']}")
        print(f"  {'ITL p99 (s):':<{lw}} {metrics['inter_chunk_p99_s']}")
    if metrics.get("output_throughput") is not None:
        print(f"  {'Output throughput (tok/s):':<{lw}} {metrics['output_throughput']}")
    if metrics.get("output_tok_per_req_s") is not None:
        print(
            f"  {'Output tokens/request-s:':<{lw}} "
            f"{metrics['output_tok_per_req_s']}"
        )
    if metrics.get("output_tokens_mean") is not None:
        print(f"  {'Output tokens (mean):':<{lw}} {metrics['output_tokens_mean']:.0f}")
        print(f"  {'Output tokens (total):':<{lw}} {metrics['output_tokens_total']}")
    if metrics.get("prompt_tokens_mean") is not None:
        print(f"  {'Prompt tokens (mean):':<{lw}} {metrics['prompt_tokens_mean']:.0f}")
        print(f"  {'Prompt tokens (total):':<{lw}} {metrics['prompt_tokens_total']}")
    print(f"  {'Throughput (req/s):':<{lw}} {metrics.get('throughput_qps', 'N/A')}")
    print(f"{'=' * w}")


def build_speed_results(
    outputs: list[RequestResult],
    metrics: dict,
    config: dict,
) -> dict:
    return {
        "summary": metrics,
        "config": config,
        "per_request": [_request_result_to_dict(output) for output in outputs],
    }


def _request_result_to_dict(output: RequestResult) -> dict:
    inter = getattr(output, "inter_chunk_s", None) or None
    ttfp = getattr(output, "audio_ttfp_s", None)
    ttft = getattr(output, "text_ttft_s", None)
    return {
        "id": output.request_id,
        "text": output.text,
        "is_success": output.is_success,
        "latency_s": round(output.latency_s, 4),
        "audio_duration_s": round(output.audio_duration_s, 4),
        "rtf": round(output.rtf, 4) if output.rtf < float("inf") else None,
        "prompt_tokens": output.prompt_tokens or None,
        "completion_tokens": output.completion_tokens or None,
        "output_token_rate": (
            round(output.tok_per_s, 1) if output.tok_per_s > 0 else None
        ),
        "wav_path": output.wav_path or None,
        "error": output.error or None,
        "audio_ttfp_s": round(ttfp, 4) if ttfp is not None else None,
        "text_ttft_s": round(ttft, 4) if ttft is not None else None,
        "inter_chunk_s": [round(d, 4) for d in inter] if inter else None,
    }
