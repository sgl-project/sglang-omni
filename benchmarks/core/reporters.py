from __future__ import annotations

import csv
import json
import os

from .types import BenchmarkResults


def print_summary(results: BenchmarkResults) -> None:
    summary = results.summary
    line_width = 68
    print(f"\n{'=' * line_width}")
    print(f"{'Benchmark Result':^{line_width}}")
    print(f"{'=' * line_width}")
    print(f"  {'Model profile:':<28} {summary.model_profile}")
    print(f"  {'Case:':<28} {summary.case}")
    print(f"  {'Scenario:':<28} {summary.scenario}")
    print(f"  {'Request family:':<28} {summary.request_family}")
    print(f"  {'Completed requests:':<28} {summary.completed_requests}")
    print(f"  {'Failed requests:':<28} {summary.failed_requests}")
    print(f"  {'Run wall time (s):':<28} {summary.to_dict()['run_wall_time_s']}")
    print("-" * line_width)
    if summary.latency_mean_s is not None:
        print(f"  {'Latency mean (s):':<28} {summary.to_dict()['latency_mean_s']}")
        print(
            f"  {'Latency median (s):':<28} {summary.to_dict()['latency_median_s']}"
        )
        print(f"  {'Latency p95 (s):':<28} {summary.to_dict()['latency_p95_s']}")
        print(f"  {'Latency p99 (s):':<28} {summary.to_dict()['latency_p99_s']}")
    if summary.throughput_qps is not None:
        print(f"  {'Throughput (req/s):':<28} {summary.to_dict()['throughput_qps']}")
    if summary.audio_duration_mean_s is not None:
        print(
            f"  {'Audio duration mean (s):':<28} "
            f"{summary.to_dict()['audio_duration_mean_s']}"
        )
    if summary.rtf_mean is not None:
        print(f"  {'RTF mean:':<28} {summary.to_dict()['rtf_mean']}")
        print(f"  {'RTF median:':<28} {summary.to_dict()['rtf_median']}")
    if summary.tok_per_s_mean is not None:
        print(
            f"  {'Tok/s (per-request mean):':<28} "
            f"{summary.to_dict()['tok_per_s_mean']}"
        )
    if summary.tok_per_s_agg is not None:
        print(
            f"  {'Tok/s (aggregate):':<28} {summary.to_dict()['tok_per_s_agg']}"
        )
    print(f"{'=' * line_width}")


def save_results(results: BenchmarkResults, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "benchmark_results.json")
    csv_path = os.path.join(output_dir, "per_request.csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results.to_dict(), handle, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "input_preview",
                "success",
                "latency_s",
                "first_audio_latency_s",
                "audio_duration_s",
                "rtf",
                "prompt_tokens",
                "completion_tokens",
                "engine_time_s",
                "tok_per_s",
                "error",
            ],
        )
        writer.writeheader()
        for result in results.per_request:
            writer.writerow(result.to_dict())

