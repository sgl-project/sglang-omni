# SPDX-License-Identifier: Apache-2.0
"""Run SenseNova-applicable metrics acceptance and report unsupported coverage."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

APPLICABLE_METRICS = (
    "sglang:e2e_request_latency_seconds",
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:num_requests_total",
    "sglang:queue_time_seconds",
    "sglang_omni:generated_images_total",
    "sglang_omni:http_requests_total",
    "sglang_omni:http_request_duration_seconds",
    "sglang_omni:http_requests_inflight",
    "sglang_omni:stage_request_duration_seconds",
    "sglang_omni:stage_compute_duration_seconds",
    "sglang_omni:scheduler_batch_size",
)
AR_METRICS = (
    "sglang:time_to_first_token_seconds",
    "sglang:inter_token_latency_seconds",
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    "sglang:prompt_tokens_histogram",
    "sglang:uncached_prompt_tokens_histogram",
    "sglang:generation_tokens_histogram",
    "sglang:cached_tokens_total",
    "sglang:cache_hit_rate",
    "sglang:gen_throughput",
    "sglang:token_usage",
    "sglang:full_token_usage",
    "sglang:num_used_tokens",
    "sglang:kv_available_tokens",
    "sglang:kv_evictable_tokens",
    "sglang:kv_used_tokens",
    "sglang:max_total_num_tokens",
    "sglang:eviction_duration_seconds",
    "sglang:evicted_tokens_total",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", default="512x512")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--skip-unit", action="store_true")
    options = parser.parse_args()
    if options.steps <= 0 or options.concurrency < 2 or options.timeout <= 0:
        parser.error("steps/timeout must be positive; concurrency must be at least 2")
    else:
        pass
    output = options.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    summary = {
        "passed": False,
        "scope": "SenseNova Images on an idle service with scheduler max_batch_size=1",
        "all_implemented_metrics_validated": False,
        "catalog_scope": "Current P0/P1 and retained Omni metrics; conditional P2 families excluded",
        "unit_regression": "SKIPPED" if options.skip_unit else "NOT_RUN",
        "metrics": [{"name": name, "status": "NOT_RUN"} for name in APPLICABLE_METRICS]
        + [
            {
                "name": name,
                "status": "NOT_APPLICABLE",
                "reason": "Requires an AR engine",
            }
            for name in AR_METRICS
        ]
        + [
            {
                "name": "sglang_omni:audio_output_seconds_total",
                "status": "NOT_APPLICABLE",
                "reason": "Requires waveform output",
            }
        ],
        "not_covered": [
            "backend failure/abort recovery",
            "worker restart",
            "monitoring overhead",
            "TP rank aggregation",
            "multiple stages/scheduler types",
            "dynamic request batching",
            "AR/Audio metrics",
            "platform PromQL mapping",
        ],
    }
    summary["metrics"] += [
        {
            "name": name,
            "status": "NOT_COVERED",
            "reason": "Negative checks only; no backend fault/cancellation injection",
        }
        for name in (
            "sglang:num_aborted_requests_total",
            "sglang_omni:failed_requests_total",
        )
    ]
    for operation, command in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "branch", "--show-current"]),
        ("working-tree", ["git", "status", "--short"]),
    ):
        try:
            result = subprocess.run(
                command, cwd=scripts.parent, capture_output=True, text=True
            )
            (output / f"{operation}.txt").write_text(
                result.stdout if result.returncode == 0 else result.stderr,
                encoding="utf-8",
            )
        except FileNotFoundError:
            (output / f"{operation}.txt").write_text(
                "Git unavailable; copied source", encoding="utf-8"
            )
    common = [
        "--base-url",
        options.base_url,
        "--model",
        options.model,
        "--size",
        options.size,
        "--timeout",
        str(options.timeout),
    ]

    def run_case(case: str, script: str, arguments: list[str]) -> None:
        command = [
            sys.executable,
            str(scripts / script),
            *arguments,
            "--out-dir",
            str(output / case),
        ]
        with (output / f"{case}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        print(f"{case}: exit_code={result.returncode}", flush=True)
        if result.returncode != 0:
            if case == "unit":
                summary["unit_regression"] = "FAIL"
            else:
                pass
            raise RuntimeError(f"{case} failed; see {output / (case + '.log')}")
        else:
            pass

    try:
        if not options.skip_unit:
            run_case(
                "unit",
                "verify_metrics.py",
                ["--repo", str(scripts.parent), "--mode", "full"],
            )
            summary["unit_regression"] = "PASS"
        else:
            pass
        run_case(
            "smoke",
            "verify_model_metrics.py",
            ["--api", "images", *common, "--steps", "10", "--n", "1"],
        )
        reference = output / "smoke/responses/t2i-0.png"
        run_case(
            "formal",
            "verify_model_metrics.py",
            [
                "--api",
                "images",
                *common,
                "--steps",
                str(options.steps),
                "--n",
                "2",
                "--image",
                str(reference),
            ],
        )
        run_case(
            "lifecycle",
            "verify_metrics_lifecycle.py",
            [
                *common,
                "--steps",
                str(options.steps),
                "--concurrency",
                str(options.concurrency),
                "--image",
                str(reference),
                "--require-queue",
                "--all-sensenova-metrics",
            ],
        )
        for metric in summary["metrics"][: len(APPLICABLE_METRICS)]:
            metric["status"] = "PASS"
        summary["passed"] = True
    except (RuntimeError, OSError) as error:
        summary["error"] = str(error)
    summary["counts"] = {
        status: sum(metric["status"] == status for metric in summary["metrics"])
        for status in ("PASS", "NOT_RUN", "NOT_APPLICABLE", "NOT_COVERED")
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
