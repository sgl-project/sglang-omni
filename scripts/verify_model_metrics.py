#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check live metrics snapshots/deltas around AR Chat or SenseNova Images requests."""

import argparse
import json
import math
import subprocess
import sys
import urllib.request
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--metrics-model", help="model_name label, if different from API model"
    )
    parser.add_argument("--api", choices=("chat", "images"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", default="512x512")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.api == "chat" and args.image:
        parser.error("--image is only applicable to Images")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    label = args.metrics_model or args.model
    summary = {
        "api": args.api,
        "metrics_model": label,
        "passed": False,
        "scope": "Live snapshot/finished-request deltas, not exclusive-load ground truth",
        "not_covered": [
            "queue/running transitions under load",
            "abort/failure recovery",
            "worker restart",
            "monitoring overhead",
            "TP rank aggregation",
        ],
    }

    def scrape(name):
        with urllib.request.urlopen(
            args.base_url.rstrip("/") + "/metrics", timeout=30
        ) as response:
            raw = response.read().decode("utf-8")
        (out / f"{name}.prom").write_text(raw, encoding="utf-8")
        return [
            sample
            for family in text_string_to_metric_families(raw)
            for sample in family.samples
            if sample.labels.get("model_name") == label
        ]

    def values(samples, name):
        return [sample.value for sample in samples if sample.name == name]

    try:
        before = scrape("before")
        for name in ("sglang:num_running_reqs", "sglang:num_queue_reqs"):
            samples = values(before, name)
            assert (
                samples
            ), f"Missing {name} for model_name={label}; check --metrics-model"
            assert all(math.isfinite(value) and value >= 0 for value in samples), name
        script = (
            "verify_model_response_usage.py"
            if args.api == "chat"
            else "verify_sensenova_images.py"
        )
        command = [
            sys.executable,
            str(Path(__file__).with_name(script)),
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--out-dir",
            str(out / "responses"),
            "--timeout",
            str(args.timeout),
        ]
        if args.api == "images":
            command += [
                "--size",
                args.size,
                "--steps",
                str(args.steps),
                "--n",
                str(args.n),
            ]
            if args.image:
                command += ["--image", str(args.image)]
        with (out / "requests.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, text=True
            )
        summary["request_exit_code"] = result.returncode
        after = scrape("after")
        summary["deltas"] = {}
        names = [
            "sglang:e2e_request_latency_seconds_count",
            "sglang:num_requests_total",
        ]
        if args.api == "chat":
            names += [
                "sglang:time_to_first_token_seconds_count",
                "sglang:inter_token_latency_seconds_count",
                "sglang:prompt_tokens_total",
                "sglang:generation_tokens_total",
            ]
        else:
            names += ["sglang_omni:generated_images_total"]
            summary["not_covered"] += ["text TTFT/ITL/token metrics", "Chat usage"]
        assert result.returncode == 0, "Response checks failed; see requests.log"
        for name in names:
            observed = values(after, name)
            assert observed, f"Missing metric {name}"
            delta = sum(observed) - sum(values(before, name))
            summary["deltas"][name] = delta
            assert math.isfinite(delta) and delta > 0, f"No positive delta for {name}"
        for name in ("sglang:num_running_reqs", "sglang:num_queue_reqs"):
            observed = values(after, name)
            assert observed and all(
                math.isfinite(value) and value >= 0 for value in observed
            ), name
        summary["passed"] = True
    except Exception as error:
        summary["error"] = str(error)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
