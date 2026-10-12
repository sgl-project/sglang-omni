# SPDX-License-Identifier: Apache-2.0
"""Validate Images metric transitions on an otherwise idle, dedicated service."""

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics-model")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--size", default="512x512")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--require-queue", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 2 or args.timeout <= 0 or args.poll_seconds <= 0:
        parser.error(
            "concurrency must be at least 2; timeout/poll-seconds must be positive"
        )
    else:
        pass
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    label = args.metrics_model or args.model
    summary = {
        "passed": False,
        "scope": "Exclusive-load Images concurrency and rejected-input recovery",
        "not_covered": [
            "backend failure/abort recovery",
            "worker restart",
            "monitoring overhead",
            "text TTFT/ITL/token metrics",
            "TP rank aggregation",
        ],
    }
    snapshots = []

    def scrape() -> dict[str, float]:
        with urllib.request.urlopen(
            args.base_url.rstrip("/") + "/metrics", timeout=30
        ) as response:
            raw = response.read().decode()
        (out / f"snapshot-{len(snapshots):04d}.prom").write_text(raw, encoding="utf-8")
        values = {}
        for family in text_string_to_metric_families(raw):
            for sample in family.samples:
                if sample.labels.get("model_name") == label:
                    values[sample.name] = values.get(sample.name, 0) + sample.value
                else:
                    pass
        for name in ("sglang:num_running_reqs", "sglang:num_queue_reqs"):
            assert (
                name in values and math.isfinite(values[name]) and values[name] >= 0
            ), name
        snapshots.append({"elapsed_seconds": time.monotonic() - started, **values})
        return values

    def request(index: int) -> int:
        command = [
            sys.executable,
            str(Path(__file__).with_name("verify_model_metrics.py")),
            "--api",
            "images",
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--metrics-model",
            label,
            "--size",
            args.size,
            "--steps",
            str(args.steps),
            "--n",
            "1",
            "--timeout",
            str(args.timeout),
            "--out-dir",
            str(out / f"request-{index}"),
        ]
        if args.image:
            command += ["--image", str(args.image)]
        else:
            pass
        with (out / f"request-{index}.log").open("w", encoding="utf-8") as log:
            return subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT
            ).returncode

    started = time.monotonic()
    try:
        before = scrape()
        assert (
            before["sglang:num_running_reqs"] == before["sglang:num_queue_reqs"] == 0
        ), "Service must be idle and must not receive other clients during this test"
        invalid = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/images/generations",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(invalid, timeout=30) as response:
                raise AssertionError(
                    f"Invalid request unexpectedly returned {response.status}"
                )
        except urllib.error.HTTPError as error:
            assert error.code in (
                400,
                422,
            ), f"Expected input rejection, got {error.code}"
            summary["rejected_http_status"] = error.code
        after_rejection = scrape()
        counters = (
            "sglang:num_requests_total",
            "sglang:e2e_request_latency_seconds_count",
            "sglang_omni:generated_images_total",
            "sglang:num_failed_requests_total",
            "sglang:num_aborted_requests_total",
        )
        assert all(
            after_rejection.get(name, 0) == before.get(name, 0) for name in counters
        ), "Rejected input changed pipeline counters"
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(request, index) for index in range(args.concurrency)
            ]
            while not all(future.done() for future in futures):
                scrape()
                time.sleep(args.poll_seconds)
            codes = [future.result() for future in futures]
        summary["request_exit_codes"] = codes
        assert all(
            code == 0 for code in codes
        ), "Response/stage checks failed; see request logs"
        deadline = time.monotonic() + 30
        while True:
            after = scrape()
            if after["sglang:num_running_reqs"] == after["sglang:num_queue_reqs"] == 0:
                break
            elif time.monotonic() >= deadline:
                raise AssertionError(
                    "Scheduler did not return to idle within 30 seconds"
                )
            else:
                time.sleep(args.poll_seconds)
        summary["max_running"] = max(
            item["sglang:num_running_reqs"] for item in snapshots
        )
        summary["max_queue"] = max(item["sglang:num_queue_reqs"] for item in snapshots)
        assert (
            summary["max_running"] > 0
        ), "No running transition captured; increase --steps"
        summary["queue_transition_observed"] = summary["max_queue"] > 0
        if args.require_queue:
            assert summary[
                "queue_transition_observed"
            ], "No queue transition; increase concurrency"
        else:
            if not summary["queue_transition_observed"]:
                summary["not_covered"].append("queue transition was not observed")
            else:
                pass
        expected = args.concurrency * (2 if args.image else 1)
        summary["deltas"] = {}
        for name in counters[:3]:
            delta = after.get(name, 0) - before.get(name, 0)
            summary["deltas"][name] = delta
            assert delta == expected, f"{name}: expected {expected}, got {delta}"
        for name in counters[3:]:
            assert after.get(name, 0) == before.get(name, 0), f"Unexpected {name}"
        summary["passed"] = True
    except Exception as error:
        summary["error"] = str(error)
    (out / "samples.json").write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
