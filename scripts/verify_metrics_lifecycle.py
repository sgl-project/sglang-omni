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
from prometheus_client.samples import Sample


def check_histogram_delta(
    before: list[Sample], after: list[Sample], name: str, expected_count: int
) -> dict[str, float]:
    deltas = {}
    for suffix in ("_count", "_sum", "_bucket"):
        current = [sample for sample in after if sample.name == name + suffix]
        assert current, f"Missing {name + suffix}"
        for sample in current:
            assert math.isfinite(sample.value) and sample.value >= 0, name
            baseline = sum(
                previous.value
                for previous in before
                if previous.name == sample.name and previous.labels == sample.labels
            )
            delta = sample.value - baseline
            assert delta >= 0, f"Counter reset for {name}"
            if delta > 0 and name in (
                "sglang:queue_time_seconds",
                "sglang_omni:scheduler_batch_size",
                "sglang_omni:stage_compute_duration_seconds",
                "sglang_omni:stage_request_duration_seconds",
            ):
                assert (
                    sample.labels.get("stage") == "generate"
                ), f"Unexpected stage for {name}: {sample.labels}"
            else:
                pass
            if suffix == "_bucket":
                bound = sample.labels["le"]
                deltas[bound] = deltas.get(bound, 0) + delta
            else:
                deltas[suffix] = deltas.get(suffix, 0) + delta
    assert deltas["_count"] == expected_count, f"{name}: {deltas}"
    assert deltas["+Inf"] == expected_count, f"{name}: +Inf/count mismatch"
    buckets = [
        value
        for bound, value in sorted(
            (
                (float(bound), value)
                for bound, value in deltas.items()
                if bound not in ("_count", "_sum")
            ),
        )
    ]
    assert buckets == sorted(buckets), f"{name}: non-monotonic bucket deltas"
    return {"count_delta": deltas["_count"], "sum_delta": deltas["_sum"]}


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
    parser.add_argument("--all-sensenova-metrics", action="store_true")
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
    sample_snapshots: list[list[Sample]] = []

    def scrape() -> dict[str, float]:
        with urllib.request.urlopen(
            args.base_url.rstrip("/") + "/metrics", timeout=30
        ) as response:
            raw = response.read().decode()
        (out / f"snapshot-{len(snapshots):04d}.prom").write_text(raw, encoding="utf-8")
        values = {}
        selected = []
        for family in text_string_to_metric_families(raw):
            for sample in family.samples:
                if sample.labels.get("model_name") == label or sample.name.startswith(
                    "sglang_omni:http_"
                ):
                    selected.append(sample)
                    values[sample.name] = values.get(sample.name, 0) + sample.value
                else:
                    pass
        sample_snapshots.append(selected)
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
            "sglang_omni:failed_requests_total",
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
        if args.all_sensenova_metrics:
            histogram_checks = {}
            for name in (
                "sglang:prompt_tokens_total",
                "sglang:generation_tokens_total",
                "sglang:time_to_first_token_seconds_count",
                "sglang:inter_token_latency_seconds_count",
                "sglang:prompt_tokens_histogram_count",
                "sglang:uncached_prompt_tokens_histogram_count",
                "sglang:generation_tokens_histogram_count",
                "sglang:cached_tokens_total",
                "sglang_omni:audio_output_seconds_total",
            ):
                assert after.get(name, 0) == before.get(
                    name, 0
                ), f"Images request incorrectly updated {name}"
            for name in (
                "sglang:e2e_request_latency_seconds",
                "sglang:queue_time_seconds",
                "sglang_omni:scheduler_batch_size",
                "sglang_omni:stage_compute_duration_seconds",
                "sglang_omni:stage_request_duration_seconds",
            ):
                histogram_checks[name] = check_histogram_delta(
                    sample_snapshots[0], sample_snapshots[-1], name, expected
                )
            assert (
                histogram_checks["sglang:queue_time_seconds"]["sum_delta"] > 0
            ), "No positive queue duration; increase concurrency/steps"
            assert (
                histogram_checks["sglang_omni:scheduler_batch_size"]["sum_delta"]
                == expected
            ), "This n=1 test requires scheduler max_batch_size=1"
            for name in (
                "sglang:e2e_request_latency_seconds",
                "sglang_omni:stage_compute_duration_seconds",
                "sglang_omni:stage_request_duration_seconds",
            ):
                assert histogram_checks[name]["sum_delta"] > 0, name
            histogram_checks["sglang_omni:http_request_duration_seconds"] = (
                check_histogram_delta(
                    sample_snapshots[0],
                    sample_snapshots[-1],
                    "sglang_omni:http_request_duration_seconds",
                    expected + 1,
                )
            )

            def http_count(samples: list[Sample], route: str, status: str) -> float:
                return sum(
                    sample.value
                    for sample in samples
                    if sample.name == "sglang_omni:http_requests_total"
                    and sample.labels.get("method") == "POST"
                    and sample.labels.get("route") == route
                    and sample.labels.get("status_code") == status
                    and sample.labels.get("outcome") == "completed"
                )

            for route, status, count in (
                ("/v1/images/generations", "200", args.concurrency),
                ("/v1/images/edits", "200", args.concurrency if args.image else 0),
                ("/v1/images/generations", str(summary["rejected_http_status"]), 1),
            ):
                delta = http_count(sample_snapshots[-1], route, status) - http_count(
                    sample_snapshots[0], route, status
                )
                assert (
                    delta == count
                ), f"HTTP {route}/{status}: expected {count}, got {delta}"
            assert (
                before.get("sglang_omni:http_requests_inflight") == 0
            ), "HTTP inflight missing or service not idle"
            assert (
                after.get("sglang_omni:http_requests_inflight") == 0
            ), "HTTP inflight not idle"
            assert (
                max(
                    snapshot.get("sglang_omni:http_requests_inflight", 0)
                    for snapshot in snapshots
                )
                > 0
            ), "No HTTP inflight transition captured"
            summary["histogram_checks"] = histogram_checks
            summary["sensenova_applicable_metrics_passed"] = 12
            summary["scope"] = (
                "SenseNova single-stage, max_batch_size=1 success/queue/HTTP rejection"
            )
        else:
            pass
        summary["passed"] = True
    except Exception as error:
        summary["error"] = str(error)
    (out / "samples.json").write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
