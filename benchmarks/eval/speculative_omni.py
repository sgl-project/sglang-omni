# SPDX-License-Identifier: Apache-2.0
"""Run Omni clients through the shared speculative sweep."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, FiniteFloat

from benchmarks.eval.speculative_workloads import (
    HOST,
    LoadSettings,
    OmniWorkload,
    run_client,
)
from benchmarks.metrics.speculative import LoadMeasurement, mean_or_none
from benchmarks.runtime_metrics import ResourceMonitor


class OmniAccuracy(BaseModel):
    total_samples: int
    correct: int


class OmniSpeed(BaseModel):
    total_requests: int
    completed_requests: int
    failed_requests: int
    latency_p95_s: float | None = None
    text_ttft_p95_s: float | None = None


class OmniTiming(BaseModel):
    wall_clock_s: FiniteFloat


class OmniResults(BaseModel):
    summary: OmniAccuracy
    config: OmniTiming
    speed: OmniSpeed | None = None
    speed_metrics: OmniSpeed | None = None


def measure_omni(
    workload: OmniWorkload,
    settings: LoadSettings,
    *,
    port: int,
    model_path: str,
    leg_dir: Path,
) -> list[LoadMeasurement]:
    loads: list[LoadMeasurement] = []
    for concurrency in settings.concurrencies:
        speeds: list[OmniSpeed] = []
        throughputs_requests_per_s: list[float] = []
        accuracies: list[float] = []
        memory_peaks: list[float] = []
        for repeat in range(1, settings.repeats + 1):
            run_dir = leg_dir / f"c{concurrency}_r{repeat}"
            monitor = ResourceMonitor().start()
            try:
                run_client(
                    [
                        workload.client,
                        "--host",
                        HOST,
                        "--port",
                        str(port),
                        "--model",
                        model_path,
                        "--max-concurrency",
                        str(concurrency),
                        "--warmup",
                        str(concurrency if settings.warmup else 0),
                        "--output-dir",
                        str(run_dir),
                        "--disable-tqdm",
                        "--stream",
                        *(
                            ["--max-samples", str(settings.max_samples)]
                            if settings.max_samples is not None
                            else []
                        ),
                        *settings.client_arguments,
                    ],
                    run_dir.with_suffix(".log"),
                    check=True,
                )
            finally:
                monitor.stop()
            if monitor.samples:
                memory_peaks.append(
                    max(sample.gpu_memory_used_mib for sample in monitor.samples)
                )
            else:
                pass
            (run_dir / "resources.json").write_text(
                json.dumps(
                    {
                        "samples": [asdict(sample) for sample in monitor.samples],
                        "error": monitor.error,
                    },
                    indent=2,
                )
                + "\n"
            )
            result = OmniResults.model_validate_json(
                (run_dir / workload.results_file).read_text()
            )
            speed = (
                result.speed
                if workload.speed_field == "speed"
                else result.speed_metrics
            )
            if speed is None:
                raise ValueError(f"Missing {workload.speed_field} in {run_dir}")
            elif speed.total_requests != result.summary.total_samples:
                raise ValueError(
                    f"Accuracy and performance cohorts differ in {run_dir}"
                )
            else:
                speeds.append(speed)
            # note (daii-0818): the client rounds QPS for display, so use the raw elapsed time for the sweep.
            if result.config.wall_clock_s > 0:
                throughputs_requests_per_s.append(
                    speed.completed_requests / result.config.wall_clock_s
                )
            else:
                pass
            if result.summary.total_samples > 0:
                accuracies.append(result.summary.correct / result.summary.total_samples)
            else:
                pass
        accuracy = mean_or_none(accuracies)
        loads.append(
            LoadMeasurement(
                concurrency=concurrency,
                total_requests=sum(speed.total_requests for speed in speeds),
                failed_requests=sum(speed.failed_requests for speed in speeds),
                throughput_requests_per_s=mean_or_none(throughputs_requests_per_s),
                latency_p95_s=mean_or_none(
                    [
                        speed.latency_p95_s
                        for speed in speeds
                        if speed.latency_p95_s is not None
                    ]
                ),
                first_output_latency_p95_s=mean_or_none(
                    [
                        speed.text_ttft_p95_s
                        for speed in speeds
                        if speed.text_ttft_p95_s is not None
                    ]
                ),
                accuracy={"Accuracy": accuracy} if accuracy is not None else {},
                gpu_memory_peak_mib=max(memory_peaks, default=None),
                incomplete_reasons=(["No scored samples"] if accuracy is None else [])
                + (
                    ["Missing positive measurement duration"]
                    if len(throughputs_requests_per_s) != len(speeds)
                    else []
                )
                + (
                    ["Missing streamed first-output latency"]
                    if any(speed.text_ttft_p95_s is None for speed in speeds)
                    else []
                ),
            )
        )
    return loads
