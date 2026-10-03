# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for trace attribution, GPU intervals, and missing measurements."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from benchmarks.eval.personaplex_profiling_report import (
    ChromeTrace,
    TraceArguments,
    TraceEvent,
    build_component_report,
)


def trace_event(
    category: str,
    start_microseconds: float,
    duration_microseconds: float,
    name: str = "",
    *,
    external_id: int | None = None,
    correlation: int | None = None,
) -> TraceEvent:
    is_gpu_activity = category == "kernel" or category.startswith("gpu_")
    name = name or ("cudaLaunchKernel" if category == "cuda_runtime" else "operation")
    return TraceEvent(
        ph="X",
        cat=category,
        name=name,
        pid=0 if is_gpu_activity else 101,
        tid=999 if category == "cuda_runtime" else 7,
        ts=start_microseconds,
        dur=duration_microseconds,
        args=TraceArguments(
            **{
                "External id": external_id,
                "correlation": correlation,
                "device": 0 if is_gpu_activity else None,
            }
        ),
    )


def write_trace(trace_path: Path, events: list[TraceEvent]) -> Path:
    serialized_trace = ChromeTrace(traceEvents=events).model_dump_json(
        by_alias=True, exclude_none=True
    )
    if trace_path.suffix == ".gz":
        with gzip.open(trace_path, "wt", encoding="utf-8") as trace_file:
            trace_file.write(serialized_trace)
    else:
        trace_path.write_text(serialized_trace)
    return trace_path


def test_worker_attribution_and_physical_gpu_overlap(tmp_path: Path) -> None:
    first_trace = write_trace(
        tmp_path / "lm.json",
        [
            trace_event("user_annotation", 0, 5000, "personaplex.temporal_transformer"),
            trace_event("user_annotation", 1000, 2000, "personaplex.depformer"),
            trace_event("cpu_op", 100, 800, external_id=10),
            trace_event("cpu_op", 1500, 400, external_id=20),
            trace_event("cuda_runtime", 300, 100, external_id=10, correlation=1),
            trace_event("cuda_runtime", 1600, 200, external_id=20, correlation=2),
            trace_event("kernel", 400, 2000, external_id=10, correlation=1),
            trace_event("kernel", 2000, 600, external_id=10, correlation=1),
            trace_event("kernel", 1800, 2000, external_id=20, correlation=2),
            trace_event(
                "gpu_user_annotation", 0, 100000, "personaplex.temporal_transformer"
            ),
            trace_event("cuda_runtime", 6000, 10000, "cudaDeviceSynchronize"),
        ],
    )
    second_trace = write_trace(
        tmp_path / "encoder.json.gz",
        [
            trace_event("user_annotation", 2000, 2500, "personaplex.mimi_encode"),
            trace_event("cpu_op", 2100, 400, external_id=10),
            trace_event("cuda_runtime", 2200, 100, external_id=10, correlation=1),
            trace_event("kernel", 2400, 2000, external_id=10, correlation=1),
        ],
    )
    report = build_component_report([first_trace, second_trace])
    components = {component["name"]: component for component in report["components"]}
    temporal = components["personaplex.temporal_transformer"]
    assert temporal["cpu_scope_milliseconds"] == 5.0
    assert temporal["scope_count"] == 1
    assert temporal["kernel_count"] == 2
    assert temporal["gpu_kernel_milliseconds"] == 2.2
    assert temporal["cuda_launch_milliseconds"] == 0.1
    assert temporal["cuda_synchronization_milliseconds"] == 0.0
    assert components["personaplex.depformer"]["gpu_kernel_milliseconds"] == 2.0
    assert components["personaplex.mimi_encode"]["gpu_kernel_milliseconds"] == 2.0
    assert report["cpu_scope_busy_milliseconds"] == 7.5
    assert report["gpu_kernel_milliseconds"] == 4.0
    assert report["kernel_attribution_fraction"] == 1.0
    assert len(report["devices"]) == 1
    assert report["devices"][0]["gpu_busy_milliseconds"] == 4.0
    assert report["devices"][0]["gpu_idle_milliseconds"] == 1.0


def test_memory_costs_and_gpu_busy_intervals(tmp_path: Path) -> None:
    trace_path = write_trace(
        tmp_path / "memory.json",
        [
            trace_event("user_annotation", 0, 2000, "personaplex.h2d"),
            trace_event("cpu_op", 100, 1300, external_id=1),
            trace_event(
                "cuda_runtime",
                200,
                100,
                "cudaMemcpyAsync",
                external_id=1,
                correlation=3,
            ),
            trace_event("gpu_memcpy", 300, 200, correlation=3),
            trace_event(
                "cuda_runtime",
                100,
                100,
                "cudaMemsetAsync",
                external_id=1,
                correlation=2,
            ),
            trace_event("gpu_memset", 300, 500, external_id=1, correlation=2),
            trace_event(
                "cuda_runtime", 500, 500, "cudaStreamSynchronize", external_id=1
            ),
            trace_event("kernel", 500, 700, external_id=1),
            trace_event("gpu_memcpy", 1500, 200),
            trace_event("gpu_memset", 1000, 100),
        ],
    )
    report = build_component_report([trace_path])
    transfer = next(
        component
        for component in report["components"]
        if component["name"] == "personaplex.h2d"
    )
    assert transfer["gpu_transfer_milliseconds"] == 0.2
    assert transfer["transfer_count"] == 1
    assert transfer["gpu_memset_milliseconds"] == 0.5
    assert transfer["cuda_synchronization_milliseconds"] == 0.5
    assert report["gpu_transfer_milliseconds"] == 0.4
    assert report["gpu_memset_milliseconds"] == 0.6
    assert report["unattributed_gpu_transfer_milliseconds"] == 0.2
    assert report["unattributed_gpu_memset_milliseconds"] == 0.1
    assert report["transfer_attribution_fraction"] == 0.5
    assert report["devices"][0]["gpu_busy_milliseconds"] == 1.1
    assert report["devices"][0]["gpu_idle_milliseconds"] == pytest.approx(0.9)


def test_missing_scopes_preserve_unattributed_costs(tmp_path: Path) -> None:
    trace_path = write_trace(
        tmp_path / "unannotated.json",
        [
            trace_event("kernel", 100, 300),
            trace_event("kernel", 200, 500),
        ],
    )
    report = build_component_report([trace_path])
    assert len(report["missing_components"]) == 8
    assert all(
        component["cpu_scope_milliseconds"] is None
        and component["gpu_kernel_milliseconds"] is None
        for component in report["components"]
    )
    assert report["gpu_kernel_milliseconds"] == 0.6
    assert report["unattributed_gpu_kernel_milliseconds"] == 0.6
    assert report["kernel_attribution_fraction"] == 0.0
    assert report["transfer_attribution_fraction"] is None
    assert report["devices"][0]["window_source"] == "gpu_events"
    assert report["devices"][0]["gpu_idle_milliseconds"] == 0.0
