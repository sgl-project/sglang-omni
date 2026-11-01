# SPDX-License-Identifier: Apache-2.0
"""An offload run records what it measured under, and two runs compare only when that matches."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.offload.offload_record import (
    CompareError,
    LatencyRecord,
    MemoryRecord,
    OffloadRun,
    Provenance,
    compare_runs,
    format_comparison,
    load_run,
    write_run,
)


def provenance(**overrides: object) -> Provenance:
    fields = {
        "git_commit": "1b726ab",
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "gpu_memory_mib": 81920.0,
        "driver_version": "580.82",
        "model": "Qwen3-Omni-30B-A3B-Instruct",
        "checkpoint": "main",
        "concurrency": 1,
        "workload": "seedtts-en-100",
        "recorded_at": "2026-11-01T00:00:00Z",
    }
    return Provenance(**{**fields, **overrides})


def run(
    label: str = "baseline",
    *,
    peak_mib: float = 70000.0,
    throughput_qps: float = 2.0,
    ttfa_p95_s: float = 0.5,
    restore_s: float | None = None,
    **provenance_overrides: object,
) -> OffloadRun:
    return OffloadRun(
        label=label,
        provenance=provenance(**provenance_overrides),
        memory=MemoryRecord(
            peak_mib=peak_mib,
            steady_mib=peak_mib - 5000.0,
            after_idle_mib=peak_mib - 20000.0,
            process_peak_mib=peak_mib - 1000.0,
        ),
        latency=LatencyRecord(
            ttfa_p50_s=ttfa_p95_s / 2,
            ttfa_p95_s=ttfa_p95_s,
            inter_chunk_p95_s=0.08,
            inter_chunk_p99_s=0.12,
            rtf_p95=0.4,
            throughput_qps=throughput_qps,
        ),
        startup_s=30.0,
        restore_s=restore_s,
    )


def test_a_run_round_trips_through_its_json(tmp_path: Path) -> None:
    written = write_run(run(), tmp_path / "offload-baseline.json")
    loaded = load_run(written)
    assert loaded == run()
    assert json.loads(written.read_text())["schema_version"] == 1


def test_a_pair_measured_on_another_gpu_is_refused() -> None:
    with pytest.raises(CompareError) as error:
        compare_runs(run(), run("candidate", gpu_name="NVIDIA H200"))
    assert "gpu_name" in str(error.value)


def test_a_pair_measured_at_another_concurrency_is_refused() -> None:
    with pytest.raises(CompareError) as error:
        compare_runs(run(), run("candidate", concurrency=8))
    assert "concurrency" in str(error.value)


def test_a_ratio_says_which_direction_is_better() -> None:
    comparison = compare_runs(
        run(), run("candidate", peak_mib=60000.0, throughput_qps=1.5)
    )
    verdicts = {(delta.metric, delta.is_improvement) for delta in comparison.deltas}
    assert ("memory.peak_mib", True) in verdicts
    assert ("latency.throughput_qps", False) in verdicts
    slow = compare_runs(run(), run("candidate", ttfa_p95_s=0.8))
    assert all(
        not delta.is_improvement
        for delta in slow.deltas
        if delta.metric.startswith("latency.ttfa")
    )


def test_a_field_either_side_did_not_measure_is_skipped() -> None:
    comparison = compare_runs(run(), run("candidate", restore_s=None))
    assert "restore_s" in comparison.skipped
    assert "moe_cache_miss_bytes" in comparison.skipped
    assert all(delta.metric not in comparison.skipped for delta in comparison.deltas)


def test_a_restore_time_compares_once_both_runs_have_it() -> None:
    comparison = compare_runs(run(restore_s=14.0), run("candidate", restore_s=3.0))
    restore = next(delta for delta in comparison.deltas if delta.metric == "restore_s")
    assert restore.is_improvement and restore.ratio == pytest.approx(3.0 / 14.0)
    assert "restore_s" not in comparison.skipped


def test_the_report_names_every_metric_it_compared() -> None:
    text = format_comparison(compare_runs(run(), run("candidate", peak_mib=65000.0)))
    assert "candidate against baseline" in text
    assert "memory.peak_mib" in text
    assert "not compared" in text
