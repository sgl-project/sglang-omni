# SPDX-License-Identifier-Apache-2.0
"""Tests for GPU metric collection and aggregation (no GPU required).

These tests run in a CPU-only CI environment. ``sample_gpu_metrics`` is
monkeypatched to return deterministic values so the aggregation logic in
:mod:`sglang_omni.profiler.views` and the CLI can be exercised end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang_omni.profiler import gpu_metrics
from sglang_omni.profiler.views import build_report, gpu_breakdown, reconstruct_timelines


def test_sample_gpu_metrics_returns_empty_without_cuda(monkeypatch) -> None:
    # Force the "no CUDA" code path regardless of the host.
    monkeypatch.setattr(gpu_metrics, "_cuda_module", lambda: None)
    assert gpu_metrics.sample_gpu_metrics() == {}
    assert gpu_metrics.sample_gpu_metrics("cuda:0") == {}


def test_sample_gpu_metrics_divides_by_mb(monkeypatch) -> None:
    fake_cuda = type("C", (), {})()
    mb_bytes = 1024 * 1024

    def _alloc(device=None):
        return 2 * mb_bytes  # 2 MiB

    def _resv(device=None):
        return 5 * mb_bytes  # 5 MiB

    fake_cuda.memory_allocated = _alloc
    fake_cuda.memory_reserved = _resv
    monkeypatch.setattr(gpu_metrics, "_cuda_module", lambda: fake_cuda)

    out = gpu_metrics.sample_gpu_metrics("cuda:0")
    assert out == {"gpu_mem_allocated_mb": 2.0, "gpu_mem_reserved_mb": 5.0}


def test_compute_throughput_metrics_tokens_and_rtf() -> None:
    # 2000 ms, 100 tokens, 4s audio -> 50 tok/s, RTF=0.5 (faster than realtime)
    m = gpu_metrics.compute_throughput_metrics(2000.0, output_tokens=100, audio_seconds=4.0)
    assert m["tokens_per_sec"] == pytest.approx(50.0)
    assert m["rtf"] == pytest.approx(0.5)


def test_compute_throughput_metrics_handles_missing_inputs() -> None:
    assert gpu_metrics.compute_throughput_metrics(1000.0) == {}
    assert gpu_metrics.compute_throughput_metrics(0.0, output_tokens=10) == {}
    assert gpu_metrics.compute_throughput_metrics(-5.0, audio_seconds=1.0) == {}


def _write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for ev in events:
            fp.write(json.dumps(ev))
            fp.write("\n")


def _ev(request_id, name, ts, metadata=None):
    return {
        "request_id": request_id,
        "stage": None,
        "event_name": name,
        "timestamp_ns": ts,
        "run_id": "run_test",
        "pid": 1,
        "metadata": metadata or {},
    }


def test_gpu_breakdown_aggregates_peak_and_global(tmp_path: Path) -> None:
    events = [
        _ev("r1", "scheduler_prefill_start", 1000, {"gpu_mem_allocated_mb": 100.0, "gpu_mem_reserved_mb": 200.0}),
        _ev("r1", "scheduler_first_emit", 2000, {"gpu_mem_allocated_mb": 150.0, "gpu_mem_reserved_mb": 200.0}),
        _ev("r2", "scheduler_prefill_start", 1100, {"gpu_mem_allocated_mb": 80.0, "gpu_mem_reserved_mb": 180.0}),
    ]
    _write_events(tmp_path / "events_test_1.jsonl", events)

    tls = reconstruct_timelines(tmp_path)
    out = gpu_breakdown(tls)

    by_rid = {r["request_id"]: r for r in out["per_request"]}
    assert by_rid["r1"]["gpu_mem_allocated_peak_mb"] == 150.0
    assert by_rid["r2"]["gpu_mem_allocated_peak_mb"] == 80.0
    # global peak is the max across all requests
    assert out["global"]["gpu_mem_allocated"]["peak_mb"] == 150.0
    assert out["global"]["gpu_mem_allocated"]["mean_mb"] == pytest.approx((100 + 150 + 80) / 3)


def test_gpu_breakdown_empty_when_no_gpu_fields(tmp_path: Path) -> None:
    events = [_ev("r1", "request_admission", 1000)]
    _write_events(tmp_path / "events_test_2.jsonl", events)
    out = gpu_breakdown(reconstruct_timelines(tmp_path))
    assert out["per_request"] == []
    assert out["global"]["gpu_mem_allocated"]["peak_mb"] is None


def test_build_report_includes_gpu_breakdown(tmp_path: Path) -> None:
    events = [
        _ev("r1", "scheduler_prefill_start", 1000, {"gpu_mem_allocated_mb": 120.0}),
        _ev("r1", "scheduler_first_emit", 2000, {"gpu_mem_allocated_mb": 140.0}),
    ]
    _write_events(tmp_path / "events_test_3.jsonl", events)
    report = build_report(tmp_path)
    assert "gpu_breakdown" in report
    assert report["gpu_breakdown"]["per_request"][0]["request_id"] == "r1"
    assert report["gpu_breakdown"]["global"]["gpu_mem_allocated"]["peak_mb"] == 140.0
