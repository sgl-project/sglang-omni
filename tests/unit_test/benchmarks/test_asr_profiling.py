# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the ASR bottleneck-profiling helpers (issue #1324)."""

from __future__ import annotations

import json
import time

import pytest

from benchmarks.eval import asr_profiling
from benchmarks.eval.asr_profiling import (
    UtilizationSampler,
    build_stage_breakdown,
    collect_environment_fingerprint,
    collect_server_identity,
    start_request_profile,
    stop_request_profile,
)
from benchmarks.eval.benchmark_asr_seedtts import _aggregate, _print_table


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def test_profile_control_posts_run_id_and_event_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_post(url, *, json, timeout, proxies):
        del timeout, proxies
        calls.append((url, json))
        return _FakeResponse({"run_id": json.get("run_id"), "event_dir": "/tmp/e"})

    monkeypatch.setattr(asr_profiling.requests, "post", fake_post)

    started = start_request_profile("http://127.0.0.1:8000/", "run-1", "/tmp/e")
    stopped = stop_request_profile("http://127.0.0.1:8000")

    assert started["run_id"] == "run-1"
    assert stopped["run_id"] is None
    assert calls[0][0] == "http://127.0.0.1:8000/start_request_profile"
    assert calls[0][1] == {"run_id": "run-1", "event_dir": "/tmp/e"}
    assert calls[1][0] == "http://127.0.0.1:8000/stop_request_profile"


def test_build_stage_breakdown_pairs_queue_and_decode_intervals(tmp_path) -> None:
    base_ns = 1_000_000_000
    events = [
        ("scheduler_request_build_start", base_ns),
        ("scheduler_request_build_end", base_ns + 5_000_000),
        ("scheduler_queue_enter", base_ns + 6_000_000),
        ("scheduler_prefill_start", base_ns + 30_000_000),
        ("scheduler_prefill_end", base_ns + 60_000_000),
        ("scheduler_first_emit", base_ns + 80_000_000),
        ("stage_complete", base_ns + 200_000_000),
    ]
    event_file = tmp_path / "events_run_1.jsonl"
    with event_file.open("w") as handle:
        for name, ts in events:
            handle.write(
                json.dumps(
                    {
                        "request_id": "req-1",
                        "stage": "asr",
                        "event_name": name,
                        "timestamp_ns": ts,
                        "run_id": "run-1",
                        "pid": 1,
                    }
                )
                + "\n"
            )

    report = build_stage_breakdown(str(tmp_path))

    assert "timelines" not in report
    assert report["request_count"] == 1
    by_interval = {row["interval"]: row for row in report["stage_breakdown"]}
    build = by_interval["scheduler_request_build_start->scheduler_request_build_end"]
    assert build["avg_ms"] == pytest.approx(5.0)
    queue = by_interval["scheduler_queue_enter->scheduler_prefill_start"]
    assert queue["avg_ms"] == pytest.approx(24.0)
    prefill = by_interval["scheduler_prefill_start->scheduler_prefill_end"]
    assert prefill["avg_ms"] == pytest.approx(30.0)
    decode_tail = by_interval["scheduler_prefill_end->stage_complete"]
    assert decode_tail["avg_ms"] == pytest.approx(140.0)
    decode = by_interval["scheduler_first_emit->stage_complete"]
    assert decode["avg_ms"] == pytest.approx(120.0)


def test_utilization_sampler_summarizes_cpu_and_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stat_values = iter([(100, 1000), (200, 1200), (350, 1400), (400, 1600)])
    monkeypatch.setattr(
        asr_profiling, "_read_proc_stat", lambda: next(stat_values, (400, 1600))
    )
    monkeypatch.setattr(
        asr_profiling,
        "_query_gpu_utilization",
        lambda gpu_ids: {"3": {"util_percent": 40.0, "memory_mib": 1024.0}},
    )

    sampler = UtilizationSampler(gpu_ids=[3], interval_s=0.01)
    sampler.start()
    time.sleep(0.08)
    summary = sampler.stop()

    assert summary.samples >= 2
    assert summary.cpu_percent_mean is not None
    assert 0.0 < summary.cpu_percent_mean <= 100.0
    assert summary.gpu["3"]["util_percent_max"] == pytest.approx(40.0)
    assert summary.gpu["3"]["memory_mib_max"] == pytest.approx(1024.0)
    assert summary.to_dict()["samples"] == summary.samples


def test_environment_fingerprint_is_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asr_profiling, "_run_command", lambda command: None)

    fingerprint = collect_environment_fingerprint("Qwen/Qwen3-ASR-1.7B")

    assert fingerprint["git"]["sha"] is None
    assert fingerprint["dependency_freeze_sha256"] is None
    assert fingerprint["gpus"] is None
    assert fingerprint["model_path"] == "Qwen/Qwen3-ASR-1.7B"
    assert "torch" in fingerprint["packages"]
    assert "CUDA_VISIBLE_DEVICES" in fingerprint["env"]


def _repeat(concurrency: int, repeat: int, median: float) -> dict:
    return {
        "concurrency": concurrency,
        "repeat": repeat,
        "evaluated": 10,
        "total": 10,
        "skipped": 0,
        "corpus_wer": 0.01,
        "per_sample_wer_max": 0.05,
        "wall_clock_s": 2.0,
        "throughput_samples_per_s": 5.0,
        "rtfx": 20.0,
        "latency_mean_s": 0.5,
        "latency_median_s": median,
        "latency_p95_s": 0.9,
        "latency_p99_s": 1.1,
        "rtf_mean": 0.1,
        "rtf_p95": 0.2,
        "worker": {},
    }


def test_aggregate_and_table_surface_latency_median() -> None:
    aggregate = _aggregate([_repeat(8, 1, 0.4), _repeat(8, 2, 0.6)])

    assert aggregate["latency_median_s"]["mean"] == pytest.approx(0.5)
    assert aggregate["latency_median_s"]["n"] == 2
    # note (luojiaxuan): the markdown table must render the p50 column without
    # raising for both plain and profiled aggregates.
    aggregate["profile"] = None
    _print_table([aggregate])


def test_server_identity_reports_models_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ModelsResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"data": [{"id": "Qwen/Qwen3-ASR-1.7B"}]}

    monkeypatch.setattr(
        asr_profiling.requests,
        "get",
        lambda url, *, timeout, proxies: _ModelsResponse(),
    )
    identity = collect_server_identity("http://127.0.0.1:8000/")
    assert identity == {
        "url": "http://127.0.0.1:8000",
        "models": ["Qwen/Qwen3-ASR-1.7B"],
    }

    def _down(url, *, timeout, proxies):
        raise asr_profiling.requests.ConnectionError("down")

    monkeypatch.setattr(asr_profiling.requests, "get", _down)
    identity = collect_server_identity("http://127.0.0.1:8000")
    assert identity["url"] == "http://127.0.0.1:8000"
    assert identity["models"] is None


def test_prefill_end_emission_skips_when_no_request_is_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from sglang_omni.scheduling import omni_scheduler

    emitted: list[str] = []
    monkeypatch.setattr(
        omni_scheduler,
        "_emit_event",
        lambda **kwargs: emitted.append(kwargs["request_id"]),
    )
    scheduler = object.__new__(omni_scheduler.OmniScheduler)
    scheduler._prefill_start_done = {"r1"}
    scheduler._prefill_end_done = {"r1"}

    class _ExplodingBatch:
        # note (luojiaxuan): the O(1) fast path must return before touching
        # the batch at all -- steady-state decode pays this on every step.
        @property
        def reqs(self):
            raise AssertionError("fast path must not scan the batch")

        @property
        def is_extend_in_batch(self):
            raise AssertionError("fast path must not build metadata")

    scheduler._emit_prefill_end_for_batch(_ExplodingBatch())
    assert emitted == []

    scheduler._prefill_start_done = {"r1", "r2"}
    batch = SimpleNamespace(reqs=[SimpleNamespace(rid="r2")], is_extend_in_batch=True)
    scheduler._emit_prefill_end_for_batch(batch)
    assert emitted == ["r2"]
    assert scheduler._prefill_end_done == {"r1", "r2"}
