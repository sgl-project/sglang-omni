# SPDX-License-Identifier: Apache-2.0
"""Production-mimic TTS serving benchmark CI through a two-worker router."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.request import ProxyHandler, build_opener

import pytest

from benchmarks.tts_serving.spec import load_spec
from tests.test_model.omni_router_utils import (
    CiRouterTopology,
    ManagedRouterHandle,
    assert_router_healthy,
    assert_workers_served_requests_since,
    launch_managed_router,
    print_router_diagnostics,
    router_get_json,
    worker_request_delta,
)
from tests.test_model.tts_ci_config import (
    THRESHOLD_SLACK_HIGHER,
    THRESHOLD_SLACK_LOWER,
    TTS_CI_PRESETS,
)
from tests.utils import MetricCheckCollector, no_proxy_env, wait_for_gpu_memory_release

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVING_SPEC = PROJECT_ROOT / "benchmarks/tts_serving/examples/stress.json"
REFERENCE_AUDIO_ROOT = PROJECT_ROOT / "docs/_static/audio"
OUTPUT_ROOT_ENV = "TTS_SERVING_STAGE_OUTPUT_ROOT"
MODEL_PRESET = TTS_CI_PRESETS["higgs"].model
MODEL_PATH = MODEL_PRESET.model_path
BENCHMARK_VALIDATION_FILE = "benchmark_validation.json"
CLEANUP_VALIDATION_FILE = "cleanup_validation.json"
BENCHMARK_TIMEOUT_S = 1800
BENCHMARK_TIMEOUT_RETURNCODE = 124
MIXED_STAGE = "mixed-production"
ROUTER_STAGE_SNAPSHOTS = "router_stage_snapshots.json"
ROUTER_REJECTION_METRIC = "sglang_omni_router_rejections_total"
EXPECTED_WORKLOAD_SAMPLES = {
    "speech_normal": 50,
    "rest_stream": 50,
    "ws_normal": 50,
    "ws_stream_audio": 40,
    "batch_32_all_valid": 20,
    "long_prefill_decode": 20,
}
EXPECTED_COLLISION_EPOCHS = 20
EXPECTED_COVERAGE_REQUESTS = 102
EXPECTED_COVERAGE_ERRORS = 35
SERVING_MIXED_SPEECH_NORMAL_LATENCY_P95_S_REF: float | None = 0.930258288004552
SERVING_MIXED_SPEECH_NORMAL_RTF_P95_REF: float | None = 0.1662
SERVING_MIXED_REST_STREAM_TTFA_P95_S_REF: float | None = 0.20333050900080707
SERVING_MIXED_REST_STREAM_INTER_CHUNK_P95_S_REF: float | None = 0.5503353969979798
SERVING_MIXED_REST_STREAM_LATENCY_P95_S_REF: float | None = 1.061
SERVING_MIXED_REST_STREAM_RTF_P95_REF: float | None = 0.1787005540906896
SERVING_MIXED_BATCH32_LATENCY_P95_S_REF: float | None = 9.522
SERVING_MIXED_WS_NORMAL_TTFA_P95_S_REF: float | None = 1.0704407309967792
SERVING_MIXED_WS_NORMAL_LATENCY_P95_S_REF: float | None = 1.071
SERVING_MIXED_WS_STREAM_TTFA_P95_S_REF: float | None = 0.1661
SERVING_MIXED_WS_STREAM_INTER_CHUNK_P95_S_REF: float | None = 0.5645
SERVING_MIXED_WS_STREAM_LATENCY_P95_S_REF: float | None = 7.827005511997413
SERVING_MIXED_WS_STREAM_RTF_P95_REF: float | None = 0.1195
SERVING_MIXED_LONG_PROMPT_TOKENS_MIN_REF: float | None = 681.0
SERVING_MIXED_LONG_COMPLETION_TOKENS_MIN_REF: float | None = 95.0
SERVING_MIXED_LONG_LATENCY_P95_S_REF: float | None = 9.521216713998001
SERVING_MIXED_LONG_AUDIO_DURATION_MIN_S_REF: float | None = 3.52
SERVING_MIXED_LONG_OUTPUT_TOK_PER_REQ_S_REF: float | None = 219.47325440026654


def _minimum(reference: float | None) -> float | None:
    return None if reference is None else round(reference * THRESHOLD_SLACK_HIGHER, 6)


def _maximum(reference: float | None) -> float | None:
    return None if reference is None else round(reference * THRESHOLD_SLACK_LOWER, 6)


@dataclass(frozen=True)
class ServingRun:
    base_url: str
    run_dir: Path
    benchmark_dir: Path
    spec_path: Path
    request_timeout_s: int
    router: ManagedRouterHandle
    router_before: dict
    router_rejections_before: int


@dataclass(frozen=True)
class MetricGate:
    key: str
    workload: str
    metric: str
    statistic: Literal["value", "min", "p95"]
    direction: Literal["min", "max"]
    threshold: float | None


METRIC_GATES = (
    MetricGate(
        "speech_normal.latency_p95_s_max",
        "speech_normal",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_MIXED_SPEECH_NORMAL_LATENCY_P95_S_REF),
    ),
    MetricGate(
        "speech_normal.rtf_p95_max",
        "speech_normal",
        "rtf",
        "p95",
        "max",
        _maximum(SERVING_MIXED_SPEECH_NORMAL_RTF_P95_REF),
    ),
    MetricGate(
        "rest_stream.ttfa_p95_s_max",
        "rest_stream",
        "ttfa_s",
        "p95",
        "max",
        _maximum(SERVING_MIXED_REST_STREAM_TTFA_P95_S_REF),
    ),
    MetricGate(
        "rest_stream.inter_chunk_p95_s_max",
        "rest_stream",
        "inter_chunk_s",
        "p95",
        "max",
        _maximum(SERVING_MIXED_REST_STREAM_INTER_CHUNK_P95_S_REF),
    ),
    MetricGate(
        "rest_stream.latency_p95_s_max",
        "rest_stream",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_MIXED_REST_STREAM_LATENCY_P95_S_REF),
    ),
    MetricGate(
        "rest_stream.rtf_p95_max",
        "rest_stream",
        "rtf",
        "p95",
        "max",
        _maximum(SERVING_MIXED_REST_STREAM_RTF_P95_REF),
    ),
    MetricGate(
        "batch32.latency_p95_s_max",
        "batch_32_all_valid",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_MIXED_BATCH32_LATENCY_P95_S_REF),
    ),
    MetricGate(
        "ws_normal.ttfa_p95_s_max",
        "ws_normal",
        "ttfa_s",
        "p95",
        "max",
        _maximum(SERVING_MIXED_WS_NORMAL_TTFA_P95_S_REF),
    ),
    MetricGate(
        "ws_normal.latency_p95_s_max",
        "ws_normal",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_MIXED_WS_NORMAL_LATENCY_P95_S_REF),
    ),
    MetricGate(
        "ws_stream.ttfa_p95_s_max",
        "ws_stream_audio",
        "ttfa_s",
        "p95",
        "max",
        _maximum(SERVING_MIXED_WS_STREAM_TTFA_P95_S_REF),
    ),
    MetricGate(
        "ws_stream.inter_chunk_p95_s_max",
        "ws_stream_audio",
        "inter_chunk_s",
        "p95",
        "max",
        _maximum(SERVING_MIXED_WS_STREAM_INTER_CHUNK_P95_S_REF),
    ),
    MetricGate(
        "ws_stream.latency_p95_s_max",
        "ws_stream_audio",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_MIXED_WS_STREAM_LATENCY_P95_S_REF),
    ),
    MetricGate(
        "ws_stream.rtf_p95_max",
        "ws_stream_audio",
        "rtf",
        "p95",
        "max",
        _maximum(SERVING_MIXED_WS_STREAM_RTF_P95_REF),
    ),
    MetricGate(
        "long.prompt_tokens_min",
        "long_prefill_decode",
        "prompt_tokens",
        "min",
        "min",
        _minimum(SERVING_MIXED_LONG_PROMPT_TOKENS_MIN_REF),
    ),
    MetricGate(
        "long.completion_tokens_min",
        "long_prefill_decode",
        "completion_tokens",
        "min",
        "min",
        _minimum(SERVING_MIXED_LONG_COMPLETION_TOKENS_MIN_REF),
    ),
    MetricGate(
        "long.latency_p95_s_max",
        "long_prefill_decode",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_MIXED_LONG_LATENCY_P95_S_REF),
    ),
    MetricGate(
        "long.audio_duration_s_min",
        "long_prefill_decode",
        "audio_duration_s",
        "min",
        "min",
        _minimum(SERVING_MIXED_LONG_AUDIO_DURATION_MIN_S_REF),
    ),
    MetricGate(
        "long.output_tok_per_req_s_min",
        "long_prefill_decode",
        "output_tok_per_req_s",
        "value",
        "min",
        _minimum(SERVING_MIXED_LONG_OUTPUT_TOK_PER_REQ_S_REF),
    ),
)


def _materialize_spec(run_dir: Path, base_url: str) -> Path:
    spec = json.loads(SERVING_SPEC.read_text(encoding="utf-8"))
    spec["base_url"] = base_url
    spec["run_id"] = "tts-serving-ci"
    path = run_dir / "spec.json"
    path.write_text(
        json.dumps(spec, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def serving_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ServingRun]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("TTS serving CI requires two CUDA devices")

    configured_root = os.environ.get(OUTPUT_ROOT_ENV)
    if configured_root:
        run_dir = Path(configured_root).resolve()
        retry_attempt = os.environ.get("OMNI_CI_ATTEMPT")
        if retry_attempt:
            run_dir /= f"attempt-{retry_attempt}"
    else:
        run_dir = tmp_path_factory.mktemp("tts-serving-ci")
    benchmark_dir = run_dir / "benchmark"
    speaker_dir = tmp_path_factory.mktemp("tts-serving-speakers")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    worker_args = [
        "--allowed-local-media-path",
        str(REFERENCE_AUDIO_ROOT),
        *shlex.split(MODEL_PRESET.worker_extra_args),
    ]
    worker_env = {
        "PYTHONPATH": str(PROJECT_ROOT),
        "SPEAKER_SAMPLES_DIR": str(speaker_dir),
        "SPEAKER_MAX_UPLOADED": "1000",
    }
    router: ManagedRouterHandle | None = None
    cleanup_error: Exception | None = None
    try:
        with launch_managed_router(
            tmp_path_factory=tmp_path_factory,
            model_path=MODEL_PATH,
            model_name=MODEL_PATH,
            worker_extra_args=shlex.join(worker_args),
            router_topology=CiRouterTopology.TTS_SERVING,
            num_workers=2,
            num_gpus_per_worker=1,
            wait_timeout=MODEL_PRESET.startup_timeout,
            force_log=True,
            worker_env=worker_env,
        ) as router:
            base_url = f"http://127.0.0.1:{router.port}"
            spec_path = _materialize_spec(run_dir, base_url)
            request_timeout_s = load_spec(spec_path).params.timeout_s
            assert_router_healthy(router)
            router_before = router_get_json(router.port, "/diagnostics")
            router_rejections_before = _router_rejections_total(
                base_url,
                request_timeout_s,
            )
            yield ServingRun(
                base_url=base_url,
                run_dir=run_dir,
                benchmark_dir=benchmark_dir,
                spec_path=spec_path,
                request_timeout_s=request_timeout_s,
                router=router,
                router_before=router_before,
                router_rejections_before=router_rejections_before,
            )
    except Exception as exc:
        cleanup_error = exc
        raise
    finally:
        cleanup_payload = {
            "valid": False,
            "router_stopped": bool(router is None or router.stopped),
            "gpu_memory_released": False,
            "error": None,
        }
        try:
            wait_for_gpu_memory_release()
            cleanup_payload["gpu_memory_released"] = True
            cleanup_payload["valid"] = cleanup_payload["router_stopped"]
        except Exception as exc:
            cleanup_payload["error"] = f"{type(exc).__name__}: {exc}"
            if cleanup_error is None:
                cleanup_error = exc
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / CLEANUP_VALIDATION_FILE).write_text(
            json.dumps(cleanup_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        if cleanup_error is not None and sys.exc_info()[0] is None:
            raise cleanup_error


def _run_benchmark(run: ServingRun) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_tts_serving",
        "--spec",
        str(run.spec_path),
        "--out",
        str(run.benchmark_dir),
        "--router-stage-snapshots",
        str(run.benchmark_dir / ROUTER_STAGE_SNAPSHOTS),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env={**no_proxy_env(), "PYTHONPATH": str(PROJECT_ROOT)},
            check=False,
            timeout=BENCHMARK_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(f"benchmark killed after {BENCHMARK_TIMEOUT_S}s timeout", flush=True)
        completed = subprocess.CompletedProcess(
            command,
            returncode=BENCHMARK_TIMEOUT_RETURNCODE,
        )
    (run.run_dir / "benchmark.wall_time.json").write_text(
        json.dumps(
            {
                "returncode": completed.returncode,
                "wall_time_s": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return completed


def _metric_value(summary: dict, gate: MetricGate) -> float | None:
    metric = summary.get(gate.metric)
    if gate.statistic == "value":
        return float(metric) if isinstance(metric, (int, float)) else None
    if not isinstance(metric, dict):
        return None
    value = metric.get(gate.statistic)
    return float(value) if isinstance(value, (int, float)) else None


def _check_performance(
    report: dict,
    measurement_checks: MetricCheckCollector,
    threshold_checks: MetricCheckCollector,
) -> None:
    summaries = (
        report.get("metrics", {}).get("by_stage_and_workload", {}).get(MIXED_STAGE, {})
    )
    pending = sorted(gate.key for gate in METRIC_GATES if gate.threshold is None)
    threshold_checks.check(
        not pending,
        f"serving thresholds require calibration: {pending}",
    )
    workload_summaries: dict[str, dict] = {}
    for workload, expected_samples in EXPECTED_WORKLOAD_SAMPLES.items():
        summary = summaries.get(workload)
        if not isinstance(summary, dict):
            measurement_checks.fail(
                f"missing result summary for {MIXED_STAGE}/{workload}"
            )
            continue
        workload_summaries[workload] = summary
        samples = summary.get("successful_request_count")
        measurement_checks.check(
            samples == expected_samples,
            f"{MIXED_STAGE}/{workload} successful samples={samples!r}, "
            f"expected={expected_samples}",
        )

    for gate in METRIC_GATES:
        summary = workload_summaries.get(gate.workload)
        if summary is None:
            continue
        expected_samples = EXPECTED_WORKLOAD_SAMPLES[gate.workload]
        metric_samples = summary.get("metric_sample_counts", {}).get(gate.metric)
        measurement_checks.check(
            metric_samples == expected_samples,
            f"{gate.key} metric samples={metric_samples!r}, "
            f"expected={expected_samples}",
        )
        value = _metric_value(summary, gate)
        measurement_checks.check(value is not None, f"{gate.key} is missing")
        if value is None or gate.threshold is None:
            continue
        if gate.direction == "min":
            threshold_checks.check(
                value >= gate.threshold,
                f"{gate.key}={value} < {gate.threshold}",
            )
        else:
            threshold_checks.check(
                value <= gate.threshold,
                f"{gate.key}={value} > {gate.threshold}",
            )


def _mixed_result_summary(run: ServingRun) -> dict:
    events_path = run.benchmark_dir / "raw" / "events.jsonl"
    results = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mixed_results = [
        result
        for result in results
        if result.get("stage_id") == MIXED_STAGE and result.get("workload") is not None
    ]
    expected_count = sum(EXPECTED_WORKLOAD_SAMPLES.values())
    assert (
        len(mixed_results) == expected_count
    ), f"expected {expected_count} mixed results, got {len(mixed_results)}"
    scenario_ids = [result.get("scenario_id") for result in mixed_results]
    assert all(
        isinstance(value, str) for value in scenario_ids
    ), "mixed results contain an invalid scenario ID"
    assert len(set(scenario_ids)) == len(
        scenario_ids
    ), "mixed results contain duplicate scenario IDs"
    workload_counts: Counter[str] = Counter()
    for result in mixed_results:
        scenario_id = result["scenario_id"]
        workload = result.get("workload")
        assert (
            workload in EXPECTED_WORKLOAD_SAMPLES
        ), f"mixed scenario {scenario_id!r} has unexpected workload {workload!r}"
        assert (
            result.get("expected_success") is True and result.get("success") is True
        ), f"mixed scenario {scenario_id!r} did not pass"
        assert result.get("endpoint") in {
            "speech",
            "speech_stream",
            "batch",
            "websocket",
        }, f"mixed scenario {scenario_id!r} has an unexpected endpoint"
        workload_counts[workload] += 1
    assert (
        dict(workload_counts) == EXPECTED_WORKLOAD_SAMPLES
    ), f"mixed workload counts changed: {dict(workload_counts)}"
    return {
        "total_samples": len(mixed_results),
        "by_workload": dict(workload_counts),
    }


def _measured_worker_minimums(mixed_delta: dict) -> tuple[list[int], list[str]]:
    measured_by_class = {
        "speech_http": (
            EXPECTED_WORKLOAD_SAMPLES["speech_normal"]
            + EXPECTED_WORKLOAD_SAMPLES["rest_stream"]
            + EXPECTED_WORKLOAD_SAMPLES["long_prefill_decode"]
        ),
        "speech_batch": EXPECTED_WORKLOAD_SAMPLES["batch_32_all_valid"],
        "speech_websocket": (
            EXPECTED_WORKLOAD_SAMPLES["ws_normal"]
            + EXPECTED_WORKLOAD_SAMPLES["ws_stream_audio"]
        ),
    }
    workers = mixed_delta["workers"]
    minimums = [0] * len(workers)
    failures: list[str] = []
    for service_class, measured_requests in measured_by_class.items():
        dispatches = [
            int(worker["routed_requests_by_class"].get(service_class, 0))
            for worker in workers
        ]
        total_dispatches = sum(dispatches)
        if total_dispatches < measured_requests:
            failures.append(
                f"mixed stage served {total_dispatches} {service_class} requests; "
                f"expected at least {measured_requests}"
            )
            continue
        unmeasured_dispatches = total_dispatches - measured_requests
        for index, worker_dispatches in enumerate(dispatches):
            minimums[index] += max(0, worker_dispatches - unmeasured_dispatches)
    return minimums, failures


def _check_router(run: ServingRun, checks: MetricCheckCollector) -> None:
    try:
        delta = assert_workers_served_requests_since(
            handle=run.router,
            before_snapshot=run.router_before,
            label="TTS serving",
            min_worker_share=0.0,
        )
        mixed_summary = _mixed_result_summary(run)
        stage_snapshots = json.loads(
            (run.benchmark_dir / ROUTER_STAGE_SNAPSHOTS).read_text(encoding="utf-8")
        )
        mixed_snapshots = stage_snapshots[MIXED_STAGE]
        mixed_delta = worker_request_delta(
            mixed_snapshots["before"],
            mixed_snapshots["after"],
        )
        measured_worker_minimums, dispatch_failures = _measured_worker_minimums(
            mixed_delta
        )
        router_rejections_after = _router_rejections_total(
            run.base_url,
            run.request_timeout_s,
        )
    except Exception as exc:
        checks.fail(f"router validation failed: {exc}")
        return
    workers = delta["workers"]
    class_counts = {
        service_class: sum(
            int(worker["routed_requests_by_class"].get(service_class, 0))
            for worker in workers
        )
        for service_class in (
            "speech_http",
            "speech_batch",
            "speech_websocket",
            "voice_control",
        )
    }
    rejected_delta = router_rejections_after - run.router_rejections_before
    (run.run_dir / "router_validation.json").write_text(
        json.dumps(
            {
                "workers_before": run.router_before,
                "worker_delta": delta,
                "mixed_worker_delta": mixed_delta,
                "mixed_measured_worker_minimums": measured_worker_minimums,
                "mixed_results": mixed_summary,
                "class_dispatches": class_counts,
                "router_rejections": {
                    "before": run.router_rejections_before,
                    "after": router_rejections_after,
                    "delta": rejected_delta,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for failure in dispatch_failures:
        checks.fail(failure)
    checks.check(
        rejected_delta == 0,
        f"router rejected valid scheduled traffic: delta={rejected_delta}",
    )
    checks.check(
        all(count > 0 for count in measured_worker_minimums),
        "both workers must provably serve measured mixed-production traffic: "
        f"minimums={measured_worker_minimums}",
    )
    owners = [
        worker["worker_id"] for worker in workers if worker.get("voice_owner") is True
    ]
    checks.check(
        owners == ["tts-serving-1"],
        f"unexpected voice owner: {owners}",
    )
    voice_counts = {
        worker["worker_id"]: int(
            worker["routed_requests_by_class"].get("voice_control", 0)
        )
        for worker in workers
    }
    checks.check(
        voice_counts.get("tts-serving-1", 0) > 0
        and all(
            count == 0
            for worker_id, count in voice_counts.items()
            if worker_id != "tts-serving-1"
        ),
        f"voice control did not remain on the exact owner: {voice_counts}",
    )


def _get_json(url: str, timeout_s: int) -> dict:
    opener = build_opener(ProxyHandler({}))
    with opener.open(url, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected JSON object from {url}")
    return payload


def _router_rejections_total(base_url: str, timeout_s: int) -> int:
    opener = build_opener(ProxyHandler({}))
    with opener.open(f"{base_url}/metrics", timeout=timeout_s) as response:
        metrics = response.read().decode("utf-8")
    prefix = f"{ROUTER_REJECTION_METRIC}{{"
    samples = [
        int(line.rsplit(" ", 1)[1])
        for line in metrics.splitlines()
        if line.startswith(prefix)
    ]
    if not samples:
        raise AssertionError(f"router metrics omit {ROUTER_REJECTION_METRIC}")
    return sum(samples)


def _write_benchmark_validation(
    run_dir: Path,
    benchmark_checks: MetricCheckCollector,
    measurement_checks: MetricCheckCollector,
    threshold_checks: MetricCheckCollector,
) -> None:
    failures = benchmark_checks.failures + measurement_checks.failures
    (run_dir / BENCHMARK_VALIDATION_FILE).write_text(
        json.dumps(
            {
                "valid": not failures,
                "failures": failures,
                "threshold_assertion_failed": bool(threshold_checks.failures),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.benchmark
def test_tts_serving_stress(serving_run: ServingRun) -> None:
    completed = _run_benchmark(serving_run)
    benchmark_checks = MetricCheckCollector("TTS serving benchmark")
    measurement_checks = MetricCheckCollector("TTS serving measurements")
    threshold_checks = MetricCheckCollector("TTS serving thresholds")
    benchmark_checks.check(
        completed.returncode == 0,
        f"benchmark exited with return code {completed.returncode}",
    )

    results_path = serving_run.benchmark_dir / "results.json"
    benchmark_checks.check(
        results_path.is_file(),
        f"missing benchmark report: {results_path}",
    )
    if results_path.is_file():
        try:
            report = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            benchmark_checks.fail(f"could not read benchmark report: {exc}")
        else:
            overall = report.get("overall", {})
            metrics = report.get("metrics", {})
            mixed = metrics.get("mixed_arrival", {}).get(MIXED_STAGE, {})
            benchmark_checks.check(
                report.get("harness_status") == "ok",
                "benchmark harness failed",
            )
            benchmark_checks.check(
                report.get("schema_version") == 4,
                "unexpected benchmark report schema",
            )
            benchmark_checks.check(
                overall.get("passed") is True,
                "benchmark did not pass",
            )
            benchmark_checks.check(
                overall.get("failed") == 0,
                "benchmark contains failures",
            )
            for key in (
                "load_generation_valid",
                "coverage_contract_valid",
                "mixed_arrival_valid",
            ):
                benchmark_checks.check(
                    overall.get(key) is True,
                    f"{key} is false",
                )
            for key in (
                "failures",
                "unsupported_contracts",
                "coverage_failures",
                "mixed_arrival_failures",
            ):
                benchmark_checks.check(
                    not report.get(key),
                    f"{key} were reported",
                )
            benchmark_checks.check(
                not metrics.get("admission_status_counts"),
                "benchmark admission failures were reported",
            )
            benchmark_checks.check(
                mixed.get("configured_collision_epoch_count")
                == EXPECTED_COLLISION_EPOCHS,
                "configured collision epoch count changed",
            )
            benchmark_checks.check(
                mixed.get("observed_collision_epoch_count")
                == EXPECTED_COLLISION_EPOCHS,
                "not every collision epoch was observed",
            )
            benchmark_checks.check(
                mixed.get("coverage_request_count") == EXPECTED_COVERAGE_REQUESTS,
                "scheduled coverage population changed",
            )
            benchmark_checks.check(
                mixed.get("coverage_passed_count") == EXPECTED_COVERAGE_REQUESTS,
                "scheduled coverage traffic did not pass",
            )
            benchmark_checks.check(
                mixed.get("expected_error_request_count") == EXPECTED_COVERAGE_ERRORS,
                "scheduled expected-error population changed",
            )
            _check_performance(report, measurement_checks, threshold_checks)

    _check_router(serving_run, benchmark_checks)
    try:
        voices = _get_json(
            f"{serving_run.base_url}/v1/audio/voices",
            serving_run.request_timeout_s,
        )
    except Exception as exc:
        benchmark_checks.fail(f"voice cleanup probe failed: {exc}")
    else:
        leaked_voices = sorted(
            item["name"]
            for item in voices.get("uploaded_voices", [])
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and item["name"].startswith("bench_voice_")
        )
        benchmark_checks.check(
            not leaked_voices,
            f"benchmark voices leaked: {leaked_voices}",
        )

    _write_benchmark_validation(
        serving_run.run_dir,
        benchmark_checks,
        measurement_checks,
        threshold_checks,
    )
    if benchmark_checks.failures or measurement_checks.failures:
        print_router_diagnostics(serving_run.router)
    benchmark_checks.assert_all()
    measurement_checks.assert_all()
    threshold_checks.assert_all()
