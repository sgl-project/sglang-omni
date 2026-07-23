# SPDX-License-Identifier: Apache-2.0
"""Production-mimic TTS serving benchmark CI on one direct Higgs server."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.request import ProxyHandler, build_opener

import pytest

from benchmarks.tts_serving.spec import load_spec
from sglang_omni.utils import find_available_port
from tests.test_model.tts_ci_config import (
    THRESHOLD_SLACK_HIGHER,
    THRESHOLD_SLACK_LOWER,
    TTS_CI_PRESETS,
)
from tests.utils import (
    MetricCheckCollector,
    no_proxy_env,
    start_server_from_cmd,
    stop_server,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVING_SPEC = PROJECT_ROOT / "benchmarks/tts_serving/examples/stress.json"
REFERENCE_AUDIO_ROOT = PROJECT_ROOT / "docs/_static/audio"
OUTPUT_ROOT_ENV = "TTS_SERVING_STAGE_OUTPUT_ROOT"
MODEL_PRESET = TTS_CI_PRESETS["higgs"].model
MODEL_PATH = MODEL_PRESET.model_path
BENCHMARK_VALIDATION_FILE = "benchmark_validation.json"
BENCHMARK_TIMEOUT_S = 1800
BENCHMARK_TIMEOUT_RETURNCODE = 124

SERVING_CLOSED16_THROUGHPUT_QPS_REF: float | None = 4.450
SERVING_CLOSED16_LATENCY_P95_S_REF: float | None = 12.446
SERVING_RAMP_THROUGHPUT_QPS_REF: float | None = 4.231
SERVING_RAMP_LATENCY_P95_S_REF: float | None = 10.516
SERVING_SOAK_LATENCY_P95_S_REF: float | None = 4.441
SERVING_SPEECH_NORMAL_LATENCY_P95_S_REF: float | None = 0.891
SERVING_SPEECH_NORMAL_RTF_P95_REF: float | None = 0.1461
SERVING_REST_STREAM_TTFA_P95_S_REF: float | None = 0.0927
SERVING_REST_STREAM_INTER_CHUNK_P95_S_REF: float | None = 0.6098
SERVING_REST_STREAM_LATENCY_P95_S_REF: float | None = 0.685
SERVING_REST_STREAM_RTF_P95_REF: float | None = 0.1507
SERVING_BATCH32_LATENCY_P95_S_REF: float | None = 1.586
SERVING_WS_NORMAL_TTFA_P95_S_REF: float | None = 7.2764
SERVING_WS_NORMAL_LATENCY_P95_S_REF: float | None = 7.277
SERVING_WS_STREAM_TTFA_P95_S_REF: float | None = 6.9902
SERVING_WS_STREAM_INTER_CHUNK_P95_S_REF: float | None = 3.3482
SERVING_WS_STREAM_LATENCY_P95_S_REF: float | None = 37.662
SERVING_WS_STREAM_RTF_P95_REF: float | None = 0.5947
SERVING_MIXED_BURST_THROUGHPUT_QPS_REF: float | None = 9.418
SERVING_MIXED_BURST_LATENCY_P95_S_REF: float | None = 17.466
SERVING_LONG_PROMPT_TOKENS_MIN_REF: float | None = 684
SERVING_LONG_COMPLETION_TOKENS_MIN_REF: float | None = 321
SERVING_LONG_LATENCY_P95_S_REF: float | None = 21.968
SERVING_LONG_AUDIO_DURATION_MIN_S_REF: float | None = 12.560
SERVING_LONG_OUTPUT_TOK_PER_REQ_S_REF: float | None = 142.8
SERVING_LONG_BASELINE_LATENCY_MAX_S_REF: float | None = 9.894
SERVING_LONG_BASELINE_OUTPUT_TOK_PER_REQ_S_REF: float | None = 209.1


def _minimum(reference: float | None) -> float | None:
    if reference is None:
        return None
    return round(reference * THRESHOLD_SLACK_HIGHER, 6)


def _maximum(reference: float | None) -> float | None:
    if reference is None:
        return None
    return round(reference * THRESHOLD_SLACK_LOWER, 6)


@dataclass(frozen=True)
class ServingRun:
    base_url: str
    run_dir: Path
    benchmark_dir: Path
    spec_path: Path
    request_timeout_s: int


@dataclass(frozen=True)
class MetricGate:
    key: str
    source: Literal["stage", "workload", "stage_workload"]
    metric: str
    statistic: Literal["value", "min", "max", "p95"]
    direction: Literal["min", "max"]
    threshold: float | None
    stage: str | None = None
    workload: str | None = None


METRIC_GATES = (
    MetricGate(
        "closed16.throughput_qps_min",
        "stage",
        "achieved_rps",
        "value",
        "min",
        _minimum(SERVING_CLOSED16_THROUGHPUT_QPS_REF),
        stage="closed-16",
    ),
    MetricGate(
        "closed16.latency_p95_s_max",
        "stage",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_CLOSED16_LATENCY_P95_S_REF),
        stage="closed-16",
    ),
    MetricGate(
        "ramp.throughput_qps_min",
        "stage",
        "achieved_rps",
        "value",
        "min",
        _minimum(SERVING_RAMP_THROUGHPUT_QPS_REF),
        stage="ramp-128",
    ),
    MetricGate(
        "ramp.latency_p95_s_max",
        "stage",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_RAMP_LATENCY_P95_S_REF),
        stage="ramp-128",
    ),
    MetricGate(
        "soak.latency_p95_s_max",
        "stage",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_SOAK_LATENCY_P95_S_REF),
        stage="soak-300s",
    ),
    MetricGate(
        "speech_normal.latency_p95_s_max",
        "stage_workload",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_SPEECH_NORMAL_LATENCY_P95_S_REF),
        stage="soak-300s",
        workload="speech_normal",
    ),
    MetricGate(
        "speech_normal.rtf_p95_max",
        "stage_workload",
        "rtf",
        "p95",
        "max",
        _maximum(SERVING_SPEECH_NORMAL_RTF_P95_REF),
        stage="soak-300s",
        workload="speech_normal",
    ),
    MetricGate(
        "rest_stream.ttfa_p95_s_max",
        "stage_workload",
        "ttfa_s",
        "p95",
        "max",
        _maximum(SERVING_REST_STREAM_TTFA_P95_S_REF),
        stage="soak-300s",
        workload="rest_stream",
    ),
    MetricGate(
        "rest_stream.inter_chunk_p95_s_max",
        "stage_workload",
        "inter_chunk_s",
        "p95",
        "max",
        _maximum(SERVING_REST_STREAM_INTER_CHUNK_P95_S_REF),
        stage="soak-300s",
        workload="rest_stream",
    ),
    MetricGate(
        "rest_stream.latency_p95_s_max",
        "stage_workload",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_REST_STREAM_LATENCY_P95_S_REF),
        stage="soak-300s",
        workload="rest_stream",
    ),
    MetricGate(
        "rest_stream.rtf_p95_max",
        "stage_workload",
        "rtf",
        "p95",
        "max",
        _maximum(SERVING_REST_STREAM_RTF_P95_REF),
        stage="soak-300s",
        workload="rest_stream",
    ),
    MetricGate(
        "batch32.latency_p95_s_max",
        "stage_workload",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_BATCH32_LATENCY_P95_S_REF),
        stage="soak-300s",
        workload="batch_32_all_valid",
    ),
    MetricGate(
        "ws_normal.ttfa_p95_s_max",
        "stage_workload",
        "ttfa_s",
        "p95",
        "max",
        _maximum(SERVING_WS_NORMAL_TTFA_P95_S_REF),
        stage="ws-burst-512",
        workload="ws_normal",
    ),
    MetricGate(
        "ws_normal.latency_p95_s_max",
        "stage_workload",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_WS_NORMAL_LATENCY_P95_S_REF),
        stage="ws-burst-512",
        workload="ws_normal",
    ),
    MetricGate(
        "ws_stream.ttfa_p95_s_max",
        "stage_workload",
        "ttfa_s",
        "p95",
        "max",
        _maximum(SERVING_WS_STREAM_TTFA_P95_S_REF),
        stage="ws-burst-512",
        workload="ws_stream_audio",
    ),
    MetricGate(
        "ws_stream.inter_chunk_p95_s_max",
        "stage_workload",
        "inter_chunk_s",
        "p95",
        "max",
        _maximum(SERVING_WS_STREAM_INTER_CHUNK_P95_S_REF),
        stage="ws-burst-512",
        workload="ws_stream_audio",
    ),
    MetricGate(
        "ws_stream.latency_p95_s_max",
        "stage_workload",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_WS_STREAM_LATENCY_P95_S_REF),
        stage="ws-burst-512",
        workload="ws_stream_audio",
    ),
    MetricGate(
        "ws_stream.rtf_p95_max",
        "stage_workload",
        "rtf",
        "p95",
        "max",
        _maximum(SERVING_WS_STREAM_RTF_P95_REF),
        stage="ws-burst-512",
        workload="ws_stream_audio",
    ),
    MetricGate(
        "mixed_burst.throughput_qps_min",
        "stage",
        "achieved_rps",
        "value",
        "min",
        _minimum(SERVING_MIXED_BURST_THROUGHPUT_QPS_REF),
        stage="mixed-burst-512",
    ),
    MetricGate(
        "mixed_burst.latency_p95_s_max",
        "stage",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_MIXED_BURST_LATENCY_P95_S_REF),
        stage="mixed-burst-512",
    ),
    MetricGate(
        "long.prompt_tokens_min",
        "workload",
        "prompt_tokens",
        "min",
        "min",
        _minimum(SERVING_LONG_PROMPT_TOKENS_MIN_REF),
        workload="long_prefill_decode",
    ),
    MetricGate(
        "long.completion_tokens_min",
        "workload",
        "completion_tokens",
        "min",
        "min",
        _minimum(SERVING_LONG_COMPLETION_TOKENS_MIN_REF),
        workload="long_prefill_decode",
    ),
    MetricGate(
        "long.latency_p95_s_max",
        "workload",
        "latency_s",
        "p95",
        "max",
        _maximum(SERVING_LONG_LATENCY_P95_S_REF),
        workload="long_prefill_decode",
    ),
    MetricGate(
        "long.audio_duration_s_min",
        "workload",
        "audio_duration_s",
        "min",
        "min",
        _minimum(SERVING_LONG_AUDIO_DURATION_MIN_S_REF),
        workload="long_prefill_decode",
    ),
    MetricGate(
        "long.output_tok_per_req_s_min",
        "workload",
        "output_tok_per_req_s",
        "value",
        "min",
        _minimum(SERVING_LONG_OUTPUT_TOK_PER_REQ_S_REF),
        workload="long_prefill_decode",
    ),
    MetricGate(
        "long_baseline.latency_s_max",
        "stage_workload",
        "latency_s",
        "max",
        "max",
        _maximum(SERVING_LONG_BASELINE_LATENCY_MAX_S_REF),
        stage="closed-1",
        workload="long_prefill_decode",
    ),
    MetricGate(
        "long_baseline.output_tok_per_req_s_min",
        "stage_workload",
        "output_tok_per_req_s",
        "value",
        "min",
        _minimum(SERVING_LONG_BASELINE_OUTPUT_TOK_PER_REQ_S_REF),
        stage="closed-1",
        workload="long_prefill_decode",
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

    if not torch.cuda.is_available():
        pytest.skip("TTS serving CI requires CUDA")

    configured_root = os.environ.get(OUTPUT_ROOT_ENV)
    run_dir = (
        Path(configured_root).resolve()
        if configured_root
        else tmp_path_factory.mktemp("tts-serving-ci")
    )
    benchmark_dir = run_dir / "benchmark"
    speaker_dir = tmp_path_factory.mktemp("tts-serving-speakers")
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    port = find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    spec_path = _materialize_spec(run_dir, base_url)
    request_timeout_s = load_spec(spec_path).params.timeout_s
    server_env = {
        "PYTHONPATH": str(PROJECT_ROOT),
        "SPEAKER_SAMPLES_DIR": str(speaker_dir),
        "SPEAKER_MAX_UPLOADED": "1000",
    }
    command = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        MODEL_PATH,
        "--model-name",
        MODEL_PATH,
        "--allowed-local-media-path",
        str(REFERENCE_AUDIO_ROOT),
        "--port",
        str(port),
    ]
    process = start_server_from_cmd(
        command,
        run_dir / "server.log",
        port,
        timeout=MODEL_PRESET.startup_timeout,
        env=server_env,
        tee=True,
        strip_proxy=True,
    )
    try:
        yield ServingRun(
            base_url=base_url,
            run_dir=run_dir,
            benchmark_dir=benchmark_dir,
            spec_path=spec_path,
            request_timeout_s=request_timeout_s,
        )
    finally:
        stop_server(process)


def _run_benchmark(run: ServingRun) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_tts_serving",
        "--spec",
        str(run.spec_path),
        "--out",
        str(run.benchmark_dir),
    ]
    started = time.perf_counter()
    with (run.run_dir / "benchmark.stdout.log").open("w", encoding="utf-8") as stdout:
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env={**no_proxy_env(), "PYTHONPATH": str(PROJECT_ROOT)},
                stdout=stdout,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=BENCHMARK_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            stdout.write(f"\nbenchmark killed after {BENCHMARK_TIMEOUT_S}s timeout\n")
            completed = subprocess.CompletedProcess(
                command, returncode=BENCHMARK_TIMEOUT_RETURNCODE
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


def _gate_summary(report: dict, gate: MetricGate) -> dict | None:
    metrics = report.get("metrics", {})
    if gate.source == "stage":
        summary = metrics.get("by_stage", {}).get(gate.stage)
    elif gate.source == "workload":
        summary = metrics.get("by_workload", {}).get(gate.workload)
    else:
        summary = (
            metrics.get("by_stage_and_workload", {})
            .get(gate.stage, {})
            .get(gate.workload)
        )
    return summary if isinstance(summary, dict) else None


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
    pending = sorted(gate.key for gate in METRIC_GATES if gate.threshold is None)
    threshold_checks.check(
        not pending,
        f"serving thresholds require calibration: {pending}",
    )

    for gate in METRIC_GATES:
        summary = _gate_summary(report, gate)
        if summary is None:
            measurement_checks.fail(f"missing result summary for {gate.key}")
            continue

        samples = summary.get("successful_request_count")
        measurement_checks.check(
            isinstance(samples, int) and samples > 0,
            f"{gate.key} has no successful samples",
        )
        if gate.metric != "achieved_rps":
            metric_samples = summary.get("metric_sample_counts", {}).get(gate.metric)
            measurement_checks.check(
                metric_samples == samples,
                f"{gate.key} metric samples={metric_samples!r}, "
                f"successful samples={samples!r}",
            )

        value = _metric_value(summary, gate)
        measurement_checks.check(value is not None, f"{gate.key} is missing")
        threshold = gate.threshold
        if value is None or threshold is None:
            continue
        if gate.direction == "min":
            threshold_checks.check(
                value >= threshold,
                f"{gate.key}={value} < {threshold}",
            )
        else:
            threshold_checks.check(
                value <= threshold,
                f"{gate.key}={value} > {threshold}",
            )


def _get_json(url: str, timeout_s: int) -> dict:
    opener = build_opener(ProxyHandler({}))
    with opener.open(url, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected JSON object from {url}")
    return payload


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
        results_path.is_file(), f"missing benchmark report: {results_path}"
    )
    if results_path.is_file():
        try:
            report = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            benchmark_checks.fail(f"could not read benchmark report: {exc}")
        else:
            overall = report.get("overall", {})
            metrics = report.get("metrics", {})
            benchmark_checks.check(
                report.get("harness_status") == "ok", "benchmark harness failed"
            )
            benchmark_checks.check(
                overall.get("passed") is True, "benchmark did not pass"
            )
            benchmark_checks.check(
                overall.get("failed") == 0, "benchmark contains failures"
            )
            benchmark_checks.check(
                overall.get("load_generation_valid") is True,
                "load generation was invalid",
            )
            benchmark_checks.check(
                overall.get("coverage_contract_valid") is True,
                "benchmark coverage was incomplete",
            )
            benchmark_checks.check(
                not report.get("failures"), "scenario failures were reported"
            )
            benchmark_checks.check(
                not report.get("unsupported_contracts"),
                "unsupported contracts were reported",
            )
            benchmark_checks.check(
                not report.get("coverage_failures"),
                "coverage failures were reported",
            )
            benchmark_checks.check(
                not metrics.get("admission_status_counts"),
                "admission failures were reported",
            )
            _check_performance(report, measurement_checks, threshold_checks)

    try:
        health = _get_json(
            f"{serving_run.base_url}/health",
            serving_run.request_timeout_s,
        )
    except Exception as exc:
        benchmark_checks.fail(f"post-benchmark health probe failed: {exc}")
    else:
        benchmark_checks.check(
            str(health.get("status", "")).lower() == "healthy",
            f"server is unhealthy after benchmark: {health}",
        )

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
    benchmark_checks.assert_all()
    measurement_checks.assert_all()
    threshold_checks.assert_all()
