# SPDX-License-Identifier: Apache-2.0
"""TTS serving benchmark.

The harness follows the benchmark platform contract:

- read /etc/benchmark/spec.json by default
- write outputs under /var/benchmark/out by default
- exit 0 when the harness ran, even when the server/model failed
- exit non-zero for invalid specs, artifact failures, or unhandled harness errors

Docker:
    docker build -f benchmarks/tts_serving/Dockerfile \
      -t sglang-omni-tts-serving-benchmark .
    docker run --rm \
      --user "$(id -u):$(id -g)" \
      -v "$PWD/spec.json:/etc/benchmark/spec.json:ro" \
      -v "$PWD/out:/var/benchmark/out" \
      sglang-omni-tts-serving-benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from collections.abc import Iterable
from pathlib import Path

import aiohttp

from benchmarks.tts_serving.artifacts import (
    ArtifactError,
    prepare_output_dir,
    write_artifacts,
    write_harness_log,
)
from benchmarks.tts_serving.http_client import run_http_scenario
from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.report import build_results_report
from benchmarks.tts_serving.scenarios import Scenario, build_scenarios
from benchmarks.tts_serving.sdk_client import run_sdk_scenario
from benchmarks.tts_serving.spec import BenchmarkSpec, LoadStage, SpecError, load_spec
from benchmarks.tts_serving.ws_client import run_ws_scenario

LOAD_GENERATOR_LAGGED_THRESHOLD_S = 1.0
DEFAULT_SPEC_PATH = "/etc/benchmark/spec.json"
DEFAULT_OUT_DIR = "/var/benchmark/out"
SUMMARY_LINE_WIDTH = 96
SUMMARY_LABEL_WIDTH = 32


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TTS serving benchmark harness.")
    parser.add_argument("--spec", default=DEFAULT_SPEC_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    return parser


async def _run_benchmark(
    spec: BenchmarkSpec,
    scenarios: list[Scenario],
    harness_log: list[str],
) -> list[ScenarioResult]:
    timeout = aiohttp.ClientTimeout(total=spec.params.timeout_s)
    headers = _auth_headers(spec)
    connector = aiohttp.TCPConnector(limit=_connector_limit(spec))
    async with aiohttp.ClientSession(
        timeout=timeout,
        headers=headers,
        connector=connector,
    ) as session:
        results: list[ScenarioResult] = []
        for stage in spec.params.load_stages:
            stage_scenarios = [
                scenario for scenario in scenarios if scenario.stage_id == stage.id
            ]
            results.extend(
                await _run_stage(
                    session,
                    spec,
                    stage,
                    stage_scenarios,
                    harness_log,
                )
            )
        return results


async def _run_stage(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    stage: LoadStage,
    scenarios: list[Scenario],
    harness_log: list[str],
) -> list[ScenarioResult]:
    if len(scenarios) > stage.request_count:
        harness_log.append(
            f"stage={stage.id} scheduled {len(scenarios)} scenarios although "
            f"request_count={stage.request_count}; required benchmark contracts "
            "are never truncated"
        )
    if stage.mode == "closed_loop":
        return await _run_closed_loop_stage(
            session, spec, stage, scenarios, harness_log
        )
    return await _run_scheduled_stage(session, spec, stage, scenarios, harness_log)


async def _run_closed_loop_stage(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    stage: LoadStage,
    scenarios: list[Scenario],
    harness_log: list[str],
) -> list[ScenarioResult]:
    scenario_iter = iter(scenarios)
    results: list[ScenarioResult] = []
    started = time.perf_counter()
    active_requests = 0
    peak_inflight = 0

    async def worker() -> None:
        nonlocal active_requests, peak_inflight
        for scenario in scenario_iter:
            actual_start = time.perf_counter()
            active_requests += 1
            peak_inflight = max(peak_inflight, active_requests)
            try:
                result = await _run_one_scenario(session, spec, scenario)
            finally:
                active_requests -= 1
            _attach_schedule_metadata(
                result,
                stage=stage,
                planned_start=actual_start,
                actual_start=actual_start,
                peak_inflight=peak_inflight,
            )
            results.append(result)

    await asyncio.gather(
        *(worker() for _ in range(min(stage.max_concurrency, len(scenarios))))
    )
    for result in results:
        result.peak_inflight = peak_inflight
    harness_log.append(
        f"stage={stage.id} mode={stage.mode} completed {len(results)} scenarios "
        f"at concurrency={stage.max_concurrency} in {time.perf_counter() - started:.3f}s"
    )
    return results


async def _run_scheduled_stage(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    stage: LoadStage,
    scenarios: list[Scenario],
    harness_log: list[str],
) -> list[ScenarioResult]:
    stage_start = time.perf_counter()
    offsets = _planned_offsets(stage, len(scenarios), seed=spec.seed)
    active_requests = 0
    peak_inflight = 0

    async def run_planned(scenario: Scenario, offset: float) -> ScenarioResult:
        nonlocal active_requests
        planned_start = stage_start + offset
        actual_start = time.perf_counter()
        try:
            result = await _run_one_scenario(session, spec, scenario)
        finally:
            active_requests -= 1
        _attach_schedule_metadata(
            result,
            stage=stage,
            planned_start=planned_start,
            actual_start=actual_start,
            generator_lag=max(0.0, actual_start - planned_start),
        )
        return result

    started = time.perf_counter()
    pending: set[asyncio.Task[ScenarioResult]] = set()
    results: list[ScenarioResult] = []
    peak_pending_tasks = 0
    scheduled_task_count = 0
    for scenario, offset in zip(scenarios, offsets, strict=True):
        planned_start = stage_start + offset
        delay_s = planned_start - time.perf_counter()
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        done = {task for task in pending if task.done()}
        if done:
            pending.difference_update(done)
            results.extend(_harvest_completed_tasks(done))
        scheduled_task_count += 1
        if active_requests >= stage.max_concurrency:
            actual_start = time.perf_counter()
            results.append(
                _load_generator_saturated_result(
                    scenario,
                    stage=stage,
                    planned_start=planned_start,
                    actual_start=actual_start,
                    active_requests=active_requests,
                    generator_lag=max(0.0, actual_start - planned_start),
                )
            )
            continue
        active_requests += 1
        peak_inflight = max(peak_inflight, active_requests)
        pending.add(asyncio.create_task(run_planned(scenario, offset)))
        peak_pending_tasks = max(peak_pending_tasks, len(pending))
    if pending:
        results.extend(await _gather_pending_tasks(pending))
    for result in results:
        result.peak_inflight = peak_inflight
    generator_lag_s = [
        result.generator_lag_s
        for result in results
        if result.generator_lag_s is not None
    ]
    max_generator_lag_s = max(generator_lag_s, default=0.0)
    load_generator_lagged = max_generator_lag_s > LOAD_GENERATOR_LAGGED_THRESHOLD_S
    for result in results:
        result.peak_pending_tasks = peak_pending_tasks
        result.scheduled_task_count = scheduled_task_count
        result.load_generator_lagged = load_generator_lagged
    harness_log.append(
        f"stage={stage.id} mode={stage.mode} completed {len(results)} scenarios "
        f"with configured_max_concurrency={stage.max_concurrency} "
        f"peak_inflight={peak_inflight} in {time.perf_counter() - started:.3f}s "
        "with scheduled arrivals emitted independently of request completions "
        f"(generator_lag_max={max_generator_lag_s:.6f}s, "
        f"peak_pending_tasks={peak_pending_tasks}, "
        f"load_generator_lagged={load_generator_lagged})"
    )
    return results


def _load_generator_saturated_result(
    scenario: Scenario,
    *,
    stage: LoadStage,
    planned_start: float,
    actual_start: float,
    active_requests: int,
    generator_lag: float,
) -> ScenarioResult:
    result = ScenarioResult(
        scenario_id=scenario.id,
        endpoint=scenario.endpoint,
        category=scenario.category,
        capability_key=scenario.capability_key,
        workload=scenario.workload,
        expected_success=scenario.expect_success,
        response_format=str(scenario.payload.get("response_format", "")) or None,
        batch_size=scenario.planned_metadata.get("batch_size"),
        status="load_generator_saturated",
        success=False,
        capability="fail",
        error_class="load_generator_saturation",
        error=(
            "scheduled arrival could not start because benchmark client "
            f"reached max_concurrency={stage.max_concurrency} "
            f"(active_requests={active_requests})"
        ),
        load_generator_saturated=True,
    )
    _attach_schedule_metadata(
        result,
        stage=stage,
        planned_start=planned_start,
        actual_start=actual_start,
        peak_inflight=active_requests,
        generator_lag=generator_lag,
    )
    return result


async def _run_one_scenario(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
) -> ScenarioResult:
    try:
        if scenario.method == "WS":
            return await run_ws_scenario(session, spec, scenario)
        if scenario.method == "SDK":
            return await run_sdk_scenario(spec, scenario)
        return await run_http_scenario(session, spec, scenario)
    except Exception as exc:
        return _scenario_exception_result(scenario, exc)


def _harvest_completed_tasks(
    tasks: set[asyncio.Task[ScenarioResult]],
) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for task in tasks:
        try:
            results.append(task.result())
        except Exception as exc:
            results.append(_task_exception_result(exc))
    return results


async def _gather_pending_tasks(
    tasks: set[asyncio.Task[ScenarioResult]],
) -> list[ScenarioResult]:
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[ScenarioResult] = []
    for raw_result in raw_results:
        if isinstance(raw_result, ScenarioResult):
            results.append(raw_result)
        elif isinstance(raw_result, Exception):
            results.append(_task_exception_result(raw_result))
    return results


def _scenario_exception_result(scenario: Scenario, exc: Exception) -> ScenarioResult:
    result = ScenarioResult(
        scenario_id=scenario.id,
        endpoint=scenario.endpoint,
        category=scenario.category,
        capability_key=scenario.capability_key,
        workload=scenario.workload,
        expected_success=scenario.expect_success,
        response_format=str(scenario.payload.get("response_format", "")) or None,
        batch_size=scenario.planned_metadata.get("batch_size"),
        status="failed",
        success=False,
        capability="fail",
        error_type=exc.__class__.__name__,
        error_class="client_error",
        error=f"benchmark scenario failed before response classification: {exc}",
    )
    return result


def _task_exception_result(exc: Exception) -> ScenarioResult:
    return ScenarioResult(
        scenario_id="benchmark-task-exception",
        endpoint="unknown",
        category="harness_task",
        status="failed",
        success=False,
        capability="fail",
        error_type=exc.__class__.__name__,
        error_class="client_error",
        error=f"benchmark task failed before scenario result was recorded: {exc}",
    )


def _attach_schedule_metadata(
    result: ScenarioResult,
    *,
    stage: LoadStage,
    planned_start: float,
    actual_start: float,
    peak_inflight: int | None = None,
    generator_lag: float | None = None,
) -> None:
    result.stage_id = stage.id
    result.load_mode = stage.mode
    result.load_concurrency = stage.max_concurrency
    result.configured_max_concurrency = stage.max_concurrency
    result.peak_inflight = peak_inflight
    result.planned_start_s = planned_start
    result.actual_start_s = actual_start
    result.queue_wait_s = max(0.0, actual_start - planned_start)
    result.generator_lag_s = generator_lag


def _planned_offsets(stage: LoadStage, request_count: int, *, seed: int) -> list[float]:
    if request_count <= 0:
        return []
    if stage.mode == "burst":
        return [0.0] * request_count
    if stage.mode == "ramp":
        return _ramp_offsets(stage, request_count, seed=seed)
    if stage.mode == "soak":
        assert stage.duration_s is not None
        if request_count == 1:
            return [0.0]
        if stage.arrival_distribution == "poisson":
            return _duration_conditioned_poisson_offsets(
                stage.duration_s, request_count, seed=seed, stage_id=stage.id
            )
        return _duration_spaced_offsets(stage.duration_s, request_count)
    if stage.arrival_distribution == "poisson":
        rng = random.Random(f"{seed}:{stage.id}:arrival")
        elapsed = 0.0
        offsets: list[float] = []
        for _ in range(request_count):
            offsets.append(elapsed)
            elapsed += rng.expovariate(stage.request_rate)
        return offsets
    return [index / stage.request_rate for index in range(request_count)]


def _connector_limit(spec: BenchmarkSpec) -> int:
    if any(stage.mode != "closed_loop" for stage in spec.params.load_stages):
        return 0
    return max(spec.params.max_concurrency * 2, 8)


def _ramp_offsets(stage: LoadStage, request_count: int, *, seed: int) -> list[float]:
    start_rate = stage.start_request_rate or stage.request_rate
    end_rate = stage.request_rate
    elapsed = 0.0
    offsets: list[float] = []
    rng = random.Random(f"{seed}:{stage.id}:ramp-arrival")
    for index in range(request_count):
        offsets.append(elapsed)
        position = index / max(request_count - 1, 1)
        current_rate = start_rate + (end_rate - start_rate) * position
        if stage.arrival_distribution == "poisson":
            elapsed += rng.expovariate(current_rate)
        else:
            elapsed += 1.0 / current_rate
    return offsets


def _duration_spaced_offsets(duration_s: float, request_count: int) -> list[float]:
    step = duration_s / float(request_count)
    return [index * step for index in range(request_count)]


def _duration_conditioned_poisson_offsets(
    duration_s: float, request_count: int, *, seed: int, stage_id: str
) -> list[float]:
    rng = random.Random(f"{seed}:{stage_id}:soak-arrival")
    return sorted(rng.uniform(0.0, duration_s) for _ in range(request_count))


def _auth_headers(spec: BenchmarkSpec) -> dict[str, str]:
    if not spec.auth.api_key_env:
        return {}
    token = os.environ.get(spec.auth.api_key_env)
    if not token:
        raise RuntimeError(
            f"auth environment variable is not set: {spec.auth.api_key_env}"
        )
    return {"Authorization": f"Bearer {token}"}


def _print_results_summary(report: dict, out_dir: Path) -> None:
    overall = report.get("overall", {})
    config = report.get("config", {})
    metrics = report.get("metrics", {})
    latency = metrics.get("latency_s", {}) if isinstance(metrics, dict) else {}
    ttfa = metrics.get("ttfa_s", {}) if isinstance(metrics, dict) else {}
    queue_wait = metrics.get("queue_wait_s", {}) if isinstance(metrics, dict) else {}
    generator_lag = (
        metrics.get("generator_lag_s", {}) if isinstance(metrics, dict) else {}
    )
    rtf = metrics.get("rtf", {}) if isinstance(metrics, dict) else {}
    line_width = SUMMARY_LINE_WIDTH
    print(f"\n{'=' * line_width}")
    print(f"{'TTS Serving Benchmark Result':^{line_width}}")
    print(f"{'=' * line_width}")
    _print_summary_row("Model", config.get("model_name", "N/A"))
    _print_summary_row("Base URL", config.get("base_url", "N/A"))
    _print_summary_row("Profile", config.get("profile", "N/A"))
    _print_summary_row("Run ID", config.get("run_id") or "N/A")
    _print_summary_row("Seed", config.get("seed"))
    _print_summary_row("Passed", overall.get("passed"))
    _print_summary_row("Harness status", report.get("harness_status"))
    _print_summary_row(
        "Scenarios",
        (
            f"{overall.get('succeeded')}/{overall.get('total')} passed, "
            f"{overall.get('failed')} failed"
        ),
    )
    _print_summary_row("Traffic scenarios", overall.get("traffic_total"))
    _print_summary_row("Coverage valid", overall.get("coverage_contract_valid"))
    _print_summary_row("Load generation valid", overall.get("load_generation_valid"))
    _print_summary_row(
        "Status counts",
        _format_counts(metrics.get("status_counts", {})),
    )
    _print_summary_row(
        "HTTP status counts",
        _format_counts(metrics.get("http_status_counts", {})),
    )
    _print_summary_row(
        "Error classes",
        _format_counts(metrics.get("error_class_counts", {})),
    )

    _print_summary_section("Performance")
    _print_summary_row("Latency seconds", _format_summary(latency))
    _print_summary_row("TTFA seconds", _format_summary(ttfa))
    _print_summary_row("RTF", _format_summary(rtf))
    _print_summary_row("Queue wait seconds", _format_summary(queue_wait))
    _print_summary_row("Generator lag seconds", _format_summary(generator_lag))

    _print_summary_section("Load Generator")
    _print_summary_row("Peak inflight", metrics.get("peak_inflight"))
    _print_summary_row("Peak pending tasks", metrics.get("peak_pending_tasks"))
    _print_summary_row(
        "Load generator issue",
        metrics.get("load_generation_error") or "none",
    )

    _print_endpoint_summary(metrics.get("by_endpoint", {}))
    _print_operation_summary(metrics.get("by_operation", {}))
    _print_stage_summary(config, metrics.get("by_stage", {}))
    _print_failure_summary(report)

    _print_summary_section("Artifacts")
    _print_summary_row("Results JSON", out_dir / "results.json")
    _print_summary_row("Manifest JSON", out_dir / "manifest.json")
    _print_summary_row("Raw results JSONL", out_dir / "raw_results.jsonl")
    print(f"{'=' * line_width}")


def _print_summary_section(title: str) -> None:
    print(f"{title:-^{SUMMARY_LINE_WIDTH}}")


def _print_summary_row(label: str, value: object) -> None:
    print(f"  {label + ':':<{SUMMARY_LABEL_WIDTH}} {_format_value(value)}")


def _format_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def _format_summary(summary: object) -> str:
    if not isinstance(summary, dict) or not summary:
        return "N/A"
    fields = (
        ("mean", "mean"),
        ("p50", "p50"),
        ("p95", "p95"),
        ("p99", "p99"),
        ("p99_9", "p99.9"),
        ("max", "max"),
    )
    return " ".join(
        f"{label}={_format_value(summary[key])}"
        for key, label in fields
        if summary.get(key) is not None
    )


def _format_counts(counts: object) -> str:
    if not isinstance(counts, dict) or not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _summary_column_width(header: str, values: Iterable[object], minimum: int) -> int:
    return max(len(header), minimum, *(len(str(value)) for value in values))


def _print_endpoint_summary(by_endpoint: object) -> None:
    if not isinstance(by_endpoint, dict) or not by_endpoint:
        return
    summaries = sorted(
        by_endpoint.items(),
        key=lambda item: (
            int(item[1].get("total", 0)) if isinstance(item[1], dict) else 0
        ),
        reverse=True,
    )
    endpoint_width = _summary_column_width(
        "Endpoint",
        (endpoint for endpoint, summary in summaries if isinstance(summary, dict)),
        16,
    )
    _print_summary_section("Endpoint Summary")
    print(
        "  "
        f"{'Endpoint':<{endpoint_width}}"
        f"{'Total':>8}"
        f"{'Pass':>8}"
        f"{'Fail':>8}"
        f"{'Lat p95':>12}"
        f"{'TTFA p95':>12}"
        f"{'RTF mean':>12}"
    )
    for endpoint, summary in summaries:
        if not isinstance(summary, dict):
            continue
        print(
            "  "
            f"{str(endpoint):<{endpoint_width}}"
            f"{summary.get('total', 0):>8}"
            f"{summary.get('succeeded', 0):>8}"
            f"{summary.get('failed', 0):>8}"
            f"{_summary_metric(summary, 'latency_s', 'p95'):>12}"
            f"{_summary_metric(summary, 'ttfa_s', 'p95'):>12}"
            f"{_summary_metric(summary, 'rtf', 'mean'):>12}"
        )


def _print_operation_summary(by_operation: object) -> None:
    if not isinstance(by_operation, dict) or not by_operation:
        return
    summaries = sorted(
        by_operation.items(),
        key=lambda item: (
            int(item[1].get("total", 0)) if isinstance(item[1], dict) else 0
        ),
        reverse=True,
    )
    operation_width = _summary_column_width(
        "Operation",
        (operation for operation, summary in summaries if isinstance(summary, dict)),
        28,
    )
    _print_summary_section("Operation Summary")
    print(
        "  "
        f"{'Operation':<{operation_width}}"
        f"{'Total':>8}"
        f"{'Pass':>8}"
        f"{'Fail':>8}"
        f"{'Lat p95':>12}"
        f"{'RTF mean':>12}"
    )
    for operation, summary in summaries:
        if not isinstance(summary, dict):
            continue
        print(
            "  "
            f"{str(operation):<{operation_width}}"
            f"{summary.get('total', 0):>8}"
            f"{summary.get('succeeded', 0):>8}"
            f"{summary.get('failed', 0):>8}"
            f"{_summary_metric(summary, 'latency_s', 'p95'):>12}"
            f"{_summary_metric(summary, 'rtf', 'mean'):>12}"
        )


def _print_stage_summary(config: dict, by_stage: object) -> None:
    if not isinstance(by_stage, dict) or not by_stage:
        return
    stage_ids = [
        str(stage.get("id"))
        for stage in config.get("load_stages", [])
        if isinstance(stage, dict) and stage.get("id") in by_stage
    ]
    stage_ids.extend(stage_id for stage_id in by_stage if stage_id not in stage_ids)
    stage_width = _summary_column_width("Stage", stage_ids, 24)
    _print_summary_section("Load Stage Summary")
    print(
        "  "
        f"{'Stage':<{stage_width}}"
        f"{'Total':>8}"
        f"{'Fail':>8}"
        f"{'Peak':>8}"
        f"{'RPS':>10}"
        f"{'Lat p95':>12}"
        f"{'TTFA p95':>12}"
    )
    for stage_id in stage_ids:
        summary = by_stage.get(stage_id)
        if not isinstance(summary, dict):
            continue
        print(
            "  "
            f"{str(stage_id):<{stage_width}}"
            f"{summary.get('total', 0):>8}"
            f"{summary.get('failed', 0):>8}"
            f"{_format_value(summary.get('peak_inflight')):>8}"
            f"{_format_value(summary.get('achieved_rps')):>10}"
            f"{_summary_metric(summary, 'latency_s', 'p95'):>12}"
            f"{_summary_metric(summary, 'ttfa_s', 'p95'):>12}"
        )


def _summary_metric(summary: dict, metric_name: str, value_name: str) -> str:
    metric = summary.get(metric_name)
    if not isinstance(metric, dict):
        return "N/A"
    return _format_value(metric.get(value_name))


def _print_failure_summary(report: dict) -> None:
    failures = report.get("failures", [])
    coverage_failures = report.get("coverage_failures", [])
    unsupported_contracts = report.get("unsupported_contracts", [])
    if not failures and not coverage_failures and not unsupported_contracts:
        return
    _print_summary_section("Failures")
    _print_summary_row("Coverage failures", len(coverage_failures))
    _print_summary_row("Unsupported contracts", len(unsupported_contracts))
    _print_summary_row("Recorded failures", len(failures))
    if not isinstance(failures, list):
        return
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        scenario_id = str(failure.get("scenario_id", "unknown"))
        status = str(failure.get("status", "unknown"))
        error_class = str(failure.get("error_class") or "unknown")
        error = str(failure.get("error") or "")
        print(f"  - {scenario_id}: {status}, {error_class}, {error}")


def main() -> int:
    args = _build_arg_parser().parse_args()
    harness_log: list[str] = []
    try:
        spec = load_spec(args.spec)
        out_dir = prepare_output_dir(args.out)
    except (SpecError, ArtifactError) as exc:
        print(f"benchmark harness failed: {exc}")
        return 2

    scenarios = build_scenarios(spec)
    stage_request_total = sum(stage.request_count for stage in spec.params.load_stages)
    harness_log.append(
        f"loaded spec={Path(args.spec)} profile={spec.params.profile} "
        f"stage_requests={stage_request_total} scenarios={len(scenarios)} "
        f"load_stages={[stage.id for stage in spec.params.load_stages]}"
    )
    try:
        results = asyncio.run(_run_benchmark(spec, scenarios, harness_log))
        report = build_results_report(spec, results, scenarios=scenarios)
        write_artifacts(out_dir, spec, scenarios, results, report)
        write_harness_log(out_dir, harness_log)
        _print_results_summary(report, out_dir)
    except ArtifactError as exc:
        print(f"benchmark harness failed: {exc}")
        return 2
    except Exception as exc:
        harness_log.append(f"unhandled harness error: {exc.__class__.__name__}: {exc}")
        report = build_results_report(
            spec,
            [],
            scenarios=scenarios,
            harness_status="error",
            harness_error=f"{exc.__class__.__name__}: {exc}",
        )
        try:
            write_artifacts(out_dir, spec, scenarios, [], report)
            write_harness_log(out_dir, harness_log)
        except ArtifactError:
            pass
        print(f"benchmark harness failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
