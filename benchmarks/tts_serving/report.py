# SPDX-License-Identifier: Apache-2.0
"""Report aggregation for the TTS serving benchmark."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from benchmarks.tts_serving import voice_contracts
from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.scenarios import (
    BATCH_OVERSIZED_SIZE,
    BATCH_SIZES,
    LENGTH_EXTREME_TEXTS,
    MALFORMED_CASE_NAMES,
    MULTILINGUAL_TEXTS,
    REFERENCE_FAILURES,
    RESPONSE_FORMATS,
    SCENARIO_SCHEMA_VERSION,
    SDK_RESPONSE_FORMATS,
    SPEED_BOUNDARY_VALUES,
    TASK_TYPES,
    VOICE_NEAR_LIMIT_FORMATS,
    VOICE_NEAR_LIMIT_GENERATED_FORMATS,
    VOICE_UPLOAD_REJECT_FORMATS,
    VOICE_UPLOAD_SUCCESS_FORMATS,
    WEBSOCKET_SPLIT_GRANULARITIES,
    Scenario,
    scenario_set_hash,
)
from benchmarks.tts_serving.spec import BenchmarkSpec, redact_sensitive_metadata


def build_results_report(
    spec: BenchmarkSpec,
    results: list[ScenarioResult],
    *,
    scenarios: list[Scenario] | None = None,
    harness_status: str = "ok",
    harness_error: str | None = None,
) -> dict[str, Any]:
    total = len(results)
    traffic_total = sum(
        1 for result in results if result.category != "capability_probe"
    )
    capability_total = total - traffic_total
    succeeded = sum(1 for result in results if _result_passed(spec, result))
    failed = total - succeeded
    operation_capabilities = _operation_capabilities(results)
    capabilities = _endpoint_capabilities(results, operation_capabilities)
    category_counts = Counter(result.category for result in results)
    status_counts = Counter(result.status for result in results)
    latencies = [r.latency_s for r in results if r.latency_s > 0]
    ttfas = [r.ttfa_s for r in results if r.ttfa_s is not None]
    rtfs = [r.rtf for r in results if r.rtf > 0]
    queue_waits = [r.queue_wait_s for r in results if r.queue_wait_s is not None]
    generator_lags = [
        r.generator_lag_s for r in results if r.generator_lag_s is not None
    ]
    load_generation_valid = not any(
        result.load_generator_lagged or result.load_generator_saturated
        for result in results
    )
    planned_scenarios = scenarios or []
    executed_scenarios = _executed_scenarios(planned_scenarios, results)
    planned_voice_upload_coverage = _voice_upload_coverage(planned_scenarios)
    voice_upload_coverage = _voice_upload_coverage(executed_scenarios)
    planned_coverage_matrix = _coverage_matrix(
        spec, planned_scenarios, planned_voice_upload_coverage
    )
    coverage_matrix = _coverage_matrix(spec, executed_scenarios, voice_upload_coverage)
    coverage_failures = _coverage_failures(
        spec,
        executed_scenarios,
        voice_upload_coverage,
        coverage_matrix,
    )
    coverage_contract_valid = not coverage_failures
    passed = (
        harness_status == "ok"
        and total > 0
        and load_generation_valid
        and coverage_contract_valid
        and _is_benchmark_passed(spec, results, capabilities)
    )
    return {
        "schema_version": 2,
        "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
        "scenario_set_hash": scenario_set_hash(scenarios) if scenarios else None,
        "harness_status": harness_status,
        "harness_error": harness_error,
        "overall": {
            "passed": passed,
            "total": total,
            "traffic_total": traffic_total,
            "capability_probe_total": capability_total,
            "succeeded": succeeded,
            "failed": failed,
            "load_generation_valid": load_generation_valid,
            "coverage_contract_valid": coverage_contract_valid,
        },
        "config": {
            "test_type": spec.test_type,
            "base_url": spec.base_url,
            "model_name": spec.model_name,
            "run_id": spec.run_id,
            "seed": spec.seed,
            "platform_metadata": redact_sensitive_metadata(spec.platform_metadata),
            "provider_label": spec.params.provider_label,
            "implementation_label": spec.params.implementation_label,
            "profile": spec.params.profile,
            "total_requests": spec.params.total_requests,
            "stage_request_total": sum(
                stage.request_count for stage in spec.params.load_stages
            ),
            "planned_scenario_total": len(planned_scenarios),
            "executed_coverage_scenario_total": len(executed_scenarios),
            "max_concurrency": spec.params.max_concurrency,
            "concurrency_levels": list(
                spec.params.concurrency_levels or (spec.params.max_concurrency,)
            ),
            "request_rate": (
                "inf"
                if spec.params.request_rate == float("inf")
                else spec.params.request_rate
            ),
            "timeout_s": spec.params.timeout_s,
            "enabled_endpoints": list(spec.params.enabled_endpoints),
            "voice_cache_pressure_voice_count": spec.params.voice_cache_pressure_voice_count,
            "voice_speaker_cap_count": spec.params.voice_speaker_cap_count,
            "speaker_max_uploaded": spec.params.speaker_max_uploaded,
            "voice_upload_coverage": voice_upload_coverage,
            "load_stages": [stage.to_json() for stage in spec.params.load_stages],
        },
        "capabilities": capabilities,
        "operation_capabilities": operation_capabilities,
        "metrics": {
            "latency_s": _summary(latencies),
            "ttfa_s": _summary(ttfas),
            "queue_wait_s": _summary(queue_waits),
            "generator_lag_s": _summary(generator_lags),
            "peak_inflight": max(
                (result.peak_inflight for result in results if result.peak_inflight),
                default=None,
            ),
            "peak_pending_tasks": max(
                (
                    result.peak_pending_tasks
                    for result in results
                    if result.peak_pending_tasks
                ),
                default=None,
            ),
            "load_generator_lagged": any(
                result.load_generator_lagged for result in results
            ),
            "load_generator_saturated": any(
                result.load_generator_saturated for result in results
            ),
            "load_generation_error": _load_generation_error(results),
            "rtf": _summary(rtfs),
            "rtf_sampled_formats": ["wav", "pcm"],
            "rtf_unsupported_format_counts": _rtf_unsupported_format_counts(results),
            "rtf_note": (
                "RTF is computed only when audio duration can be derived from WAV "
                "headers or raw PCM byte counts; compressed responses are excluded."
            ),
            "status_counts": dict(status_counts),
            "http_status_counts": dict(
                Counter(
                    str(result.http_status)
                    for result in results
                    if result.http_status is not None
                )
            ),
            "admission_status_counts": _admission_status_counts(results),
            "error_class_counts": dict(
                Counter(result.error_class for result in results if result.error_class)
            ),
            "category_counts": dict(category_counts),
            "endpoint_mix": dict(Counter(result.endpoint for result in results)),
            "by_category": _by_category(spec, results),
            "by_stage": _by_stage(spec, results),
            "by_endpoint": _by_endpoint(spec, results),
            "by_operation": _by_operation(spec, results),
            "by_configured_concurrency": _by_configured_concurrency(spec, results),
            "by_peak_inflight": _by_peak_inflight(spec, results),
        },
        "failures": [
            result.to_json() for result in results if not _result_passed(spec, result)
        ][:100],
        "unsupported_contracts": _unsupported_contract_summary(
            results,
            scenarios or [],
        ),
        "coverage_failures": coverage_failures,
        "coverage_matrix": coverage_matrix,
        "planned_coverage_matrix": planned_coverage_matrix,
    }


def _is_benchmark_passed(
    spec: BenchmarkSpec,
    results: list[ScenarioResult],
    capabilities: dict[str, str],
) -> bool:
    return all(_result_passed(spec, result) for result in results) and all(
        status != "fail" for status in capabilities.values()
    )


def _result_passed(spec: BenchmarkSpec, result: ScenarioResult) -> bool:
    if result.expected_success:
        return result.success
    return result.status == "expected_error"


def _executed_scenarios(
    scenarios: list[Scenario],
    results: list[ScenarioResult],
) -> list[Scenario]:
    executable_result_ids = {
        result.scenario_id
        for result in results
        if _result_reached_contract_classifier(result)
    }
    return [scenario for scenario in scenarios if scenario.id in executable_result_ids]


def _result_reached_contract_classifier(result: ScenarioResult) -> bool:
    if result.status in {"load_generator_saturated", "transport_error"}:
        return False
    if result.load_generator_saturated:
        return False
    if result.error_class in {"transport_error", "client_error"}:
        return False
    return True


def _operation_capabilities(results: list[ScenarioResult]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for result in results:
        if result.capability:
            key = result.capability_key or result.endpoint
            grouped[key].append(result.capability)
    return {
        key: _roll_up_capability(statuses) for key, statuses in sorted(grouped.items())
    }


def _endpoint_capabilities(
    results: list[ScenarioResult],
    operation_capabilities: dict[str, str],
) -> dict[str, str]:
    endpoint_operations: dict[str, set[str]] = defaultdict(set)
    for result in results:
        key = result.capability_key or result.endpoint
        if key in operation_capabilities:
            endpoint_operations[result.endpoint].add(key)
    return {
        endpoint: _roll_up_endpoint_capability(
            [operation_capabilities[key] for key in sorted(keys)]
        )
        for endpoint, keys in sorted(endpoint_operations.items())
    }


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "p99_9": None,
            "max": None,
        }
    values_sorted = sorted(values)
    return {
        "mean": statistics.fmean(values_sorted),
        "p50": _percentile(values_sorted, 0.50),
        "p95": _percentile(values_sorted, 0.95),
        "p99": _percentile(values_sorted, 0.99),
        "p99_9": _percentile(values_sorted, 0.999),
        "max": max(values_sorted),
    }


def _percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    idx = min(round((len(values) - 1) * pct), len(values) - 1)
    return values[idx]


def _by_category(
    spec: BenchmarkSpec, results: list[ScenarioResult]
) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for result in results:
        grouped[result.category]["total"] += 1
        grouped[result.category][
            "succeeded" if _result_passed(spec, result) else "failed"
        ] += 1
    return {key: dict(value) for key, value in grouped.items()}


def _by_stage(
    spec: BenchmarkSpec, results: list[ScenarioResult]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ScenarioResult]] = defaultdict(list)
    for result in results:
        grouped[result.stage_id or "unknown"].append(result)
    return {
        stage_id: _result_group_summary(spec, stage_results)
        for stage_id, stage_results in sorted(grouped.items())
    }


def _by_endpoint(
    spec: BenchmarkSpec, results: list[ScenarioResult]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ScenarioResult]] = defaultdict(list)
    for result in results:
        grouped[result.endpoint].append(result)
    return {
        endpoint: _result_group_summary(spec, endpoint_results)
        for endpoint, endpoint_results in sorted(grouped.items())
    }


def _by_operation(
    spec: BenchmarkSpec, results: list[ScenarioResult]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ScenarioResult]] = defaultdict(list)
    for result in results:
        grouped[result.capability_key or result.endpoint].append(result)
    return {
        operation: _result_group_summary(spec, operation_results)
        for operation, operation_results in sorted(grouped.items())
    }


def _by_configured_concurrency(
    spec: BenchmarkSpec, results: list[ScenarioResult]
) -> dict[str, dict[str, Any]]:
    grouped: dict[int, list[ScenarioResult]] = defaultdict(list)
    for result in results:
        if result.load_concurrency is not None:
            grouped[result.load_concurrency].append(result)

    summaries: dict[str, dict[str, Any]] = {}
    for level, level_results in sorted(grouped.items()):
        summaries[str(level)] = _result_group_summary(spec, level_results)
    return summaries


def _by_peak_inflight(
    spec: BenchmarkSpec, results: list[ScenarioResult]
) -> dict[str, dict[str, Any]]:
    grouped: dict[int, list[ScenarioResult]] = defaultdict(list)
    for result in results:
        if result.peak_inflight is not None:
            grouped[result.peak_inflight].append(result)

    summaries: dict[str, dict[str, Any]] = {}
    for level, level_results in sorted(grouped.items()):
        summaries[str(level)] = _result_group_summary(spec, level_results)
    return summaries


def _voice_upload_coverage(scenarios: list[Scenario]) -> dict[str, Any]:
    successful_formats = {
        str(scenario.planned_metadata.get("upload_format"))
        for scenario in scenarios
        if scenario.capability_key == "voices.upload"
        and scenario.expect_success
        and scenario.planned_metadata.get("upload_case") == "format"
    }
    near_limit_formats = {
        str(scenario.planned_metadata.get("upload_format"))
        for scenario in scenarios
        if scenario.capability_key == "voices.upload"
        and scenario.planned_metadata.get("upload_case") == "near_limit"
    }
    cache_pressure_voice_counts = [
        scenario.planned_metadata.get("voice_count")
        for scenario in scenarios
        if scenario.capability_key == "voices.cache_pressure_traffic"
    ]
    speaker_cap_cases = sum(
        1
        for scenario in scenarios
        if scenario.capability_key in {"voices.upload", "voices.speaker_cap"}
        and scenario.planned_metadata.get("upload_case")
        in {"speaker_cap", "speaker_cap_sequence"}
    )
    speaker_cap_attempts = sum(
        _speaker_cap_attempt_count(scenario)
        for scenario in scenarios
        if scenario.capability_key in {"voices.upload", "voices.speaker_cap"}
    )
    configured_formats = [
        upload_format for upload_format, _ in VOICE_UPLOAD_SUCCESS_FORMATS
    ]
    configured_near_limit_formats = [
        upload_format for upload_format, _ in VOICE_NEAR_LIMIT_FORMATS
    ]
    generated_near_limit_formats = [
        upload_format for upload_format, _ in VOICE_NEAR_LIMIT_GENERATED_FORMATS
    ]
    near_limit_missing_generated_formats = sorted(
        set(generated_near_limit_formats) - set(near_limit_formats)
    )
    return {
        "accepted_format_cases": sorted(successful_formats),
        "configured_accepted_formats": configured_formats,
        "regular_upload_format_contract_present": bool(successful_formats),
        "near_limit_formats": sorted(near_limit_formats),
        "configured_near_limit_formats": configured_near_limit_formats,
        "generated_near_limit_formats": generated_near_limit_formats,
        "near_limit_missing_generated_formats": near_limit_missing_generated_formats,
        "near_limit_contract_complete": not near_limit_missing_generated_formats,
        "metadata_sequence_present": any(
            scenario.capability_key == "voices.upload_metadata"
            for scenario in scenarios
        ),
        "cache_pressure_voice_counts": cache_pressure_voice_counts,
        "speaker_cap_cases": speaker_cap_cases,
        "speaker_cap_attempts": speaker_cap_attempts,
    }


def _speaker_cap_attempt_count(scenario: Scenario) -> int:
    if scenario.planned_metadata.get("upload_case") == "speaker_cap":
        return 1
    if scenario.planned_metadata.get("upload_case") == "speaker_cap_sequence":
        attempt_count = scenario.planned_metadata.get("attempt_count")
        return attempt_count if isinstance(attempt_count, int) else 0
    return 0


def _coverage_failures(
    spec: BenchmarkSpec,
    scenarios: list[Scenario],
    voice_upload_coverage: dict[str, Any],
    coverage_matrix: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    enabled_endpoint_set = _enabled_endpoint_set(spec)
    generated_endpoint_set = {scenario.endpoint for scenario in scenarios}
    for endpoint in sorted(enabled_endpoint_set - generated_endpoint_set):
        failures.append(_coverage_gap(f"{endpoint}.enabled", [endpoint]))
    if "speech" in enabled_endpoint_set:
        failures.extend(_speech_coverage_failures(scenarios))
    if "speech_sse" in enabled_endpoint_set and not _has_capability(
        scenarios, "speech.sse"
    ):
        failures.append(_coverage_gap("speech.sse", ["speech.sse"]))
    if "batch" in enabled_endpoint_set:
        failures.extend(_batch_coverage_failures(scenarios))
    if "voices" in enabled_endpoint_set:
        failures.extend(
            _voice_coverage_failures(spec, scenarios, voice_upload_coverage)
        )
    if "websocket" in enabled_endpoint_set:
        failures.extend(_websocket_coverage_failures(scenarios))
    failures.extend(_coverage_matrix_failures(coverage_matrix, failures))
    return failures


def _coverage_matrix(
    spec: BenchmarkSpec,
    scenarios: list[Scenario],
    voice_upload_coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    enabled_endpoint_set = _enabled_endpoint_set(spec)
    generated_endpoint_set = {scenario.endpoint for scenario in scenarios}
    rows: list[dict[str, Any]] = []

    for endpoint, contract in (
        ("speech", "api.speech"),
        ("speech_sse", "api.speech_sse"),
        ("batch", "api.batch"),
        ("voices", "api.voices"),
        ("websocket", "api.websocket"),
    ):
        endpoint_enabled = endpoint in enabled_endpoint_set
        rows.append(
            _coverage_matrix_row(
                spec,
                contract,
                tested=endpoint_enabled and endpoint in generated_endpoint_set,
                expected=[endpoint],
                observed=sorted(generated_endpoint_set & {endpoint}),
                out_of_scope_reason=(
                    None
                    if endpoint_enabled
                    else "endpoint is not enabled by this benchmark spec"
                ),
            )
        )

    if "speech" in enabled_endpoint_set:
        rows.extend(_speech_coverage_matrix(spec, scenarios))
    if "batch" in enabled_endpoint_set:
        rows.extend(_batch_coverage_matrix(spec, scenarios))
    if "voices" in enabled_endpoint_set:
        rows.extend(_voice_coverage_matrix(spec, scenarios, voice_upload_coverage))
    if "websocket" in enabled_endpoint_set:
        rows.extend(_websocket_coverage_matrix(spec, scenarios))
    return rows


def _coverage_matrix_failures(
    coverage_matrix: list[dict[str, Any]],
    existing_failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_contracts = {
        failure.get("contract")
        for failure in existing_failures
        if isinstance(failure.get("contract"), str)
    }
    return [
        _coverage_gap(
            str(row["contract"]),
            list(row.get("missing", [])),
            status=str(row["status"]),
            error=str(row["error"]),
        )
        for row in coverage_matrix
        if row.get("status") not in {"tested", "out_of_scope"}
        and row.get("contract") not in existing_contracts
    ]


def _speech_coverage_matrix(
    spec: BenchmarkSpec,
    scenarios: list[Scenario],
) -> list[dict[str, Any]]:
    speech_scenarios = [
        scenario for scenario in scenarios if scenario.endpoint == "speech"
    ]
    rows = [
        _coverage_matrix_row(
            spec,
            "speech.languages",
            tested=not _value_coverage_gap(
                "speech.languages",
                {language for language, _ in MULTILINGUAL_TEXTS},
                _metadata_values(speech_scenarios, "language"),
            ),
            expected=sorted(language for language, _ in MULTILINGUAL_TEXTS),
            observed=sorted(_metadata_values(speech_scenarios, "language")),
        ),
        _coverage_matrix_row(
            spec,
            "speech.response_formats",
            tested=not _value_coverage_gap(
                "speech.response_formats",
                set(RESPONSE_FORMATS),
                _metadata_values(speech_scenarios, "response_format"),
            ),
            expected=list(RESPONSE_FORMATS),
            observed=sorted(_metadata_values(speech_scenarios, "response_format")),
        ),
        _coverage_matrix_row(
            spec,
            "speech.task_types",
            tested=not _value_coverage_gap(
                "speech.task_types",
                set(TASK_TYPES),
                _metadata_values(speech_scenarios, "task_type"),
            ),
            expected=list(TASK_TYPES),
            observed=sorted(_metadata_values(speech_scenarios, "task_type")),
        ),
        _coverage_matrix_row(
            spec,
            "speech.speed_boundaries",
            tested=not _value_coverage_gap(
                "speech.speed_boundaries",
                set(SPEED_BOUNDARY_VALUES),
                _metadata_values(speech_scenarios, "speed_boundary"),
            ),
            expected=list(SPEED_BOUNDARY_VALUES),
            observed=sorted(_metadata_values(speech_scenarios, "speed_boundary")),
        ),
        _coverage_matrix_row(
            spec,
            "speech.reference_cases",
            tested=not _value_coverage_gap(
                "speech.reference_cases",
                {
                    "valid_reference",
                    "valid_base64_ref_audio",
                    "valid_file_ref_audio",
                    "x_vector_only_mode",
                    *(case for case, _ in REFERENCE_FAILURES),
                },
                _metadata_values(speech_scenarios, "reference_case"),
            ),
            expected=[
                "valid_reference",
                "valid_base64_ref_audio",
                "valid_file_ref_audio",
                "x_vector_only_mode",
                *(case for case, _ in REFERENCE_FAILURES),
            ],
            observed=sorted(_metadata_values(speech_scenarios, "reference_case")),
        ),
        _coverage_matrix_row(
            spec,
            "speech.malformed_cases",
            tested=not _value_coverage_gap(
                "speech.malformed_cases",
                set(MALFORMED_CASE_NAMES),
                _metadata_values(speech_scenarios, "malformed_case"),
            ),
            expected=list(MALFORMED_CASE_NAMES),
            observed=sorted(_metadata_values(speech_scenarios, "malformed_case")),
        ),
        _coverage_matrix_row(
            spec,
            "speech.length_extremes",
            tested=not _value_coverage_gap(
                "speech.length_extremes",
                {len(text) for text in LENGTH_EXTREME_TEXTS},
                _metadata_values(speech_scenarios, "input_chars"),
            ),
            expected=sorted({len(text) for text in LENGTH_EXTREME_TEXTS}),
            observed=sorted(_metadata_values(speech_scenarios, "input_chars")),
        ),
        _coverage_matrix_row(
            spec,
            "speech.initial_codec_chunk_frames",
            tested=any(
                scenario.planned_metadata.get("initial_codec_chunk_frames")
                for scenario in speech_scenarios
            ),
            expected=["present"],
            observed=[
                scenario.planned_metadata.get("initial_codec_chunk_frames")
                for scenario in speech_scenarios
                if scenario.planned_metadata.get("initial_codec_chunk_frames")
            ],
        ),
        _coverage_matrix_row(
            spec,
            "speech.openai_sdk_response_formats",
            tested=not _value_coverage_gap(
                "speech.openai_sdk_response_formats",
                set(SDK_RESPONSE_FORMATS),
                _metadata_values(speech_scenarios, "sdk_response_format"),
            ),
            expected=list(SDK_RESPONSE_FORMATS),
            observed=sorted(_metadata_values(speech_scenarios, "sdk_response_format")),
        ),
        _coverage_matrix_row(
            spec,
            "speech.openai_sdk_error_contract",
            tested=_has_capability(speech_scenarios, "speech.openai_sdk_error"),
            expected=["speech.openai_sdk_error"],
            observed=sorted(
                {
                    scenario.capability_key
                    for scenario in speech_scenarios
                    if scenario.capability_key == "speech.openai_sdk_error"
                }
            ),
        ),
        _coverage_matrix_row(
            spec,
            "speech.openai_sdk",
            tested=(
                _has_capability(speech_scenarios, "speech.openai_sdk")
                and _has_capability(speech_scenarios, "speech.openai_sdk_error")
            ),
            expected=["speech.openai_sdk"],
            observed=sorted(
                {
                    scenario.capability_key
                    for scenario in speech_scenarios
                    if scenario.capability_key
                    in {"speech.openai_sdk", "speech.openai_sdk_error"}
                }
            ),
        ),
    ]
    return rows


def _batch_coverage_matrix(
    spec: BenchmarkSpec,
    scenarios: list[Scenario],
) -> list[dict[str, Any]]:
    batch_scenarios = [
        scenario for scenario in scenarios if scenario.endpoint == "batch"
    ]
    return [
        _coverage_matrix_row(
            spec,
            "batch.sizes",
            tested=not _value_coverage_gap(
                "batch.sizes",
                {*BATCH_SIZES, BATCH_OVERSIZED_SIZE},
                _metadata_values(batch_scenarios, "batch_size"),
            ),
            expected=sorted({*BATCH_SIZES, BATCH_OVERSIZED_SIZE}),
            observed=sorted(_metadata_values(batch_scenarios, "batch_size")),
        ),
        _coverage_matrix_row(
            spec,
            "batch.cases",
            tested=not _value_coverage_gap(
                "batch.cases",
                {"all_valid", "item_error", "item_overrides"},
                _metadata_values(batch_scenarios, "batch_case"),
            ),
            expected=["all_valid", "item_error", "item_overrides"],
            observed=sorted(_metadata_values(batch_scenarios, "batch_case")),
        ),
    ]


def _voice_coverage_matrix(
    spec: BenchmarkSpec,
    scenarios: list[Scenario],
    voice_upload_coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    voice_scenarios = [
        scenario for scenario in scenarios if scenario.endpoint == "voices"
    ]
    configured_cache_pressure_voice_count = (
        _configured_voice_cache_pressure_voice_count(spec)
    )
    configured_speaker_cap_count = _configured_voice_speaker_cap_count(spec)
    cache_observability_tested = (
        configured_cache_pressure_voice_count > 0
        and _has_capability(
            voice_scenarios,
            "voices.cache_pressure_traffic",
        )
    )
    rows = [
        _coverage_matrix_row(
            spec,
            "voices.list",
            tested=_has_capability(voice_scenarios, "voices.list"),
            expected=["voices.list"],
            observed=_capabilities_for(voice_scenarios, {"voices.list"}),
        ),
        _coverage_matrix_row(
            spec,
            "voices.accepted_upload_formats",
            tested=not _value_coverage_gap(
                "voices.accepted_upload_formats",
                {upload_format for upload_format, _ in VOICE_UPLOAD_SUCCESS_FORMATS},
                {
                    scenario.planned_metadata.get("upload_format")
                    for scenario in voice_scenarios
                    if scenario.capability_key == "voices.upload"
                    and scenario.expect_success
                    and scenario.planned_metadata.get("upload_case") == "format"
                },
            ),
            expected=[
                upload_format for upload_format, _ in VOICE_UPLOAD_SUCCESS_FORMATS
            ],
            observed=voice_upload_coverage["accepted_format_cases"],
        ),
        _coverage_matrix_row(
            spec,
            "voices.synthetic_reject_formats",
            tested=not _value_coverage_gap(
                "voices.synthetic_reject_formats",
                {upload_format for upload_format, _ in VOICE_UPLOAD_REJECT_FORMATS},
                {
                    scenario.planned_metadata.get("upload_format")
                    for scenario in voice_scenarios
                    if scenario.capability_key == "voices.upload"
                    and not scenario.expect_success
                    and scenario.planned_metadata.get("upload_case")
                    == "synthetic_header_reject"
                },
            ),
            expected=[
                upload_format for upload_format, _ in VOICE_UPLOAD_REJECT_FORMATS
            ],
            observed=sorted(
                {
                    scenario.planned_metadata.get("upload_format")
                    for scenario in voice_scenarios
                    if scenario.capability_key == "voices.upload"
                    and not scenario.expect_success
                    and scenario.planned_metadata.get("upload_case")
                    == "synthetic_header_reject"
                }
            ),
        ),
        _coverage_matrix_row(
            spec,
            "voices.near_limit_upload_formats",
            tested=voice_upload_coverage["near_limit_contract_complete"],
            expected=voice_upload_coverage["generated_near_limit_formats"],
            observed=voice_upload_coverage["near_limit_formats"],
        ),
        _coverage_matrix_row(
            spec,
            "voices.lifecycle_cases",
            tested=not _value_coverage_gap(
                "voices.lifecycle_cases",
                {
                    "oversized",
                    "corrupt_audio",
                    "overwrite",
                    "delete",
                    "lifecycle",
                    "upload_delete_race",
                    "upload_metadata",
                },
                {
                    _voice_case(scenario)
                    for scenario in voice_scenarios
                    if _voice_case(scenario) is not None
                },
            ),
            expected=[
                "oversized",
                "corrupt_audio",
                "overwrite",
                "delete",
                "lifecycle",
                "upload_delete_race",
                "upload_metadata",
            ],
            observed=sorted(
                {
                    _voice_case(scenario)
                    for scenario in voice_scenarios
                    if _voice_case(scenario) is not None
                }
            ),
        ),
        _coverage_matrix_row(
            spec,
            "voices.upload_metadata",
            tested=voice_upload_coverage["metadata_sequence_present"],
            expected=["voices.upload_metadata"],
            observed=_capabilities_for(voice_scenarios, {"voices.upload_metadata"}),
        ),
        _coverage_matrix_row(
            spec,
            "voices.speaker_cap",
            tested=(
                configured_speaker_cap_count > 0
                and voice_upload_coverage["speaker_cap_attempts"] > 0
            ),
            expected=["speaker_cap_sequence"],
            observed=(
                ["speaker_cap_sequence"]
                if voice_upload_coverage["speaker_cap_attempts"] > 0
                else []
            ),
            out_of_scope_reason=(
                None
                if configured_speaker_cap_count
                else "speaker-cap sequence is not configured in this benchmark spec"
            ),
        ),
        _coverage_matrix_row(
            spec,
            "voices.cache_pressure_traffic",
            tested=(
                configured_cache_pressure_voice_count > 0
                and _has_capability(
                    voice_scenarios,
                    "voices.cache_pressure_traffic",
                )
            ),
            expected=["voices.cache_pressure_traffic"],
            observed=_capabilities_for(
                voice_scenarios,
                {"voices.cache_pressure_traffic"},
            ),
            out_of_scope_reason=(
                None
                if configured_cache_pressure_voice_count
                else "voice cache-pressure traffic is not configured in this benchmark spec"
            ),
        ),
        _coverage_matrix_row(
            spec,
            "voices.cache_observability",
            tested=cache_observability_tested,
            expected=list(voice_contracts.VOICE_CACHE_OBSERVABILITY_COUNTERS),
            observed=(
                list(voice_contracts.VOICE_CACHE_OBSERVABILITY_COUNTERS)
                if cache_observability_tested
                else []
            ),
            out_of_scope_reason=(
                None
                if configured_cache_pressure_voice_count
                else "voice cache-pressure traffic is not configured in this benchmark spec"
            ),
        ),
    ]
    return rows


def _websocket_coverage_matrix(
    spec: BenchmarkSpec,
    scenarios: list[Scenario],
) -> list[dict[str, Any]]:
    websocket_scenarios = [
        scenario for scenario in scenarios if scenario.endpoint == "websocket"
    ]
    split_values = _websocket_split_granularity_values(websocket_scenarios)
    non_sentence_split_values = set(WEBSOCKET_SPLIT_GRANULARITIES) - {"sentence"}
    return [
        _coverage_matrix_row(
            spec,
            "websocket.cases",
            tested=not _websocket_coverage_failures(scenarios),
            expected=[
                "ws.normal",
                "ws.multi_sentence",
                "ws.stream_audio",
                "ws.clause_split",
                "ws.done_before_config",
                "ws.malformed_json",
                "ws.disconnect",
            ],
            observed=_capabilities_for(
                websocket_scenarios,
                {
                    "ws.normal",
                    "ws.multi_sentence",
                    "ws.stream_audio",
                    "ws.clause_split",
                    "ws.done_before_config",
                    "ws.malformed_json",
                    "ws.disconnect",
                },
            ),
        ),
        _coverage_matrix_row(
            spec,
            "websocket.split_granularity.sentence",
            tested="sentence" in split_values,
            expected=["sentence"],
            observed=sorted(split_values),
        ),
        _coverage_matrix_row(
            spec,
            "websocket.split_granularity.non_sentence",
            tested=non_sentence_split_values <= split_values,
            expected=sorted(non_sentence_split_values),
            observed=sorted(split_values - {"sentence"}),
        ),
    ]


def _coverage_matrix_row(
    spec: BenchmarkSpec,
    contract: str,
    *,
    tested: bool,
    expected: list[Any],
    observed: list[Any],
    gap_status: str = "coverage_gap",
    gap_error: str | None = None,
    out_of_scope_reason: str | None = None,
) -> dict[str, Any]:
    if tested:
        status = "tested"
        reason = None
        missing: list[Any] = []
        error = None
    elif out_of_scope_reason:
        status = "out_of_scope"
        reason = out_of_scope_reason
        missing = []
        error = None
    else:
        status = gap_status
        reason = None
        missing = sorted(set(expected) - set(observed), key=str)
        error = (
            gap_error
            or f"enabled benchmark contract is missing required coverage: {missing}"
        )
    return {
        "contract": contract,
        "status": status,
        "expected": expected,
        "observed": observed,
        "missing": missing,
        "reason": reason,
        "error": error,
    }


def _enabled_endpoint_set(spec: BenchmarkSpec) -> set[str]:
    endpoint_set: set[str] = set()
    for stage in spec.params.load_stages:
        endpoint_set.update(stage.enabled_endpoints or spec.params.enabled_endpoints)
    return endpoint_set


def _speech_coverage_failures(scenarios: list[Scenario]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    speech_scenarios = [
        scenario for scenario in scenarios if scenario.endpoint == "speech"
    ]
    languages = {language for language, _ in MULTILINGUAL_TEXTS}
    failures.extend(
        _value_coverage_gap(
            "speech.languages",
            languages,
            _metadata_values(speech_scenarios, "language"),
        )
    )
    failures.extend(
        _value_coverage_gap(
            "speech.response_formats",
            set(RESPONSE_FORMATS),
            _metadata_values(speech_scenarios, "response_format"),
        )
    )
    failures.extend(
        _value_coverage_gap(
            "speech.task_types",
            set(TASK_TYPES),
            _metadata_values(speech_scenarios, "task_type"),
        )
    )
    failures.extend(
        _value_coverage_gap(
            "speech.speed_boundaries",
            set(SPEED_BOUNDARY_VALUES),
            _metadata_values(speech_scenarios, "speed_boundary"),
        )
    )
    failures.extend(
        _value_coverage_gap(
            "speech.length_extremes",
            {len(text) for text in LENGTH_EXTREME_TEXTS},
            _metadata_values(speech_scenarios, "input_chars"),
        )
    )
    failures.extend(
        _value_coverage_gap(
            "speech.reference_cases",
            {
                "valid_reference",
                "valid_base64_ref_audio",
                "valid_file_ref_audio",
                "x_vector_only_mode",
                *(case for case, _ in REFERENCE_FAILURES),
            },
            _metadata_values(speech_scenarios, "reference_case"),
        )
    )
    failures.extend(
        _value_coverage_gap(
            "speech.malformed_cases",
            set(MALFORMED_CASE_NAMES),
            _metadata_values(speech_scenarios, "malformed_case"),
        )
    )
    if not any(
        scenario.planned_metadata.get("initial_codec_chunk_frames")
        for scenario in speech_scenarios
    ):
        failures.append(_coverage_gap("speech.initial_codec_chunk_frames", ["present"]))
    failures.extend(
        _value_coverage_gap(
            "speech.openai_sdk_response_formats",
            set(SDK_RESPONSE_FORMATS),
            _metadata_values(speech_scenarios, "sdk_response_format"),
        )
    )
    if not _has_capability(speech_scenarios, "speech.openai_sdk_error"):
        failures.append(
            _coverage_gap(
                "speech.openai_sdk_error_contract",
                ["speech.openai_sdk_error"],
            )
        )
    return failures


def _batch_coverage_failures(scenarios: list[Scenario]) -> list[dict[str, Any]]:
    batch_scenarios = [
        scenario for scenario in scenarios if scenario.endpoint == "batch"
    ]
    failures: list[dict[str, Any]] = []
    failures.extend(
        _value_coverage_gap(
            "batch.sizes",
            {*BATCH_SIZES, BATCH_OVERSIZED_SIZE},
            _metadata_values(batch_scenarios, "batch_size"),
        )
    )
    failures.extend(
        _value_coverage_gap(
            "batch.cases",
            {"all_valid", "item_error", "item_overrides"},
            _metadata_values(batch_scenarios, "batch_case"),
        )
    )
    return failures


def _voice_coverage_failures(
    spec: BenchmarkSpec,
    scenarios: list[Scenario],
    voice_upload_coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    voice_scenarios = [
        scenario for scenario in scenarios if scenario.endpoint == "voices"
    ]
    cap_stage_ids = {
        scenario.stage_id
        for scenario in voice_scenarios
        if scenario.capability_key == "voices.speaker_cap"
    }
    regular_voice_scenarios = [
        scenario
        for scenario in voice_scenarios
        if scenario.stage_id not in cap_stage_ids
    ]
    failures: list[dict[str, Any]] = []
    if regular_voice_scenarios or _regular_voice_coverage_required(spec):
        failures.extend(_regular_voice_coverage_failures(regular_voice_scenarios))
        if (
            voice_upload_coverage["regular_upload_format_contract_present"]
            and not voice_upload_coverage["near_limit_contract_complete"]
        ):
            failures.append(
                _coverage_gap(
                    "voices.near_limit_upload_formats",
                    voice_upload_coverage["near_limit_missing_generated_formats"],
                )
            )
        if not voice_upload_coverage["metadata_sequence_present"]:
            failures.append(
                _coverage_gap(
                    "voices.upload_metadata",
                    ["voices.upload_metadata"],
                )
            )
    if cap_stage_ids and voice_upload_coverage["speaker_cap_attempts"] <= 0:
        failures.append(_coverage_gap("voices.speaker_cap", ["speaker_cap_sequence"]))
    if _configured_voice_cache_pressure_voice_count(spec) and not _has_capability(
        voice_scenarios, "voices.cache_pressure_traffic"
    ):
        failures.append(
            _coverage_gap(
                "voices.cache_pressure_traffic",
                ["voices.cache_pressure_traffic"],
            )
        )
    return failures


def _regular_voice_coverage_required(spec: BenchmarkSpec) -> bool:
    for stage in spec.params.load_stages:
        endpoints = stage.enabled_endpoints or spec.params.enabled_endpoints
        if "voices" not in endpoints:
            continue
        if _effective_voice_speaker_cap_count(spec, stage):
            continue
        return True
    return False


def _effective_voice_speaker_cap_count(spec: BenchmarkSpec, stage: Any) -> int:
    if stage.voice_speaker_cap_count:
        return stage.voice_speaker_cap_count
    endpoints = stage.enabled_endpoints or spec.params.enabled_endpoints
    if tuple(endpoints) == ("voices",):
        return spec.params.voice_speaker_cap_count
    return 0


def _regular_voice_coverage_failures(scenarios: list[Scenario]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not _has_capability(scenarios, "voices.list"):
        failures.append(_coverage_gap("voices.list", ["voices.list"]))
    failures.extend(
        _value_coverage_gap(
            "voices.accepted_upload_formats",
            {upload_format for upload_format, _ in VOICE_UPLOAD_SUCCESS_FORMATS},
            {
                scenario.planned_metadata.get("upload_format")
                for scenario in scenarios
                if scenario.capability_key == "voices.upload"
                and scenario.expect_success
                and scenario.planned_metadata.get("upload_case") == "format"
            },
        )
    )
    failures.extend(
        _value_coverage_gap(
            "voices.synthetic_reject_formats",
            {upload_format for upload_format, _ in VOICE_UPLOAD_REJECT_FORMATS},
            {
                scenario.planned_metadata.get("upload_format")
                for scenario in scenarios
                if scenario.capability_key == "voices.upload"
                and not scenario.expect_success
                and scenario.planned_metadata.get("upload_case")
                == "synthetic_header_reject"
            },
        )
    )
    failures.extend(
        _value_coverage_gap(
            "voices.lifecycle_cases",
            {
                "oversized",
                "corrupt_audio",
                "overwrite",
                "delete",
                "lifecycle",
                "upload_delete_race",
                "upload_metadata",
            },
            {
                _voice_case(scenario)
                for scenario in scenarios
                if scenario.endpoint == "voices"
            },
        )
    )
    return failures


def _websocket_coverage_failures(scenarios: list[Scenario]) -> list[dict[str, Any]]:
    return _value_coverage_gap(
        "websocket.cases",
        {
            "ws.normal",
            "ws.multi_sentence",
            "ws.stream_audio",
            "ws.clause_split",
            "ws.done_before_config",
            "ws.malformed_json",
            "ws.disconnect",
        },
        {
            scenario.capability_key
            for scenario in scenarios
            if scenario.endpoint == "websocket"
        },
    )


def _metadata_values(scenarios: list[Scenario], key: str) -> set[Any]:
    return {
        scenario.planned_metadata.get(key)
        for scenario in scenarios
        if scenario.planned_metadata.get(key) is not None
    }


def _value_coverage_gap(
    contract: str, expected: set[Any], actual: set[Any]
) -> list[dict[str, Any]]:
    missing = sorted(expected - actual)
    return [_coverage_gap(contract, missing)] if missing else []


def _coverage_gap(
    contract: str,
    missing: list[Any],
    *,
    status: str = "coverage_gap",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "contract": contract,
        "status": status,
        "missing": missing,
        "error": error
        or f"enabled benchmark contract is missing required scenarios: {missing}",
    }


def _has_capability(scenarios: list[Scenario], capability_key: str) -> bool:
    return any(scenario.capability_key == capability_key for scenario in scenarios)


def _capabilities_for(
    scenarios: list[Scenario],
    capability_keys: set[str],
) -> list[str]:
    return sorted(
        {
            scenario.capability_key
            for scenario in scenarios
            if scenario.capability_key in capability_keys
        }
    )


def _websocket_split_granularity_values(scenarios: list[Scenario]) -> set[str]:
    values: set[str] = set()
    for scenario in scenarios:
        for action in scenario.script:
            payload = action.get("payload")
            if not isinstance(payload, dict):
                continue
            split_granularity = payload.get("split_granularity")
            if isinstance(split_granularity, str) and split_granularity:
                values.add(split_granularity)
    return values


def _unsupported_contract_summary(
    results: list[ScenarioResult],
    scenarios: list[Scenario],
) -> list[dict[str, Any]]:
    scenario_by_id = {scenario.id: scenario for scenario in scenarios}
    grouped: dict[tuple[str, str, str, int | None], list[ScenarioResult]] = defaultdict(
        list
    )
    for result in results:
        if result.status != "unsupported_contract":
            continue
        scenario = scenario_by_id.get(result.scenario_id)
        path = scenario.path if scenario is not None else ""
        grouped[
            (
                result.endpoint,
                result.capability_key or result.endpoint,
                path,
                result.http_status,
            )
        ].append(result)

    summaries: list[dict[str, Any]] = []
    for (endpoint, operation, path, http_status), group in sorted(grouped.items()):
        summaries.append(
            {
                "endpoint": endpoint,
                "operation": operation,
                "path": path or None,
                "http_status": http_status,
                "count": len(group),
                "samples": [
                    {
                        "scenario_id": result.scenario_id,
                        "error": _truncate(result.error, 500),
                    }
                    for result in group[:5]
                ],
            }
        )
    return summaries


def _truncate(value: str | None, limit: int) -> str | None:
    if value is None or len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _voice_case(scenario: Scenario) -> str | None:
    if scenario.capability_key == "voices.overwrite":
        return "overwrite"
    if scenario.capability_key == "voices.delete":
        return "delete"
    if scenario.capability_key == "voices.lifecycle":
        return "lifecycle"
    if scenario.capability_key == "voices.upload_delete_race":
        return "upload_delete_race"
    if scenario.capability_key == "voices.upload_metadata":
        return "upload_metadata"
    return scenario.planned_metadata.get("upload_case")


def _configured_voice_cache_pressure_voice_count(spec: BenchmarkSpec) -> int:
    stage_count = sum(
        stage.voice_cache_pressure_voice_count for stage in spec.params.load_stages
    )
    return stage_count or spec.params.voice_cache_pressure_voice_count


def _configured_voice_speaker_cap_count(spec: BenchmarkSpec) -> int:
    stage_count = sum(
        stage.voice_speaker_cap_count for stage in spec.params.load_stages
    )
    return stage_count or spec.params.voice_speaker_cap_count


def _result_group_summary(
    spec: BenchmarkSpec, results: list[ScenarioResult]
) -> dict[str, Any]:
    latencies = [result.latency_s for result in results if result.latency_s > 0]
    ttfas = [result.ttfa_s for result in results if result.ttfa_s is not None]
    rtfs = [result.rtf for result in results if result.rtf > 0]
    queue_waits = [
        result.queue_wait_s for result in results if result.queue_wait_s is not None
    ]
    generator_lags = [
        result.generator_lag_s
        for result in results
        if result.generator_lag_s is not None
    ]
    planned_starts = [
        result.planned_start_s
        for result in results
        if result.planned_start_s is not None
    ]
    first_start = min(
        (result.actual_start_s for result in results if result.actual_start_s),
        default=None,
    )
    last_completion = max(
        (result.completed_s for result in results if result.completed_s),
        default=None,
    )
    wall_time_s = (
        last_completion - first_start
        if first_start is not None and last_completion is not None
        else None
    )
    planned_window_s = (
        max(planned_starts) - min(planned_starts) if len(planned_starts) > 1 else None
    )
    succeeded = sum(1 for result in results if _result_passed(spec, result))
    failed = len(results) - succeeded
    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "wall_time_s": wall_time_s,
        "planned_window_s": planned_window_s,
        "configured_max_concurrency": sorted(
            {
                result.configured_max_concurrency
                for result in results
                if result.configured_max_concurrency is not None
            }
        ),
        "peak_inflight": max(
            (result.peak_inflight for result in results if result.peak_inflight),
            default=None,
        ),
        "peak_pending_tasks": max(
            (
                result.peak_pending_tasks
                for result in results
                if result.peak_pending_tasks
            ),
            default=None,
        ),
        "scheduled_task_count": max(
            (
                result.scheduled_task_count
                for result in results
                if result.scheduled_task_count
            ),
            default=None,
        ),
        "load_generator_lagged": any(
            result.load_generator_lagged for result in results
        ),
        "load_generator_saturated": any(
            result.load_generator_saturated for result in results
        ),
        "offered_rps": (
            len(results) / planned_window_s
            if planned_window_s and planned_window_s > 0
            else None
        ),
        "achieved_rps": (
            len(results) / wall_time_s if wall_time_s and wall_time_s > 0 else None
        ),
        "latency_s": _summary(latencies),
        "ttfa_s": _summary(ttfas),
        "queue_wait_s": _summary(queue_waits),
        "generator_lag_s": _summary(generator_lags),
        "rtf": _summary(rtfs),
        "status_counts": dict(Counter(result.status for result in results)),
        "http_status_counts": dict(
            Counter(
                str(result.http_status)
                for result in results
                if result.http_status is not None
            )
        ),
        "admission_status_counts": _admission_status_counts(results),
        "error_class_counts": dict(
            Counter(result.error_class for result in results if result.error_class)
        ),
        "category_counts": dict(Counter(result.category for result in results)),
    }


def _roll_up_endpoint_capability(statuses: list[str]) -> str:
    return _roll_up_capability(statuses)


def _admission_status_counts(results: list[ScenarioResult]) -> dict[str, int]:
    counts = Counter()
    for result in results:
        if result.http_status in {429, 503}:
            counts[str(result.http_status)] += 1
        if result.status == "transport_error":
            counts["transport_error"] += 1
        if result.error_type and "Timeout" in result.error_type:
            counts["timeout"] += 1
    return dict(counts)


def _load_generation_error(results: list[ScenarioResult]) -> str | None:
    if any(result.load_generator_saturated for result in results):
        return "scheduled arrivals saturated the benchmark client concurrency cap"
    if any(result.load_generator_lagged for result in results):
        return "scheduled arrivals lagged beyond benchmark threshold"
    return None


def _rtf_unsupported_format_counts(results: list[ScenarioResult]) -> dict[str, int]:
    counts = Counter()
    for result in results:
        response_format = (result.response_format or "").lower()
        if (
            response_format
            and response_format not in {"wav", "pcm"}
            and result.audio_bytes > 0
            and result.rtf == 0
        ):
            counts[response_format] += 1
    return dict(counts)


def _roll_up_capability(statuses: list[str]) -> str:
    if not statuses:
        return "missing"
    if "fail" in statuses:
        return "fail"
    unique = set(statuses)
    if len(unique) == 1:
        return statuses[0]
    return "partial"
