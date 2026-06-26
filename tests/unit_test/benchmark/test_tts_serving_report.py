# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.report import build_results_report
from benchmarks.tts_serving.scenarios import Scenario, build_scenarios
from benchmarks.tts_serving.spec import load_spec

STRESS_SPEC_PATH = Path("benchmarks/tts_serving/examples/stress.json")


def test_tts_serving_error_report_handles_missing_long_prefill_decode() -> None:
    spec = load_spec(STRESS_SPEC_PATH)
    scenarios = build_scenarios(spec)

    report = build_results_report(
        spec,
        [],
        scenarios=scenarios,
        harness_status="error",
        harness_error="probe",
    )

    assert report["harness_status"] == "error"
    assert report["overall"]["passed"] is False
    assert report["overall"]["coverage_contract_valid"] is False
    assert _coverage_failure(report, "speech.long_prefill_decode") is not None


def test_tts_serving_long_prefill_decode_requires_every_speech_stage() -> None:
    spec = load_spec(STRESS_SPEC_PATH)
    scenarios = build_scenarios(spec)
    long_prefill_ids = {
        scenario.id
        for scenario in scenarios
        if scenario.category == "speech_long_prefill_decode"
    }
    first_long_prefill_id = sorted(long_prefill_ids)[0]
    results = [
        _passing_result(scenario)
        for scenario in scenarios
        if scenario.category != "speech_long_prefill_decode"
        or scenario.id == first_long_prefill_id
    ]

    report = build_results_report(spec, results, scenarios=scenarios)

    failure = _coverage_failure(report, "speech.long_prefill_decode")
    assert report["overall"]["coverage_contract_valid"] is False
    assert failure is not None
    assert {item["stage_id"] for item in failure["missing"]} == {
        "closed-16",
        "mixed-burst-512",
        "ramp-128",
        "soak-300s",
    }


def _passing_result(scenario: Scenario) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario.id,
        endpoint=scenario.endpoint,
        category=scenario.category,
        capability_key=scenario.capability_key,
        stage_id=scenario.stage_id,
        expected_success=scenario.expect_success,
        success=scenario.expect_success,
        status="ok" if scenario.expect_success else "expected_error",
    )


def _coverage_failure(
    report: dict[str, object], contract: str
) -> dict[str, object] | None:
    failures = report["coverage_failures"]
    assert isinstance(failures, list)
    for failure in failures:
        assert isinstance(failure, dict)
        if failure.get("contract") == contract:
            return failure
    return None
