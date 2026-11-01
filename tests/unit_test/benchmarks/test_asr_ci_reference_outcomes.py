# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the artifacts emitted by the ASR reference CI tests."""

import json
from contextlib import nullcontext
from dataclasses import replace
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchmarks.dataset.seedtts import SampleInput
from tests.test_model import test_asr_ci_seedtts as asr_ci
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    RouterWorkerTrafficGuard,
)


@pytest.fixture
def asr_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ManagedRouterHandle, AsyncMock, MagicMock]:
    router = ManagedRouterHandle(
        proc=MagicMock(),
        port=8000,
        worker_ports=[8001, 8002],
        log_file=None,
        router_ready_s=1.0,
    )
    router_guard = MagicMock(spec=RouterWorkerTrafficGuard)
    router_guard.handle = router
    benchmark = AsyncMock()
    monkeypatch.setattr(asr_ci, "require_cuda", MagicMock())
    monkeypatch.setattr(asr_ci, "run_asr_seedtts_once", benchmark)
    monkeypatch.setattr(
        asr_ci,
        "router_worker_traffic_guard",
        MagicMock(return_value=nullcontext(router_guard)),
    )
    monkeypatch.setattr(asr_ci, "print_asr_wer_summary", MagicMock())
    monkeypatch.setattr(asr_ci, "print_asr_speed_summary", MagicMock())
    monkeypatch.setattr(asr_ci, "PRESET", replace(asr_ci.PRESET, gate_thresholds=True))
    benchmark.return_value = {
        "summary": {
            "total_samples": asr_ci.SEEDTTS_ASR_EN_SAMPLES,
            "evaluated": asr_ci.SEEDTTS_ASR_EN_SAMPLES,
            "corpus_wer": asr_ci.THRESHOLDS.en_corpus_wer_max,
        },
        "speed": {
            "throughput_samples_per_s": asr_ci.THRESHOLDS.throughput_min,
            "latency_mean_s": asr_ci.THRESHOLDS.latency_mean_max_s,
            "latency_p95_s": asr_ci.THRESHOLDS.latency_p95_max_s,
            "rtf_mean": asr_ci.THRESHOLDS.rtf_mean_max,
            "rtf_p95": asr_ci.THRESHOLDS.rtf_p95_max,
        },
        "per_sample": [],
    }
    return router, benchmark, router_guard


def seedtts_samples(count: int) -> list[SampleInput]:
    return [
        SampleInput(
            sample_id=str(index),
            ref_text="reference",
            ref_audio="unused.wav",
            target_text="reference",
        )
        for index in range(count)
    ]


def test_speed_failure_preserves_quality_outcome(
    asr_boundaries: tuple[ManagedRouterHandle, AsyncMock, MagicMock],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    router, benchmark, router_guard = asr_boundaries
    benchmark.return_value["speed"]["throughput_samples_per_s"] = 0.0

    with pytest.raises(asr_ci.ReferenceThresholdError, match="throughput"):
        asr_ci.test_asr_matches_seedtts_reference_text_en(
            seedtts_samples(asr_ci.SEEDTTS_ASR_EN_SAMPLES), router, tmp_path_factory
        )

    artifact = json.loads(
        (tmp_path_factory.getbasetemp() / asr_ci.EN_RESULTS_BASENAME).read_text()
    )
    assert artifact.get("reference_outcomes") == {
        "quality": {"status": "passed", "failed_checks": []},
        "speed": {"status": "failed", "failed_checks": ["throughput_samples_per_s"]},
    }
    assert artifact["execution"] == {
        "completed": True,
        "valid": True,
        "failed_checks": [],
    }
    assert artifact["reference_assertion_failed"] is True
    router_guard.assert_served.assert_called_once_with(
        min_total_requests=asr_ci.SEEDTTS_ASR_EN_SAMPLES,
        min_worker_share=0.40,
    )


def test_chinese_quality_pass_is_recorded(
    asr_boundaries: tuple[ManagedRouterHandle, AsyncMock, MagicMock],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    router, benchmark, router_guard = asr_boundaries
    benchmark.return_value["summary"] = {
        "total_samples": asr_ci.SEEDTTS_ASR_ZH_SAMPLES,
        "evaluated": asr_ci.SEEDTTS_ASR_ZH_SAMPLES,
        "corpus_wer": asr_ci.THRESHOLDS.zh_corpus_wer_max,
    }
    asr_ci.test_asr_matches_seedtts_reference_text_zh(
        seedtts_samples(asr_ci.SEEDTTS_ASR_ZH_SAMPLES), router, tmp_path_factory
    )
    artifact = json.loads(
        (tmp_path_factory.getbasetemp() / asr_ci.ZH_RESULTS_BASENAME).read_text()
    )
    assert artifact.get("reference_outcomes") == {
        "quality": {"status": "passed", "failed_checks": []},
    }
    assert artifact["execution"]["valid"] is True
    assert artifact["execution"]["completed"] is True
    assert artifact["reference_assertion_failed"] is False
    router_guard.assert_served.assert_called_once_with(
        min_total_requests=asr_ci.SEEDTTS_ASR_ZH_SAMPLES,
        min_worker_share=0.40,
    )


@pytest.mark.parametrize("split", ["en", "zh"])
def test_quality_failures_keep_their_check_labels(
    asr_boundaries: tuple[ManagedRouterHandle, AsyncMock, MagicMock],
    tmp_path_factory: pytest.TempPathFactory,
    split: Literal["en", "zh"],
) -> None:
    router, benchmark, _router_guard = asr_boundaries
    if split == "en":
        sample_count = asr_ci.SEEDTTS_ASR_EN_SAMPLES
        run_test = asr_ci.test_asr_matches_seedtts_reference_text_en
        results_basename = asr_ci.EN_RESULTS_BASENAME
    else:
        sample_count = asr_ci.SEEDTTS_ASR_ZH_SAMPLES
        run_test = asr_ci.test_asr_matches_seedtts_reference_text_zh
        results_basename = asr_ci.ZH_RESULTS_BASENAME
    benchmark.return_value["summary"] = {
        "total_samples": sample_count,
        "evaluated": sample_count,
        "corpus_wer": 1.0,
    }
    benchmark.return_value["per_sample"] = [
        {
            "id": "misrecognized",
            "ref_text": "reference",
            "hyp_text": "different",
            "ref_norm": "reference",
            "hyp_norm": "different",
            "is_success": True,
            "wer": 1.0,
        }
    ]
    with pytest.raises(asr_ci.ReferenceThresholdError, match="high-WER"):
        run_test(seedtts_samples(sample_count), router, tmp_path_factory)
    artifact = json.loads(
        (tmp_path_factory.getbasetemp() / results_basename).read_text()
    )
    assert artifact["reference_outcomes"]["quality"] == {
        "status": "failed",
        "failed_checks": ["corpus_wer", "sample_wer"],
    }
    assert artifact["execution"]["valid"] is True
    assert artifact["reference_assertion_failed"] is True


def test_report_only_does_not_claim_reference_pass(
    asr_boundaries: tuple[ManagedRouterHandle, AsyncMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    router, benchmark, _router_guard = asr_boundaries
    monkeypatch.setattr(asr_ci, "PRESET", replace(asr_ci.PRESET, gate_thresholds=False))
    benchmark.return_value["summary"]["corpus_wer"] = 1.0
    benchmark.return_value["speed"]["throughput_samples_per_s"] = 0.0
    asr_ci.test_asr_matches_seedtts_reference_text_en(
        seedtts_samples(asr_ci.SEEDTTS_ASR_EN_SAMPLES), router, tmp_path_factory
    )
    artifact = json.loads(
        (tmp_path_factory.getbasetemp() / asr_ci.EN_RESULTS_BASENAME).read_text()
    )
    assert artifact.get("reference_outcomes") == {
        "quality": {"status": "not_checked", "failed_checks": []},
        "speed": {"status": "not_checked", "failed_checks": []},
    }
    assert artifact["execution"]["valid"] is True
    assert artifact["reference_assertion_failed"] is False


@pytest.mark.parametrize(
    ("missing_count", "failed_check"),
    [
        ("input", "input_sample_count"),
        ("scored", "scored_sample_count"),
        ("evaluated", "evaluated_samples"),
    ],
)
def test_incomplete_execution_is_not_a_reference_failure(
    asr_boundaries: tuple[ManagedRouterHandle, AsyncMock, MagicMock],
    tmp_path_factory: pytest.TempPathFactory,
    missing_count: str,
    failed_check: str,
) -> None:
    router, benchmark, _router_guard = asr_boundaries
    sample_count = asr_ci.SEEDTTS_ASR_EN_SAMPLES
    if missing_count == "input":
        sample_count -= 1
    elif missing_count == "scored":
        benchmark.return_value["summary"]["total_samples"] -= 1
        benchmark.return_value["summary"]["evaluated"] -= 1
    else:
        benchmark.return_value["summary"]["evaluated"] -= 1
    benchmark.return_value["speed"]["throughput_samples_per_s"] = 0.0

    with pytest.raises(AssertionError) as raised:
        asr_ci.test_asr_matches_seedtts_reference_text_en(
            seedtts_samples(sample_count), router, tmp_path_factory
        )
    assert not isinstance(raised.value, asr_ci.ReferenceThresholdError)
    artifact = json.loads(
        (tmp_path_factory.getbasetemp() / asr_ci.EN_RESULTS_BASENAME).read_text()
    )
    assert artifact.get("execution") == {
        "completed": False,
        "valid": False,
        "failed_checks": [failed_check],
    }
    assert artifact["reference_assertion_failed"] is False


@pytest.mark.parametrize("failure", [AssertionError("router share"), OSError("router")])
def test_router_failure_keeps_execution_invalid(
    asr_boundaries: tuple[ManagedRouterHandle, AsyncMock, MagicMock],
    tmp_path_factory: pytest.TempPathFactory,
    failure: AssertionError | OSError,
) -> None:
    router, benchmark, router_guard = asr_boundaries
    benchmark.return_value["speed"]["throughput_samples_per_s"] = 0.0
    router_guard.assert_served.side_effect = failure

    with pytest.raises(type(failure)) as raised:
        asr_ci.test_asr_matches_seedtts_reference_text_en(
            seedtts_samples(asr_ci.SEEDTTS_ASR_EN_SAMPLES), router, tmp_path_factory
        )
    assert raised.value is failure
    artifact = json.loads(
        (tmp_path_factory.getbasetemp() / asr_ci.EN_RESULTS_BASENAME).read_text()
    )
    assert artifact.get("execution") == {
        "completed": True,
        "valid": False,
        "failed_checks": ["router_traffic"],
    }
    assert artifact["reference_assertion_failed"] is False
