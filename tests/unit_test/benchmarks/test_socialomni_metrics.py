# SPDX-License-Identifier: Apache-2.0
"""SocialOmni scoring denominators and judge completeness."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

import benchmarks.eval.benchmark_omni_socialomni as socialomni_eval
import benchmarks.tasks.socialomni as socialomni_tasks
from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.socialomni import SocialOmniLevel2Sample
from benchmarks.metrics.socialomni import (
    SOCIALOMNI_JUDGE_NAMES,
    JudgeCompletenessError,
    compute_socialomni_level2_metrics,
)
from benchmarks.tasks.socialomni_protocol import JudgeSpec


@pytest.fixture
def scored_response() -> dict[str, str | bool | dict[str, int]]:
    return {
        "sample_id": "scored",
        "gold_when": "YES",
        "predicted_when": "YES",
        "when_success": True,
        "gold_response": "Hello",
        "gold_response_success": True,
        "gold_judge_scores": {judge: 75 for judge in SOCIALOMNI_JUDGE_NAMES},
    }


@pytest.mark.parametrize(
    ("response_success", "response_text"),
    [(False, "Failed continuation"), (True, ""), (True, " \n")],
    ids=["failed", "empty", "whitespace"],
)
def test_failed_or_empty_responses_remain_in_quality_denominator(
    scored_response: dict[str, str | bool | dict[str, int]],
    response_success: bool,
    response_text: str,
) -> None:
    records = [
        scored_response,
        {
            **scored_response,
            "sample_id": "unscored",
            "gold_response": response_text,
            "gold_response_success": response_success,
            "gold_judge_scores": {},
        },
    ]

    metrics = compute_socialomni_level2_metrics(records, bootstrap_samples=100)

    assert socialomni_eval.has_complete_judges(records, configured=True)
    assert metrics["quality"]["gold_positive_samples"] == 2
    assert metrics["quality"]["covered_positive_samples"] == 1
    assert metrics["quality"]["qgold"] == 37.5
    assert metrics["quality"]["qens"] == 75.0
    assert metrics["quality"]["cov_plus"] == 0.5
    assert metrics["quality"]["qens_joint"] == 37.5


@pytest.mark.parametrize(
    "judge_scores",
    [
        {"gpt-4o": 75, "gemini-2.5-pro": 75},
        {"gpt-4o": 75, "gemini-2.5-pro": 75, "qwen3-omni": 37},
    ],
    ids=["missing-judge", "invalid-score"],
)
def test_missing_or_invalid_judge_score_prevents_quality_metrics(
    scored_response: dict[str, str | bool | dict[str, int]],
    judge_scores: dict[str, int],
) -> None:
    records = [{**scored_response, "gold_judge_scores": judge_scores}]

    assert not socialomni_eval.has_complete_judges(records, configured=True)
    with pytest.raises(JudgeCompletenessError, match="sample 'scored'"):
        compute_socialomni_level2_metrics(records, bootstrap_samples=100)


def test_incorrect_turn_decision_only_contributes_to_gold_quality(
    scored_response: dict[str, str | bool | dict[str, int]],
) -> None:
    records = [
        scored_response,
        {
            **scored_response,
            "sample_id": "missed-turn",
            "predicted_when": "NO",
            "gold_judge_scores": {judge: 25 for judge in SOCIALOMNI_JUDGE_NAMES},
        },
    ]

    metrics = compute_socialomni_level2_metrics(records, bootstrap_samples=100)

    assert metrics["quality"]["qgold"] == 50.0
    assert metrics["quality"]["qens"] == 75.0
    assert metrics["quality"]["cov_plus"] == 0.5
    assert metrics["quality"]["qens_joint"] == 37.5


def test_quality_bootstrap_is_reproducible(
    scored_response: dict[str, str | bool | dict[str, int]],
) -> None:
    records = [
        {
            **scored_response,
            "sample_id": str(score),
            "gold_judge_scores": {judge: score for judge in SOCIALOMNI_JUDGE_NAMES},
        }
        for score in (0, 25, 75, 100)
    ]

    first = compute_socialomni_level2_metrics(
        records, bootstrap_seed=42, bootstrap_samples=100
    )
    second = compute_socialomni_level2_metrics(
        records, bootstrap_seed=42, bootstrap_samples=100
    )

    assert first == second
    for interval_name in ("qgold_ci95", "qens_ci95", "qens_joint_ci95"):
        interval = first["quality"][interval_name]
        assert 0 <= interval["low"] < 50 < interval["high"] <= 100


@pytest.mark.asyncio
async def test_no_eligible_responses_need_no_judge_requests(
    monkeypatch: pytest.MonkeyPatch,
    scored_response: dict[str, str | bool | dict[str, int]],
) -> None:
    records = [
        {
            **scored_response,
            "gold_response_success": False,
            "gold_judge_scores": {},
        }
    ]
    judges = [
        JudgeSpec(name, name, "http://judge.invalid", None, 1)
        for name in SOCIALOMNI_JUDGE_NAMES
    ]
    request = AsyncMock()
    monkeypatch.setattr(socialomni_tasks, "request_chat_completion", request)

    requests, failures = await socialomni_tasks.run_judges(
        [], records, judges, timeout_s=1, disable_tqdm=True
    )
    metrics = compute_socialomni_level2_metrics(records, bootstrap_samples=100)

    request.assert_not_awaited()
    assert requests == failures == []
    assert socialomni_eval.has_complete_judges(records, configured=True)
    assert not socialomni_eval.has_complete_judges(records, configured=False)
    assert metrics["quality"]["gold_positive_samples"] == 1
    assert metrics["quality"]["qgold"] == 0.0
    assert metrics["quality"]["qens"] is None
    assert metrics["quality"]["cov_plus"] == 0.0
    assert metrics["quality"]["qens_joint"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gold_when", "response_success", "response_text"),
    [("NO", True, "Unused continuation"), ("YES", False, "Failed"), ("YES", True, "")],
    ids=["negative", "failed", "empty"],
)
@pytest.mark.parametrize("judge_success", [True, False], ids=["scored", "judge-failed"])
async def test_evaluation_counts_only_required_judgments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scored_response: dict[str, str | bool | dict[str, int]],
    gold_when: str,
    response_success: bool,
    response_text: str,
    judge_success: bool,
) -> None:
    records = [
        {**scored_response, "gold_judge_scores": {}, "judge_results": {}},
        {
            **scored_response,
            "sample_id": "excluded",
            "gold_when": gold_when,
            "gold_response": response_text,
            "gold_response_success": response_success,
            "gold_judge_scores": {},
            "judge_results": {},
        },
    ]
    samples = [
        SocialOmniLevel2Sample(
            sample_id=record["sample_id"],
            video_path=str(tmp_path / "video.mp4"),
            timestamp_s=1.0,
            target_participant="Alice",
            question_when="Should Alice speak?",
            question_how="What should Alice say?",
            gold_when=record["gold_when"],
            reference_response="Hello",
            reference_context="Alice meets a friend.",
        )
        for record in records
    ]
    judges = [
        JudgeSpec(name, name, "http://judge.invalid", None, 1)
        for name in SOCIALOMNI_JUDGE_NAMES
    ]
    monkeypatch.setattr(
        socialomni_eval,
        "inspect_socialomni_dataset",
        Mock(return_value={"metadata_matches_expected_revision": True}),
    )
    monkeypatch.setattr(
        socialomni_eval, "load_socialomni_level2_samples", Mock(return_value=samples)
    )
    monkeypatch.setattr(socialomni_eval, "load_judge_config", Mock(return_value=judges))
    monkeypatch.setattr(
        socialomni_eval, "collect_benchmark_provenance", Mock(return_value={})
    )
    monkeypatch.setattr(
        socialomni_eval,
        "run_level2_model",
        AsyncMock(return_value=(records, [], 1.0)),
    )
    request = AsyncMock(
        return_value=RequestResult(
            request_id="judge",
            text="75" if judge_success else "",
            is_success=judge_success,
            error="" if judge_success else "Judge unavailable",
        )
    )
    monkeypatch.setattr(socialomni_tasks, "request_chat_completion", request)
    config = socialomni_eval.SocialOmniEvalConfig(
        dataset_root=str(tmp_path),
        model="test-model",
        base_url="http://model.invalid",
        level="level2",
        judge_config=str(tmp_path / "judges.json"),
        prefix_cache_dir=str(tmp_path / "prefixes"),
        mini=False,
        max_samples=None,
        max_concurrency=1,
        timeout_s=1,
        output_dir=str(tmp_path),
        warmup=0,
        disable_tqdm=True,
    )

    output = await socialomni_eval.run_socialomni(config)

    assert request.await_count == 3
    assert all(
        call.kwargs["request_id"].startswith("scored:judge:")
        for call in request.await_args_list
    )
    metrics = output["summary"]["level2"]["metrics"]
    assert metrics["judge_status"] == {
        "configured": True,
        "complete": judge_success,
        "eligible_responses": 1,
        "completed_scores": 3 if judge_success else 0,
        "required_scores": 3,
    }
    assert (metrics["quality"] is not None) is judge_success
    assert output["summary"]["status"] == (
        "complete" if judge_success else "incomplete"
    )
