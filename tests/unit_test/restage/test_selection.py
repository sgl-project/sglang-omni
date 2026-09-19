from dataclasses import replace

import pytest

from sglang_omni.restage.evaluation import SLO, Observation, evaluate
from sglang_omni.restage.search import search_rates
from sglang_omni.restage.selection import select_candidate


async def measured(pass_until, goodput):
    async def trial(rate, repeat):
        result = evaluate(
            [Observation("a", 0, 0.1, True, rate <= pass_until)],
            SLO(max_latency_s=1),
            expected_requests=1,
            elapsed_s=1,
        )
        return replace(result, completed_qps=1000, goodput_qps=goodput)

    return await search_rates(trial, [1, 2, 3], repeats=2)


@pytest.mark.asyncio
async def test_selects_slo_capacity_and_retains_baseline_explanation():
    results = {
        "default": await measured(1, 1),
        "split": await measured(2, 2),
        "overloaded": await measured(0, 100),
    }
    selection = select_candidate(results, baseline="default")
    assert selection.recommended == "split"
    assert selection.baseline_rate == 1
    assert selection.rate_gain_over_baseline == 2
    assert selection.ranking[0].key == "split"
    assert selection.ranking[-1].key == "overloaded"
    assert selection.ranking[-1].best_tested_rate is None
    assert "no tested rate" in selection.ranking[-1].reason


@pytest.mark.asyncio
async def test_no_feasible_candidate_does_not_export_a_winner():
    selection = select_candidate(
        {"default": await measured(0, 100)}, baseline="default"
    )
    assert selection.recommended is None
    assert selection.rate_gain_over_baseline is None


@pytest.mark.asyncio
async def test_tied_baseline_is_retained_and_unbounded_capacity_is_visible():
    result = await measured(3, 3)
    selection = select_candidate(
        {"alternative": result, "default": result}, baseline="default"
    )
    assert selection.recommended == "default"
    assert selection.ranking[0].upper_limit_passed
    assert "upper limit" in selection.ranking[0].reason


@pytest.mark.asyncio
async def test_different_load_grids_cannot_be_compared_as_one_search():
    result = await measured(2, 2)
    with pytest.raises(ValueError, match="same rates and repeat counts"):
        select_candidate(
            {"default": result, "short": replace(result, points=result.points[:1])},
            baseline="default",
        )


@pytest.mark.asyncio
async def test_isolated_high_pass_does_not_outrank_passing_lower_grid():
    async def trial(rate, repeat):
        return evaluate(
            [Observation("a", 0, 0.1, True, rate != 2)],
            SLO(max_latency_s=1),
            expected_requests=1,
            elapsed_s=1,
        )

    isolated = await search_rates(trial, [1, 2, 3], repeats=2)
    selection = select_candidate(
        {"default": await measured(2, 2), "isolated": isolated}, baseline="default"
    )
    assert selection.recommended == "default"
    row = next(row for row in selection.ranking if row.key == "isolated")
    assert row.best_tested_rate == 3
    assert row.passing_prefix_rate == 1


@pytest.mark.asyncio
async def test_prediction_agreement_counts_correctly_ordered_pairs():
    results = {
        "default": await measured(1, 1),
        "split": await measured(2, 2),
        "colocated": await measured(3, 3),
    }
    agreeing = select_candidate(
        results,
        baseline="default",
        predicted={"default": 1.0, "split": 2.0, "colocated": 3.0},
    )
    assert agreeing.prediction_agreement == 1.0
    assert agreeing.ranking[0].predicted_requests_per_s == 3.0
    disagreeing = select_candidate(
        results,
        baseline="default",
        predicted={"default": 3.0, "split": 2.0, "colocated": 1.0},
    )
    assert disagreeing.prediction_agreement == 0.0
    assert select_candidate(results, baseline="default").prediction_agreement is None
