from dataclasses import replace

import pytest

from sglang_omni.restage.evaluation import SLO, Observation, evaluate
from sglang_omni.restage.search import search_rates


def verdict(good):
    return evaluate(
        [Observation("a", 0, 0.1, True, good)],
        SLO(max_latency_s=1),
        expected_requests=1,
        elapsed_s=1,
    )


@pytest.mark.asyncio
async def test_single_failed_repeat_excludes_load_point():
    calls = []

    async def trial(rate, repeat):
        calls.append((rate, repeat))
        return verdict(rate == 1 or repeat != 1)

    result = await search_rates(trial, [1, 2], repeats=3)
    assert len(calls) == 6
    assert result.best_tested_rate == 1
    assert result.upper_limit_passed is False
    assert len(result.points[1].evaluations) == 3


@pytest.mark.asyncio
async def test_all_passing_rates_leave_capacity_unbounded_above():
    async def trial(rate, repeat):
        return verdict(True)

    result = await search_rates(trial, [1, 2], repeats=2)
    assert result.best_tested_rate == 2
    assert result.upper_limit_passed is True


@pytest.mark.asyncio
async def test_no_feasible_point_means_no_recommendation_even_with_high_throughput():
    async def trial(rate, repeat):
        return replace(verdict(False), completed_qps=1000)

    result = await search_rates(trial, [1, 2], repeats=2)
    assert result.best_tested_rate is None


@pytest.mark.asyncio
async def test_nonmonotonic_results_are_reported_and_not_interpolated():
    async def trial(rate, repeat):
        return verdict(rate != 2)

    result = await search_rates(trial, [1, 2, 3], repeats=2)
    assert result.nonmonotonic is True
    assert result.best_tested_rate == 3


@pytest.mark.asyncio
async def test_invalid_rates_fail_before_trial():
    async def trial(rate, repeat):
        raise AssertionError("must validate before measuring")

    with pytest.raises(ValueError, match="positive"):
        await search_rates(trial, [0, 1], repeats=2)
