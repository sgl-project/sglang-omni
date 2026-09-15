"""Select among candidates measured with one workload, SLO and load grid."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

from sglang_omni.restage.search import RateSearchResult


@dataclass(frozen=True)
class CandidateRank:
    key: str
    best_tested_rate: float | None
    passing_prefix_rate: float | None
    median_goodput_qps: float | None
    predicted_requests_per_s: float | None
    upper_limit_passed: bool
    nonmonotonic: bool
    reason: str


@dataclass(frozen=True)
class Selection:
    recommended: str | None
    ranking: tuple[CandidateRank, ...]
    baseline: str
    baseline_rate: float | None
    rate_gain_over_baseline: float | None
    prediction_agreement: float | None
    scope: str = "best among measured candidates at tested rates; not global optimality"


def select_candidate(
    results: Mapping[str, RateSearchResult],
    *,
    baseline: str,
    predicted: Mapping[str, float] | None = None,
) -> Selection:
    """Prefer the highest passing prefix of the tested grid, then median goodput.

    The caller must hold model, workload, SLO and hardware budget constant.
    Equal scores retain the baseline; other exact ties use the candidate key.
    ``predicted`` maps candidates to the planner's requests/s; the agreement is
    the fraction of candidate pairs whose measured order matches it.
    """
    if baseline not in results:
        raise ValueError("The measured baseline must be included")
    grid = [(p.rate, len(p.evaluations)) for p in results[baseline].points]
    if not grid or any(
        [(p.rate, len(p.evaluations)) for p in result.points] != grid
        for result in results.values()
    ):
        raise ValueError("Candidates must use the same rates and repeat counts")
    predicted = dict(predicted or {})
    ranking = []
    for key, result in results.items():
        passing = [point for point in result.points if point.feasible]
        highest = max(passing, key=lambda point: point.rate) if passing else None
        best = None
        for point in result.points:
            if not point.feasible:
                break
            best = point
        reason = (
            f"All {len(best.evaluations)} repeats passed at every tested rate "
            f"up to {best.rate:g} requests/s"
            if best is not None
            else "No recommendation: no tested rate in a passing prefix"
        )
        if result.upper_limit_passed:
            reason += "; upper limit passed, capacity boundary remains unmeasured"
        if result.nonmonotonic:
            reason += "; passing above a failed lower rate, requires confirmation"
        ranking.append(
            CandidateRank(
                key=key,
                best_tested_rate=highest.rate if highest else None,
                passing_prefix_rate=best.rate if best else None,
                median_goodput_qps=best.median_goodput_qps if best else None,
                predicted_requests_per_s=predicted.get(key),
                upper_limit_passed=result.upper_limit_passed,
                nonmonotonic=result.nonmonotonic,
                reason=reason,
            )
        )
    ranking.sort(
        key=lambda row: (
            -(row.passing_prefix_rate or 0),
            -(row.median_goodput_qps or 0),
            row.key != baseline,
            row.key,
        )
    )
    winner = ranking[0] if ranking[0].passing_prefix_rate is not None else None
    baseline_rate = next(
        row.passing_prefix_rate for row in ranking if row.key == baseline
    )
    return Selection(
        recommended=winner.key if winner else None,
        ranking=tuple(ranking),
        baseline=baseline,
        baseline_rate=baseline_rate,
        rate_gain_over_baseline=(
            winner.passing_prefix_rate / baseline_rate
            if winner is not None and baseline_rate is not None
            else None
        ),
        prediction_agreement=_agreement(ranking),
    )


def _agreement(ranking):
    scored = [
        row
        for row in ranking
        if row.predicted_requests_per_s is not None
        and row.passing_prefix_rate is not None
    ]
    pairs = [
        (a, b)
        for a, b in combinations(scored, 2)
        if a.predicted_requests_per_s != b.predicted_requests_per_s
    ]
    if not pairs:
        return None
    agree = sum(
        (a.predicted_requests_per_s > b.predicted_requests_per_s)
        == (
            (a.passing_prefix_rate, a.median_goodput_qps or 0)
            > (b.passing_prefix_rate, b.median_goodput_qps or 0)
        )
        for a, b in pairs
    )
    return agree / len(pairs)
