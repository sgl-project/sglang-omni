"""Repeated measured load sweeps with explicit finite-search conclusions."""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from statistics import median

from sglang_omni.restage.evaluation import Evaluation


@dataclass(frozen=True)
class RatePoint:
    rate: float
    evaluations: tuple[Evaluation, ...]

    @property
    def feasible(self) -> bool:
        return bool(self.evaluations) and all(e.feasible for e in self.evaluations)

    @property
    def median_goodput_qps(self) -> float:
        return median(e.goodput_qps for e in self.evaluations)


@dataclass(frozen=True)
class RateSearchResult:
    points: tuple[RatePoint, ...]
    best_tested_rate: float | None
    upper_limit_passed: bool
    nonmonotonic: bool


async def search_rates(
    trial: Callable[[float, int], Awaitable[Evaluation]],
    rates: Sequence[float],
    *,
    repeats: int = 3,
) -> RateSearchResult:
    """Measure every declared load repeatedly and retain the full evidence.

    Passing every repeat is an empirical acceptance rule, not a statistical
    confidence guarantee. No rate between or beyond tested points is inferred.
    The callback persists trial evidence and owns failure/cleanup behavior.
    """
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Rates must be nonempty, finite and positive")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    ordered = sorted(set(rates))
    evaluations = {rate: [] for rate in ordered}
    for repeat in range(repeats):
        for rate in ordered:
            evaluations[rate].append(await trial(rate, repeat))
    points = tuple(RatePoint(rate, tuple(evaluations[rate])) for rate in ordered)
    feasible = [point.rate for point in points if point.feasible]
    failed_below = False
    nonmonotonic = False
    for point in points:
        if point.feasible and failed_below:
            nonmonotonic = True
        if not point.feasible:
            failed_below = True
    return RateSearchResult(
        points=points,
        best_tested_rate=max(feasible) if feasible else None,
        upper_limit_passed=points[-1].feasible,
        nonmonotonic=nonmonotonic,
    )
