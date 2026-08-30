from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, resolve_warmup


@pytest.mark.parametrize(
    ("warmup", "max_concurrency", "expected"),
    [
        (None, 16, 16),
        (None, 1, 1),
        (None, 0, 1),
        (0, 16, 0),
        (1, 16, 1),
        (5, 16, 5),
    ],
)
def test_resolve_warmup_defaults_to_concurrency(
    warmup: int | None,
    max_concurrency: int,
    expected: int,
) -> None:
    assert resolve_warmup(warmup, max_concurrency) == expected
    config = RunConfig(max_concurrency=max_concurrency, warmup=warmup)
    assert config.effective_warmup == expected


@pytest.mark.asyncio
async def test_warmup_matches_concurrency_without_touching_measured_samples() -> None:
    starts: list[float] = []
    seen: list[str] = []

    async def _send(_session, sample: str) -> RequestResult:
        starts.append(time.perf_counter())
        seen.append(sample)
        await asyncio.sleep(0.2)
        return RequestResult(request_id=sample, is_success=True)

    samples = ["a", "b", "c", "d"]
    runner = BenchmarkRunner(RunConfig(max_concurrency=4, disable_tqdm=True))
    await runner.run(samples, _send)

    assert len(seen) == len(samples) * 2
    # note (luojiaxuan): Warmup repeats one sample so the measured cohort does
    # not start with server-side per-sample caches already filled.
    assert set(seen[: len(samples)]) == {samples[0]}
    assert sorted(seen[len(samples) :]) == sorted(samples)
    # note (luojiaxuan): Four sequential 0.2s warmups would span 0.6s.
    warmup_starts = starts[: len(samples)]
    assert max(warmup_starts) - min(warmup_starts) < 0.1


@pytest.mark.asyncio
async def test_warmup_can_be_disabled_explicitly() -> None:
    seen: list[str] = []

    async def _send(_session, sample: str) -> RequestResult:
        seen.append(sample)
        return RequestResult(request_id=sample, is_success=True)

    runner = BenchmarkRunner(RunConfig(max_concurrency=4, warmup=0, disable_tqdm=True))
    await runner.run(["a", "b"], _send)

    assert seen == ["a", "b"]


@pytest.mark.asyncio
async def test_open_loop_arrivals_overlap_in_flight_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts: list[float] = []

    async def _send(_session, sample: str) -> RequestResult:
        starts.append(time.perf_counter())
        await asyncio.sleep(0.3)
        return RequestResult(request_id=sample, is_success=True)

    monkeypatch.setattr(np.random, "exponential", lambda _scale: 0.02)
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=0,
            request_rate=50,
            warmup=0,
            disable_tqdm=True,
        )
    )
    await runner.run(["a", "b", "c", "d", "e", "f", "g", "h"], _send)

    assert len(starts) == 8
    assert max(starts) - min(starts) < 0.25
