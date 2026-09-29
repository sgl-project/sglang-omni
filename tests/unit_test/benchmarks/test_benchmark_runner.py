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

    async def send(session, sample: str) -> RequestResult:
        starts.append(time.perf_counter())
        seen.append(sample)
        await asyncio.sleep(0.2)
        return RequestResult(request_id=sample, is_success=True)

    samples = ["a", "b", "c", "d"]
    runner = BenchmarkRunner(RunConfig(max_concurrency=4, disable_tqdm=True))
    await runner.run(samples, send)

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

    async def send(session, sample: str) -> RequestResult:
        seen.append(sample)
        return RequestResult(request_id=sample, is_success=True)

    runner = BenchmarkRunner(RunConfig(max_concurrency=4, warmup=0, disable_tqdm=True))
    await runner.run(["a", "b"], send)

    assert seen == ["a", "b"]


@pytest.mark.asyncio
async def test_open_loop_arrivals_overlap_in_flight_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts: list[float] = []

    async def send(session, sample: str) -> RequestResult:
        starts.append(time.perf_counter())
        await asyncio.sleep(0.3)
        return RequestResult(request_id=sample, is_success=True)

    class _FixedGaps:
        def exponential(self, scale, size):
            return np.full(size, 0.02)

    monkeypatch.setattr(np.random, "default_rng", lambda _seed: _FixedGaps())
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=0,
            request_rate=50,
            warmup=0,
            disable_tqdm=True,
        )
    )
    await runner.run(["a", "b", "c", "d", "e", "f", "g", "h"], send)

    assert len(starts) == 8
    assert max(starts) - min(starts) < 0.25


@pytest.mark.asyncio
async def test_requests_that_queue_for_a_client_slot_are_marked() -> None:
    async def _send(_session, sample: str) -> RequestResult:
        await asyncio.sleep(0.05)
        return RequestResult(request_id=sample, is_success=True)

    runner = BenchmarkRunner(RunConfig(max_concurrency=1, warmup=0, disable_tqdm=True))
    results = await runner.run(["a", "b", "c"], _send)

    # note (luojiaxuan): with one slot and instant arrivals only the first
    # request starts on time; the rest waited, so their clocks started late.
    assert [r.waited_for_slot for r in results] == [False, True, True]


@pytest.mark.asyncio
async def test_requests_that_get_a_slot_at_once_are_not_marked() -> None:
    async def _send(_session, sample: str) -> RequestResult:
        await asyncio.sleep(0.05)
        return RequestResult(request_id=sample, is_success=True)

    runner = BenchmarkRunner(RunConfig(max_concurrency=8, warmup=0, disable_tqdm=True))
    results = await runner.run(["a", "b", "c"], _send)

    assert not any(r.waited_for_slot for r in results)


def arrival_offsets(seed: int, rate: float, count: int) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).exponential(1.0 / rate, count))


@pytest.mark.asyncio
async def test_a_seeded_run_offers_the_same_arrival_sequence_every_time() -> None:
    async def _run() -> list[float]:
        starts: list[float] = []
        loop = asyncio.get_running_loop()

        async def _send(_session, sample: str) -> RequestResult:
            starts.append(loop.time())
            return RequestResult(request_id=sample, is_success=True)

        runner = BenchmarkRunner(
            RunConfig(
                max_concurrency=0,
                request_rate=100,
                warmup=0,
                disable_tqdm=True,
                arrival_seed=7,
            )
        )
        await runner.run([str(i) for i in range(6)], _send)
        return [t - starts[0] for t in starts]

    first, second = await _run(), await _run()
    expected = arrival_offsets(7, 100, 6)
    expected = expected - expected[0]
    # note (luojiaxuan): sends land on the seeded schedule to within the
    # event loop's timer slack, run after run.
    assert np.allclose(first, expected, atol=0.01)
    assert np.allclose(second, expected, atol=0.01)


@pytest.mark.asyncio
async def test_a_late_send_is_recorded_and_does_not_shift_later_arrivals() -> None:
    async def _send(_session, sample: str) -> RequestResult:
        if sample == "0":
            # note (luojiaxuan): block the event loop so the next sends are
            # late against their plan.
            time.sleep(0.15)
        return RequestResult(request_id=sample, is_success=True)

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=0,
            request_rate=50,
            warmup=0,
            disable_tqdm=True,
            arrival_seed=3,
        )
    )
    results = await runner.run([str(i) for i in range(30)], _send)
    lateness = [r.dispatch_lateness_s for r in results]

    assert all(value is not None and value >= 0 for value in lateness)
    assert max(lateness) > 0.05
    # note (luojiaxuan): arrivals planned after the stall are sent on time
    # again instead of inheriting the delay.
    offsets = arrival_offsets(3, 50, 30)
    on_time = [
        value for value, offset in zip(lateness, offsets) if offset > offsets[0] + 0.2
    ]
    assert on_time and max(on_time) < 0.02


@pytest.mark.asyncio
async def test_closed_loop_runs_do_not_report_dispatch_lateness() -> None:
    async def _send(_session, sample: str) -> RequestResult:
        return RequestResult(request_id=sample, is_success=True)

    runner = BenchmarkRunner(RunConfig(max_concurrency=2, warmup=0, disable_tqdm=True))
    results = await runner.run(["a", "b"], _send)

    assert all(r.dispatch_lateness_s is None for r in results)
