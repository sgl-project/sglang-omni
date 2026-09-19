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


@pytest.mark.asyncio
async def test_request_timestamps_include_client_queue_wait():
    async def send(_session, sample):
        await asyncio.sleep(0.01)
        return RequestResult(request_id=sample, is_success=True)

    runner = BenchmarkRunner(RunConfig(max_concurrency=1, warmup=0, disable_tqdm=True))
    first, second = await runner.run(["first", "second"], send)
    assert first.scheduled_s <= first.dispatched_s <= first.completed_s
    assert second.scheduled_s < first.completed_s <= second.dispatched_s
    assert second.completed_s >= second.dispatched_s


@pytest.mark.asyncio
async def test_arrival_schedule_does_not_drift_with_blocked_event_loop(monkeypatch):
    monkeypatch.setattr(np.random, "exponential", lambda _scale: 0.01)

    async def send(_session, sample):
        if sample == "first":
            time.sleep(0.04)
        return RequestResult(request_id=sample, is_success=True)

    runner = BenchmarkRunner(
        RunConfig(max_concurrency=0, request_rate=100, warmup=0, disable_tqdm=True)
    )
    results = await runner.run(["first", "second", "third"], send)
    assert results[1].scheduled_s - results[0].scheduled_s == pytest.approx(0.01)
    assert results[2].scheduled_s - results[1].scheduled_s == pytest.approx(0.01)
    assert results[1].dispatched_s - results[1].scheduled_s > 0.02


@pytest.mark.asyncio
async def test_dispatch_failure_cancels_remaining_requests_before_return():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def send(_session, sample):
        if sample == "fail":
            await started.wait()
            raise RuntimeError("dispatch failed")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    runner = BenchmarkRunner(RunConfig(max_concurrency=0, warmup=0, disable_tqdm=True))
    with pytest.raises(RuntimeError, match="dispatch failed"):
        await runner.run(["fail", "pending"], send)
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_seeded_arrivals_repeat_independently_of_sender_randomness():
    async def send(session, sample):
        np.random.exponential(size=17)
        return RequestResult(request_id=sample, is_success=True)

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=0,
            request_rate=1000,
            warmup=0,
            disable_tqdm=True,
            arrival_seed=42,
        )
    )
    first = await runner.run(list("abcd"), send)
    second = await runner.run(list("abcd"), send)
    first_gaps = np.diff([r.scheduled_s for r in first])
    second_gaps = np.diff([r.scheduled_s for r in second])
    assert first_gaps == pytest.approx(second_gaps, abs=1e-8)
    expected = np.random.default_rng(42).exponential(0.001, size=4)[1:]
    assert first_gaps == pytest.approx(expected, abs=1e-8)


@pytest.mark.asyncio
async def test_separate_warmup_sample_is_excluded_from_measurement(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    sent = []
    recorded = []

    async def send(session, sample):
        sent.append(sample)
        clock[0] += 50 if sample.startswith("warm") else 1
        return RequestResult(request_id=sample, is_success=True)

    runner = BenchmarkRunner(
        RunConfig(max_concurrency=0, warmup=3, disable_tqdm=True),
        on_result=lambda result: recorded.append(result.request_id),
    )
    results = await runner.run(
        ["measured-a", "measured-b"], send, warmup_sample="warm-a"
    )
    assert sent == ["warm-a", "warm-a", "warm-a", "measured-a", "measured-b"]
    assert recorded == ["measured-a", "measured-b"]
    assert [result.request_id for result in results] == recorded
    assert runner.wall_clock_s == 2


@pytest.mark.asyncio
async def test_separate_warmup_sample_respects_zero_count():
    seen = []

    async def send(session, sample):
        seen.append(sample)
        return RequestResult(request_id=sample, is_success=True)

    runner = BenchmarkRunner(RunConfig(warmup=0, disable_tqdm=True))
    await runner.run(["measured"], send, warmup_sample="warm")
    assert seen == ["measured"]
