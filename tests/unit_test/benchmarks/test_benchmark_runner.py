"""Benchmark runner warmup, arrival schedules, client slots and dispatch lateness."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiohttp
import numpy as np
import pytest
from aiohttp import web

import benchmarks.benchmarker.runner as runner_module
from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, resolve_warmup


@pytest.mark.asyncio
@pytest.mark.parametrize("trust_env", [False, True])
async def test_environment_proxy_is_opt_in(monkeypatch, trust_env: bool) -> None:
    async def proxy(_request):
        return web.Response(text="via proxy")

    app = web.Application()
    app.router.add_get("/{path:.*}", proxy)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = server.addresses[0][1]
    monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("no_proxy", "")

    async def send(session, sample):
        assert session.trust_env is trust_env
        if not trust_env:
            return RequestResult(request_id=sample, is_success=True)
        async with session.get(
            "http://benchmark-proxy-test.invalid/result"
        ) as response:
            return RequestResult(
                request_id=sample, text=await response.text(), is_success=True
            )

    try:
        runner = BenchmarkRunner(RunConfig(warmup=0, trust_env=trust_env, timeout_s=2))
        results = await runner.run(["one"], send)
        assert results[0].text == ("via proxy" if trust_env else "")
    finally:
        await server.cleanup()


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


@pytest.mark.asyncio
async def test_after_send_runs_outside_the_slot_and_the_timed_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second_started = asyncio.Event()
    timer_stopped = asyncio.Event()
    followed: list[str] = []
    release_followups = asyncio.Event()
    clock_seconds = 10.0

    def read_clock() -> float:
        if second_started.is_set():
            timer_stopped.set()
        else:
            pass
        return clock_seconds

    monkeypatch.setattr(runner_module, "time", SimpleNamespace(perf_counter=read_clock))

    async def send(session: aiohttp.ClientSession, sample: str) -> RequestResult:
        nonlocal clock_seconds
        if sample == "b":
            clock_seconds = 12.0
            second_started.set()
        else:
            pass
        return RequestResult(request_id=sample, is_success=True)

    async def after_send(result: RequestResult) -> None:
        if result.request_id == "a":
            await asyncio.wait_for(second_started.wait(), timeout=1)
        else:
            pass
        await release_followups.wait()
        followed.append(result.request_id)

    runner = BenchmarkRunner(RunConfig(max_concurrency=1, warmup=2, disable_tqdm=True))
    task = asyncio.create_task(runner.run(["a", "b"], send, after_send=after_send))
    try:
        await asyncio.wait_for(timer_stopped.wait(), timeout=2)
        assert runner.wall_clock_s == 2.0
        assert not task.done()
        assert not followed
    finally:
        clock_seconds = 112.0
        release_followups.set()
        results = await asyncio.wait_for(task, timeout=2)

    assert [r.request_id for r in results] == ["a", "b"]
    assert sorted(followed) == ["a", "b"]
    assert runner.wall_clock_s == 2.0


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


def test_run_config_preserves_positional_arrival_seed() -> None:
    config = RunConfig(4, 2.0, 0, True, 60, 42)
    assert config.arrival_seed == 42
    assert config.trust_env is False


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["warmup", "dispatch", "followup", "arrival"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_runner_drains_tasks_before_propagating_failure(
    monkeypatch: pytest.MonkeyPatch, phase: str, cancel: bool
) -> None:
    """A failed or cancelled cohort previously left sibling requests running."""
    generator = MagicMock()
    generator.exponential.return_value = np.array([0.0, 0.0, 3600.0])
    monkeypatch.setattr(
        runner_module.np.random, "default_rng", MagicMock(return_value=generator)
    )
    sibling_started = asyncio.Event()
    never_complete = asyncio.Event()
    started = 0
    finished = 0

    async def work() -> None:
        nonlocal started, finished
        started += 1
        index = started
        try:
            if index == 1:
                await sibling_started.wait()
                if cancel:
                    await never_complete.wait()
                else:
                    raise BufferError("audio budget exceeded")
            else:
                sibling_started.set()
                await never_complete.wait()
        finally:
            finished += 1

    async def send(session: aiohttp.ClientSession, sample: str) -> RequestResult:
        assert not session.closed
        if phase == "followup":
            pass
        else:
            await work()
        return RequestResult(request_id=sample, is_success=True)

    async def after_send(result: RequestResult) -> None:
        assert result.is_success
        await work()

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=2,
            warmup=2 if phase == "warmup" else 0,
            disable_tqdm=True,
            request_rate=1.0 if phase == "arrival" else float("inf"),
        )
    )
    task = asyncio.create_task(
        runner.run(
            (
                ["first", "second", "future"]
                if phase == "arrival"
                else ["first", "second"]
            ),
            send,
            after_send=after_send if phase == "followup" else None,
        )
    )
    await asyncio.wait_for(sibling_started.wait(), timeout=3)
    if cancel:
        task.cancel()
        expected_error = asyncio.CancelledError
    else:
        expected_error = BufferError
    with pytest.raises(expected_error):
        await asyncio.wait_for(task, timeout=3)
    assert started == 2
    assert finished == 2


@pytest.mark.asyncio
async def test_generation_failure_drains_pending_outcomes() -> None:
    """A generation failure must drain outcome lookups already running."""
    lookup_started = asyncio.Event()
    sibling_started = asyncio.Event()
    never_complete = asyncio.Event()
    finished: set[str] = set()

    async def send(session: aiohttp.ClientSession, sample: str) -> RequestResult:
        assert not session.closed
        if sample == "completed":
            return RequestResult(request_id=sample, is_success=True)
        elif sample == "failing":
            await lookup_started.wait()
            await sibling_started.wait()
            raise BufferError("audio budget exceeded")
        else:
            sibling_started.set()
            try:
                await never_complete.wait()
            finally:
                finished.add("generation")
            raise AssertionError("sibling request must be cancelled")

    async def after_send(result: RequestResult) -> None:
        assert result.request_id == "completed"
        lookup_started.set()
        try:
            await never_complete.wait()
        finally:
            finished.add("outcome")

    runner = BenchmarkRunner(RunConfig(max_concurrency=2, warmup=0, disable_tqdm=True))
    with pytest.raises(BufferError, match="audio budget exceeded"):
        await asyncio.wait_for(
            runner.run(
                ["completed", "failing", "pending"], send, after_send=after_send
            ),
            timeout=3,
        )
    assert finished == {"generation", "outcome"}


@pytest.mark.asyncio
async def test_due_arrival_is_sent_before_future_samples_are_scheduled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future task creation previously delayed an already-due request."""
    generator = MagicMock()
    generator.exponential.return_value = np.array([0.0, 3600.0, 0.0])
    monkeypatch.setattr(
        runner_module.np.random, "default_rng", MagicMock(return_value=generator)
    )
    visited_sample_ids: list[str] = []
    visited_at_first_send: list[str] = []
    first_sent = asyncio.Event()

    class ScheduledSamples(list[str]):
        def __iter__(self) -> Iterator[str]:
            for sample_id in super().__iter__():
                visited_sample_ids.append(sample_id)
                yield sample_id

    async def send(session: aiohttp.ClientSession, sample_id: str) -> RequestResult:
        assert not session.closed
        assert sample_id == "first"
        visited_at_first_send.extend(visited_sample_ids)
        first_sent.set()
        return RequestResult(request_id=sample_id, is_success=True)

    runner = BenchmarkRunner(
        RunConfig(max_concurrency=0, request_rate=1, warmup=0, disable_tqdm=True)
    )
    benchmark_task = asyncio.create_task(
        runner.run(ScheduledSamples(["first", "next", "future"]), send)
    )
    try:
        await asyncio.wait_for(first_sent.wait(), timeout=3)
        assert "future" not in visited_at_first_send
    finally:
        benchmark_task.cancel()
        await asyncio.gather(benchmark_task, return_exceptions=True)
