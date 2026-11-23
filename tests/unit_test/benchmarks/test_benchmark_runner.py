"""Benchmark runner warmup, arrival schedules, client slots, dispatch lateness and fixed duration runs."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import aiohttp
import numpy as np
import pytest
from aiohttp import web

import benchmarks.benchmarker.runner as runner_module
from benchmarks.benchmarker.data import RequestClass, RequestResult
from benchmarks.benchmarker.runner import (
    BenchmarkRunner,
    ClientGroup,
    RunConfig,
    SendFn,
    resolve_warmup,
    run_fixed_duration,
)
from benchmarks.dataset.seedtts import SampleInput

POOL = [
    SampleInput(sample_id=name, ref_text="r", ref_audio="r.wav", target_text=name)
    for name in ("a", "b", "c", "d")
]


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


class FakeClock:
    def __init__(self, now_s: float) -> None:
        self.now_s = now_s

    def perf_counter(self) -> float:
        return self.now_s


def install_fake_clock(monkeypatch: pytest.MonkeyPatch, now_s: float) -> FakeClock:
    clock = FakeClock(now_s)
    monkeypatch.setattr(
        runner_module, "time", SimpleNamespace(perf_counter=clock.perf_counter)
    )
    return clock


@pytest.mark.asyncio
async def test_fixed_duration_run_keeps_each_client_closed_loop_until_the_window_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No send at or after the deadline, no cancelled request, one cursor per class."""
    clock = install_fake_clock(monkeypatch, 100.0)
    sent_ids: list[str] = []
    in_flight = {"non_streaming": 0, "streaming": 0}
    peak_in_flight = {"non_streaming": 0, "streaming": 0}

    def make_send(request_class: RequestClass) -> SendFn:
        async def send(
            session: aiohttp.ClientSession, sample: SampleInput
        ) -> RequestResult:
            sent_ids.append(sample.sample_id)
            in_flight[request_class] += 1
            peak_in_flight[request_class] = max(
                peak_in_flight[request_class], in_flight[request_class]
            )
            clock.now_s += 0.5
            await asyncio.sleep(0)
            in_flight[request_class] -= 1
            return RequestResult(request_id=sample.sample_id, is_success=True)

        return send

    run = await run_fixed_duration(
        [
            ClientGroup(
                request_class="non_streaming",
                clients=2,
                send_fn=make_send("non_streaming"),
                first_sample_index=0,
            ),
            ClientGroup(
                request_class="streaming",
                clients=1,
                send_fn=make_send("streaming"),
                first_sample_index=2,
            ),
        ],
        POOL,
        settle_s=1.0,
        duration_s=3.0,
        should_warm_up=False,
        timeout_s=5,
        after_send=None,
    )

    assert sorted(result.request_id for result in run.results) == sorted(sent_ids)
    assert [sample.sample_id for sample in run.issued_samples] == sent_ids
    assert peak_in_flight == {"non_streaming": 2, "streaming": 1}
    assert min(result.sent_at_s for result in run.results) == -1.0
    assert max(result.finished_at_s for result in run.results) == clock.now_s - 101.0
    assert all(result.sent_at_s < 3.0 for result in run.results)
    assert all(result.finished_at_s > result.sent_at_s for result in run.results)
    request_numbers: dict[RequestClass, list[int]] = {
        "non_streaming": [],
        "streaming": [],
    }
    first_sample_indexes: dict[RequestClass, int] = {"non_streaming": 0, "streaming": 2}
    for result in run.results:
        sample_id, suffix = result.request_id.split("#")
        request_class = result.request_class
        assert suffix[0] == {"non_streaming": "n", "streaming": "s"}[request_class]
        request_number = int(suffix[1:])
        expected_index = (first_sample_indexes[request_class] + request_number) % 4
        assert sample_id == POOL[expected_index].sample_id
        request_numbers[request_class].append(request_number)
    for numbers in request_numbers.values():
        assert sorted(numbers) == list(range(len(numbers)))
    assert len(request_numbers["non_streaming"]) > len(POOL)


@pytest.mark.asyncio
async def test_fixed_duration_warmup_runs_each_client_once_and_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = install_fake_clock(monkeypatch, 0.0)
    calls: list[tuple[RequestClass, str]] = []
    failing_sample_ids: set[str] = set()

    def make_send(request_class: RequestClass) -> SendFn:
        async def send(
            session: aiohttp.ClientSession, sample: SampleInput
        ) -> RequestResult:
            calls.append((request_class, sample.sample_id))
            clock.now_s += 10.0
            is_success = sample.sample_id not in failing_sample_ids
            return RequestResult(
                request_id=sample.sample_id,
                is_success=is_success,
                error="" if is_success else "HTTP 500: boom",
            )

        return send

    groups = [
        ClientGroup(
            request_class="non_streaming",
            clients=2,
            send_fn=make_send("non_streaming"),
            first_sample_index=0,
        ),
        ClientGroup(
            request_class="streaming",
            clients=1,
            send_fn=make_send("streaming"),
            first_sample_index=2,
        ),
    ]
    warmup_calls = [("non_streaming", "a"), ("non_streaming", "a"), ("streaming", "c")]

    run = await run_fixed_duration(
        groups,
        POOL,
        settle_s=0.0,
        duration_s=1.0,
        should_warm_up=True,
        timeout_s=5,
        after_send=None,
    )
    assert sorted(calls[:3]) == warmup_calls
    assert run.results
    assert all("#" in result.request_id for result in run.results)

    calls.clear()
    failing_sample_ids.add("c")
    with pytest.raises(ValueError, match="Warmup failed"):
        await run_fixed_duration(
            groups,
            POOL,
            settle_s=0.0,
            duration_s=1.0,
            should_warm_up=True,
            timeout_s=5,
            after_send=None,
        )
    assert sorted(calls) == warmup_calls


@pytest.mark.asyncio
async def test_fixed_duration_follow_ups_run_outside_the_client_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow stream outcome lookup must not add think time between a client's requests."""
    clock = install_fake_clock(monkeypatch, 0.0)
    deadline_reached = asyncio.Event()
    release_followups = asyncio.Event()
    followed: list[str] = []

    async def send(
        session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult:
        clock.now_s += 1.0
        if clock.now_s >= 3.0:
            deadline_reached.set()
        else:
            pass
        await asyncio.sleep(0)
        return RequestResult(request_id=sample.sample_id, is_success=True)

    async def after_send(result: RequestResult) -> None:
        await release_followups.wait()
        followed.append(result.request_id)

    task = asyncio.create_task(
        run_fixed_duration(
            [
                ClientGroup(
                    request_class="streaming",
                    clients=1,
                    send_fn=send,
                    first_sample_index=0,
                )
            ],
            POOL,
            settle_s=0.0,
            duration_s=3.0,
            should_warm_up=False,
            timeout_s=5,
            after_send=after_send,
        )
    )
    try:
        await asyncio.wait_for(deadline_reached.wait(), timeout=2)
        for _ in range(20):
            await asyncio.sleep(0)
        assert not task.done()
    finally:
        release_followups.set()
        run = await asyncio.wait_for(task, timeout=2)

    assert len(run.results) == 3
    assert sorted(followed) == sorted(result.request_id for result in run.results)


@pytest.mark.asyncio
async def test_a_client_stops_after_its_first_failed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead server must not turn the window into a loop of instant errors."""
    clock = install_fake_clock(monkeypatch, 0.0)
    sender_by_request_id: dict[str, asyncio.Task[None] | None] = {}

    async def send(
        session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult:
        sender_by_request_id[sample.sample_id] = asyncio.current_task()
        clock.now_s += 1.0
        await asyncio.sleep(0)
        is_success = not sample.sample_id.endswith("#n1")
        return RequestResult(
            request_id=sample.sample_id,
            is_success=is_success,
            error="" if is_success else "HTTP 500: boom",
        )

    run = await run_fixed_duration(
        [
            ClientGroup(
                request_class="non_streaming",
                clients=2,
                send_fn=send,
                first_sample_index=0,
            )
        ],
        POOL,
        settle_s=0.0,
        duration_s=6.0,
        should_warm_up=False,
        timeout_s=5,
        after_send=None,
    )

    failures = [result for result in run.results if not result.is_success]
    assert [result.request_id for result in failures] == ["b#n1"]
    failing_client = sender_by_request_id["b#n1"]
    failing_client_results = [
        result
        for result in run.results
        if sender_by_request_id[result.request_id] is failing_client
    ]
    assert failing_client_results[-1] is failures[0]
    other_client_finishes_s = [
        result.finished_at_s
        for result in run.results
        if sender_by_request_id[result.request_id] is not failing_client
    ]
    assert max(other_client_finishes_s) >= 6.0


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
