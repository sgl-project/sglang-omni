from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import numpy as np
import pytest
from aiohttp import web

import benchmarks.benchmarker.runner as runner_module
from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, resolve_warmup
from benchmarks.benchmarker.utils import save_json_results
from benchmarks.metrics.performance import compute_speed_metrics
from sglang_omni.utils.json import JsonValue


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


def reject_nonfinite_json_constant(token: str) -> None:
    raise ValueError(f"Invalid JSON constant: {token}")


@pytest.mark.asyncio
@pytest.mark.parametrize("request_rate", [float("inf"), 200.0])
async def test_benchmark_results_are_strict_json(
    tmp_path: Path, request_rate: float
) -> None:
    async def send(session: aiohttp.ClientSession, sample_id: str) -> RequestResult:
        assert isinstance(session, aiohttp.ClientSession)
        return RequestResult(
            request_id=sample_id, is_success=True, latency_s=0.25, completion_tokens=4
        )

    config = RunConfig(request_rate=request_rate, warmup=0, disable_tqdm=True)
    runner = BenchmarkRunner(config)
    outputs = await runner.run(["one", "two"], send)
    speed = compute_speed_metrics(outputs, wall_clock_s=runner.wall_clock_s)
    results = {
        "config": asdict(config),
        "speed": speed,
        "per_request": [asdict(output) for output in outputs],
    }
    path = save_json_results(results, str(tmp_path), "results.json")
    saved = json.loads(
        Path(path).read_text(), parse_constant=reject_nonfinite_json_constant
    )
    assert saved["config"]["request_rate"] == (
        "inf" if request_rate == float("inf") else request_rate
    )
    assert saved["speed"] == speed
    assert len(saved["per_request"]) == 2
    assert results["config"]["request_rate"] == request_rate


def test_nested_sweep_results_are_strict_json(tmp_path: Path) -> None:
    run = {"config": {"request_rate": float("inf")}, "speed": {"throughput_qps": 2.5}}
    results = {"runs": [{"repeats": [run]}], "combined": {"generate": run}}
    path = save_json_results(results, str(tmp_path), "sweep.json")
    saved = json.loads(
        Path(path).read_text(), parse_constant=reject_nonfinite_json_constant
    )
    expected = {"config": {"request_rate": "inf"}, "speed": {"throughput_qps": 2.5}}
    assert saved == {
        "runs": [{"repeats": [expected]}],
        "combined": {"generate": expected},
    }
    assert run["config"]["request_rate"] == float("inf")


@pytest.mark.parametrize(
    "invalid_result",
    [
        {"speed": {"latency_mean_s": float("nan")}},
        {"speed": {"latency_mean_s": float("inf")}},
        {"speed": {"latency_mean_s": float("-inf")}},
        {"config": {"request_rate": float("nan")}},
        {"config": {"request_rate": float("-inf")}},
    ],
)
def test_invalid_benchmark_results_preserve_existing_artifact(
    tmp_path: Path, invalid_result: dict[str, JsonValue]
) -> None:
    path = tmp_path / "results.json"
    previous_results = '{"speed": {"completed_requests": 3}}\n'
    path.write_text(previous_results)
    with pytest.raises(ValueError, match="Out of range float values"):
        save_json_results({"runs": [invalid_result]}, str(tmp_path), path.name)
    assert path.read_text() == previous_results
