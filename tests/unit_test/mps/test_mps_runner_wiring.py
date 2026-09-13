# SPDX-License-Identifier: Apache-2.0
"""Thin runner-to-MPS integration contracts."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from sglang_omni.config import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.mps.control import MPS_CLIENT_TOKEN_ENV, MpsDirtyStateError
from sglang_omni.mps.runtime import MpsPipelineRuntime
from sglang_omni.pipeline import mp_runner
from sglang_omni.pipeline.stage_workers import (
    StageGroup,
    StageLaunchConfig,
    StageWorkerProcessSpec,
)
from tests.unit_test.mps.test_mps_runtime import FakeControlClient

FAKE_GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-000000000000"


@pytest.fixture
def short_base():
    with tempfile.TemporaryDirectory(prefix="runner-", dir="/tmp") as root:
        yield Path(root)


def noop_factory():  # pragma: no cover - never constructed in these tests
    raise AssertionError("factory must not run")


def _make_config(base_path: Path, *, mps: str = "auto") -> PipelineConfig:
    return PipelineConfig(
        mps=mps,
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        entry_stage="preprocessing",
        stages=[
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory_path=f"{__name__}.noop_factory",
                terminal=True,
            )
        ],
        endpoints=EndpointsConfig(base_path=str(base_path)),
    )


class _FakeCoordinator:
    def __init__(self, events: list[str], *args, **kwargs) -> None:
        del args, kwargs
        self.events = events
        self.registered: dict[str, str] = {}

    async def start(self) -> None:
        return None

    async def run_completion_loop(self) -> None:
        await asyncio.Event().wait()

    def register_stage(self, name: str, endpoint: str) -> None:
        self.registered[name] = endpoint

    async def shutdown_stages(self) -> None:
        self.events.append("graceful shutdown")

    async def fail_pending_requests(self, error: BaseException) -> None:
        del error

    async def stop(self) -> None:
        self.events.append("coordinator stop")


class _FakeProcess:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.events.append("stage terminate")
        self._alive = False

    def kill(self) -> None:
        self.events.append("stage kill")
        self._alive = False

    def join(self, timeout=None) -> None:
        del timeout
        self.events.append("stage join")


class _FakeGroup:
    stage_control_endpoints = {"preprocessing": "ipc://preprocessing"}
    process_count = 1

    def __init__(
        self,
        events: list[str],
        *,
        ready_error: BaseException | None = None,
        shutdown_gate: tuple[asyncio.Event, asyncio.Event] | None = None,
        direct_process: bool = False,
    ) -> None:
        self.events = events
        self.ready_error = ready_error
        self.shutdown_gate = shutdown_gate
        self.processes = [_FakeProcess(events)] if direct_process else []
        self.spawn_env = object()
        self.before_signal = object()
        self.dead = False
        self.process_specs = [
            StageWorkerProcessSpec(
                process_name="pipeline",
                stage_specs=[
                    StageLaunchConfig(
                        stage_name="preprocessing",
                        factory=f"{__name__}.noop_factory",
                        placement_gpu_id=0,
                        gpu_id=0,
                        recv_endpoint="ipc://preprocessing",
                    )
                ],
            )
        ]

    def spawn(self, ctx, process_env_overrides=None) -> None:
        del ctx
        self.spawn_env = process_env_overrides
        self.events.append("spawn")

    async def wait_ready(self, timeout: float) -> None:
        del timeout
        self.events.append("ready")
        if self.ready_error is not None:
            raise self.ready_error

    def any_dead(self) -> bool:
        return self.dead

    def dead_summary(self) -> str:
        return "preprocessing exited" if self.dead else "(none)"

    async def shutdown(self, before_signal=None) -> None:
        self.before_signal = before_signal
        self.events.append("process shutdown")
        if self.shutdown_gate is not None:
            entered, release = self.shutdown_gate
            entered.set()
            await release.wait()

    def close_control_channels(self) -> None:
        self.events.append("channels closed")


class _FakeMps:
    def __init__(
        self,
        events: list[str],
        *,
        close_error: BaseException | None = None,
        probe_result: str | None = None,
        probe_gate: asyncio.Event | None = None,
    ) -> None:
        self.events = events
        self.close_error = close_error
        self.probe_result = probe_result
        self.probe_gate = probe_gate
        self.started = False

    @property
    def has_resources(self) -> bool:
        return self.started

    async def start(self, gpu_uuids) -> None:
        assert set(gpu_uuids) == {FAKE_GPU_UUID}
        self.events.append("MPS start")
        self.started = True

    def env_for_process(self, process_name: str) -> dict[str, str]:
        if process_name != "pipeline":
            return {}
        return {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps-pipe"}

    async def verify(self) -> None:
        self.events.append("MPS verify")

    async def retire_process_clients(self, process_name: str) -> None:
        self.events.append(f"MPS retire {process_name}")

    async def probe(self) -> str | None:
        if self.probe_gate is not None:
            await self.probe_gate.wait()
        return self.probe_result

    async def close(self) -> None:
        self.events.append("MPS close")
        self.started = False
        if self.close_error is not None:
            raise self.close_error


def _patch_runner(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    group: _FakeGroup,
    fake_mps: _FakeMps | None,
) -> _FakeCoordinator:
    coordinator = _FakeCoordinator(events)
    monkeypatch.setattr(
        mp_runner,
        "Coordinator",
        lambda *args, **kwargs: coordinator,
    )
    monkeypatch.setattr(
        mp_runner,
        "_build_stage_groups",
        lambda *args, **kwargs: [group],
    )
    if fake_mps is not None:
        monkeypatch.setattr(
            mp_runner,
            "create_for_pipeline",
            lambda mode, specs: (fake_mps, {"pipeline": FAKE_GPU_UUID}),
        )
    return coordinator


class _SpawnQueue:
    def close(self) -> None:
        return None

    def join_thread(self) -> None:
        return None


class _PreStartFailureContext:
    def Event(self):
        raise OSError("process synchronization resource exhausted")


class _ProcessStartFailure:
    def start(self) -> None:
        raise OSError("Process.start failed")


class _ProcessStartFailureContext:
    def Event(self):
        return object()

    def Queue(self):
        return _SpawnQueue()

    def Process(self, **kwargs):
        del kwargs
        return _ProcessStartFailure()


def _real_mps_group() -> StageGroup:
    return StageGroup(
        "pipeline",
        [
            StageWorkerProcessSpec(
                process_name="pipeline",
                stage_specs=[
                    StageLaunchConfig(
                        stage_name="preprocessing",
                        factory=f"{__name__}.noop_factory",
                        placement_gpu_id=0,
                        gpu_id=0,
                        recv_endpoint="ipc://preprocessing",
                    )
                ],
            )
        ],
    )


def _private_mps_runtime(root):
    client = FakeControlClient()
    runtime = MpsPipelineRuntime(client, ["pipeline"], root)
    runtime.poll_interval = 0
    runtime.verify_timeout = 0.02
    runtime.stop_timeout = 0.02
    return runtime, client


@pytest.mark.asyncio
async def test_mps_hooks_follow_resolved_spawn_lifecycle(short_base, monkeypatch):
    events: list[str] = []
    group = _FakeGroup(events)
    fake_mps = _FakeMps(events)
    coordinator = _patch_runner(monkeypatch, events, group, fake_mps)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))

    await runner.start()

    assert events.index("MPS start") < events.index("spawn")
    assert events.index("spawn") < events.index("ready")
    assert events.index("ready") < events.index("MPS verify")
    assert group.spawn_env == {
        "pipeline": {
            "CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps-pipe",
            "CUDA_VISIBLE_DEVICES": FAKE_GPU_UUID,
            "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
        }
    }
    assert coordinator.registered == {"preprocessing": "ipc://preprocessing"}

    await runner.stop()

    assert events.index("graceful shutdown") < events.index("process shutdown")
    assert events.index("process shutdown") < events.index("MPS close")
    assert group.before_signal is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_mps_startup_error_cleans_children_before_close(
    short_base, monkeypatch, error_type
):
    events: list[str] = []
    group = _FakeGroup(
        events,
        ready_error=error_type("ready failed"),
        direct_process=True,
    )
    fake_mps = _FakeMps(events)
    _patch_runner(monkeypatch, events, group, fake_mps)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))

    with pytest.raises(error_type, match="ready failed"):
        await runner.start()

    assert events.index("stage terminate") < events.index("MPS close")
    assert not fake_mps.has_resources


@pytest.mark.asyncio
async def test_startup_cancellation_remains_primary_when_mps_close_is_dirty(
    short_base,
    monkeypatch,
):
    events: list[str] = []
    group = _FakeGroup(
        events,
        ready_error=asyncio.CancelledError(),
        direct_process=True,
    )
    dirty = MpsDirtyStateError("dirty state persisted")
    fake_mps = _FakeMps(events, close_error=dirty)
    _patch_runner(monkeypatch, events, group, fake_mps)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await runner.start()

    assert exc_info.value.__cause__ is dirty
    assert events.index("stage terminate") < events.index("MPS close")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context,match",
    [
        (_PreStartFailureContext, "synchronization resource exhausted"),
        (_ProcessStartFailureContext, "Process.start failed"),
    ],
)
async def test_failed_worker_spawn_closes_private_daemon(
    short_base, monkeypatch, context, match
):
    events = []
    group = _real_mps_group()
    runtime, client = _private_mps_runtime(short_base)
    _patch_runner(monkeypatch, events, group, runtime)
    monkeypatch.setattr(mp_runner.multiprocessing, "get_context", lambda _: context())
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base, mps="on"))
    with pytest.raises(OSError, match=match):
        await runner.start()
    assert not runtime.has_resources
    assert not client.alive_pids
    assert not list(short_base.glob("run-*"))


@pytest.mark.asyncio
async def test_server_fault_reaches_runner_watchdog_and_cleans_private_daemon(
    short_base, monkeypatch
):
    events = []
    group = _FakeGroup(events)
    runtime, client = _private_mps_runtime(short_base)
    _patch_runner(monkeypatch, events, group, runtime)

    async def wait_ready(timeout):
        client.set_clients(runtime.pipe_dir, {8000: [101]})
        client.client_tokens[101] = group.spawn_env["pipeline"][MPS_CLIENT_TOKEN_ENV]

    async def shutdown(before_signal=None):
        client.set_clients(runtime.pipe_dir, {})

    monkeypatch.setattr(group, "wait_ready", wait_ready)
    monkeypatch.setattr(group, "shutdown", shutdown)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base, mps="on"))
    await runner.start()
    pipe_dir = runtime.pipe_dir
    client.server_statuses[(str(pipe_dir), 8000)] = "FAULT"
    try:
        with pytest.raises(RuntimeError, match="server 8000 is not ACTIVE: 'FAULT'"):
            await asyncio.wait_for(runner.wait_failed(), timeout=2)
    finally:
        await runner.stop()
    assert [call for call in client.calls if call[0] == "status"] == [
        ("status", pipe_dir, 8000)
    ]
    assert not runtime.has_resources
    assert not client.alive_pids


@pytest.mark.asyncio
async def test_mps_watchdog_fails_serving_before_launcher_cleanup(
    short_base,
    monkeypatch,
    caplog,
):
    events: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()
    group = _FakeGroup(events, shutdown_gate=(entered, release))
    dirty = MpsDirtyStateError("dirty state persisted")
    probe_gate = asyncio.Event()
    fake_mps = _FakeMps(
        events,
        close_error=dirty,
        probe_result="server 8000 is not ACTIVE",
        probe_gate=probe_gate,
    )
    _patch_runner(monkeypatch, events, group, fake_mps)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    await runner.start()

    probe_gate.set()
    with pytest.raises(RuntimeError, match="MPS health check failed") as exc_info:
        await runner.wait_failed()

    stop_task = asyncio.create_task(runner.stop())
    await entered.wait()
    assert not stop_task.done()
    release.set()
    await stop_task

    assert "MPS teardown incomplete: dirty state persisted" in caplog.text
    assert "server 8000 is not ACTIVE" in str(exc_info.value)
    assert exc_info.value.__cause__ is dirty


@pytest.mark.asyncio
async def test_mps_close_cancellation_finishes_runner_cleanup_before_propagating(
    short_base,
    monkeypatch,
):
    events: list[str] = []
    group = _FakeGroup(events)
    fake_mps = _FakeMps(events, close_error=asyncio.CancelledError())
    _patch_runner(monkeypatch, events, group, fake_mps)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    await runner.start()

    with pytest.raises(asyncio.CancelledError):
        await runner.stop()

    assert events.index("MPS close") < events.index("coordinator stop")


@pytest.mark.asyncio
async def test_mps_off_preserves_spawn_and_failure_order(
    short_base,
    monkeypatch,
):
    events: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()
    group = _FakeGroup(events, shutdown_gate=(entered, release))
    _patch_runner(monkeypatch, events, group, fake_mps=None)

    original_sleep = asyncio.sleep

    async def checkpoint(_delay: float) -> None:
        await original_sleep(0)

    monkeypatch.setattr(mp_runner.asyncio, "sleep", checkpoint)

    def unexpected_mps(*args, **kwargs):
        del args, kwargs
        raise AssertionError("mps=off must not create an MPS runtime")

    monkeypatch.setattr(mp_runner, "create_for_pipeline", unexpected_mps)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base, mps="off"))
    await runner.start()
    assert group.spawn_env is None

    group.dead = True
    waiter = asyncio.create_task(runner.wait_failed())
    await entered.wait()
    assert waiter.done()
    assert group.before_signal is None
    with pytest.raises(RuntimeError, match="Dead stage process"):
        await waiter
    release.set()
    assert "MPS close" not in events


@pytest.mark.asyncio
async def test_cancelled_stop_still_closes_private_daemon(
    short_base,
    monkeypatch,
):
    events: list[str] = []
    group = _FakeGroup(events)
    fake = _FakeMps(events)
    coordinator = _patch_runner(monkeypatch, events, group, fake)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocking_shutdown_stages() -> None:
        entered.set()
        await release.wait()

    monkeypatch.setattr(coordinator, "shutdown_stages", blocking_shutdown_stages)

    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    await runner.start()

    stopping = asyncio.create_task(runner.stop())
    await entered.wait()
    stopping.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await stopping

    assert not fake.has_resources
    assert "MPS close" in events


class _StuckProcess:
    """A stage process that never exits on its own."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.pid = 4321
        self.name = "stuck"
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout=None) -> None:
        del timeout

    def terminate(self) -> None:
        self.events.append("SIGTERM")
        self._alive = False

    def kill(self) -> None:
        self.events.append("SIGKILL")
        self._alive = False


@pytest.mark.asyncio
async def test_stuck_process_is_retired_from_mps_before_any_signal():
    events: list[str] = []
    group = _real_mps_group()
    group._processes = [_StuckProcess(events)]

    async def before_signal(process_name: str) -> None:
        events.append(f"retire {process_name}")

    await group.shutdown(join_timeout=0, before_signal=before_signal)

    assert "SIGTERM" in events
    assert events.index("retire pipeline") < events.index("SIGTERM")


@pytest.mark.asyncio
async def test_cancelling_startup_cleanup_still_closes_mps(short_base, monkeypatch):
    events = []
    group = _FakeGroup(events, ready_error=RuntimeError("ready failed"))
    fake = _FakeMps(events)
    coordinator = _patch_runner(monkeypatch, events, group, fake)
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked_stop():
        entered.set()
        await release.wait()

    monkeypatch.setattr(coordinator, "stop", blocked_stop)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    starting = asyncio.create_task(runner.start())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        starting.cancel()
        await asyncio.sleep(0)
        assert not starting.done()
    finally:
        release.set()
    with pytest.raises(RuntimeError, match="ready failed"):
        await starting
    assert "MPS close" in events
    assert not fake.has_resources


@pytest.mark.asyncio
@pytest.mark.parametrize("startup_failure", [False, True])
@pytest.mark.parametrize("failure", ["snapshot", "terminate_client"])
async def test_retirement_failure_still_closes_worker_pipeline_and_mps(
    short_base, monkeypatch, startup_failure, failure
):
    from sglang_omni.mps.control import MpsControlError

    events = []
    group = _real_mps_group()
    worker = _StuckProcess(events)
    runtime, client = _private_mps_runtime(short_base)
    _patch_runner(monkeypatch, events, group, runtime)

    def fail(*args):
        events.append("retirement failed")
        raise MpsControlError("control unavailable")

    def spawn(ctx, process_env_overrides=None):
        group._processes.append(worker)
        client.set_clients(runtime.pipe_dir, {7000: [101]})
        client.client_tokens[101] = process_env_overrides["pipeline"][
            MPS_CLIENT_TOKEN_ENV
        ]

    async def wait_ready(timeout):
        if startup_failure:
            monkeypatch.setattr(client, failure, fail)
            raise RuntimeError("ready failed")

    quit_daemon = client.quit_daemon

    def quit(pipe_dir):
        events.append("MPS quit")
        quit_daemon(pipe_dir)

    monkeypatch.setattr(group, "spawn", spawn)
    monkeypatch.setattr(group, "wait_ready", wait_ready)
    monkeypatch.setattr(client, "quit_daemon", quit)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    if startup_failure:
        with pytest.raises(RuntimeError, match="ready failed"):
            await runner.start()
    else:
        await runner.start()
        monkeypatch.setattr(client, failure, fail)
        await runner.stop()

    assert not worker.is_alive()
    assert events.index("retirement failed") < events.index("SIGTERM")
    assert events.index("SIGTERM") < events.index("MPS quit")
    assert "coordinator stop" in events
    assert not runtime.has_resources
    assert not client.alive_pids
    assert not list(short_base.glob("run-*"))


@pytest.mark.asyncio
async def test_startup_cleanup_reports_worker_that_survives_kill(
    short_base, monkeypatch
):
    events = []
    group = _FakeGroup(
        events, ready_error=RuntimeError("ready failed"), direct_process=True
    )
    worker = group.processes[0]
    monkeypatch.setattr(worker, "terminate", lambda: None)
    monkeypatch.setattr(worker, "kill", lambda: None)
    fake = _FakeMps(events)
    _patch_runner(monkeypatch, events, group, fake)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(short_base))
    try:
        with pytest.raises(RuntimeError, match="ready failed") as exc:
            await runner.start()
        assert "survived startup cleanup" in str(exc.value.__cause__)
        assert fake.has_resources
        assert "MPS close" not in events
        assert runner._groups == [group]
    finally:
        worker._alive = False
        await runner._cleanup_on_failure()
