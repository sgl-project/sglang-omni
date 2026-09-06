# SPDX-License-Identifier: Apache-2.0
"""Runner-to-weight-share integration: spawn order, env merge, teardown order."""

from __future__ import annotations

import asyncio
import os
from typing import ClassVar

import pytest

from sglang_omni.config import (
    EndpointsConfig,
    EngineArgs,
    PipelineConfig,
    ProcessConfig,
    StageConfig,
)
from sglang_omni.config.schema import EngineStageConfig
from sglang_omni.pipeline import mp_runner
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, StageWorkerProcessSpec
from sglang_omni.pipeline.weight_share import WeightShareError
from sglang_omni.utils.ipc_weights import ENV_WEIGHT_SHARE, ENV_WEIGHT_SHARE_COMPAT

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="weight sharing plans only on POSIX hosts",
)


def noop_factory():  # pragma: no cover - never constructed here
    raise AssertionError("factory must not run")


class _SharingPipelineConfig(PipelineConfig):
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "engine": EngineStageConfig,
    }


def _config(tmp_path, *, weight_share: str = "on") -> PipelineConfig:
    engine = EngineArgs()
    engine.max_total_tokens = 30000
    return _SharingPipelineConfig(
        model_path="model",
        entry_stage="engine",
        stages=[
            EngineStageConfig(
                name="engine",
                process="gen",
                factory_path=f"{__name__}.noop_factory",
                gpu=0,
                gpu_memory_fraction=0.45,
                terminal=True,
                engine=engine,
            )
        ],
        processes={"gen": ProcessConfig(num_replicas=2, replica_devices=[0, 0])},
        weight_share=weight_share,
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
    )


class _FakeCoordinator:
    def __init__(self, events: list[str], *args, **kwargs) -> None:
        del args, kwargs
        self.events = events
        self.registered: dict[str, str] = {}
        self.shutdown_calls: list[list[str] | None] = []

    async def start(self) -> None:
        self.events.append("coordinator start")

    async def run_completion_loop(self) -> None:
        await asyncio.Event().wait()

    def register_stage(self, name: str, endpoint: str) -> None:
        self.registered[name] = endpoint

    async def shutdown_stages(self, stage_names=None) -> None:
        self.shutdown_calls.append(None if stage_names is None else list(stage_names))
        self.events.append(f"graceful shutdown {stage_names}")

    async def fail_pending_requests(self, error: BaseException) -> None:
        del error

    async def stop(self) -> None:
        self.events.append("coordinator stop")


class _FakeProcess:
    def __init__(self) -> None:
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self._alive = False

    def kill(self) -> None:
        self._alive = False

    def join(self, timeout=None) -> None:
        del timeout


class _FakeGroup:
    process_count = 1

    def __init__(
        self,
        events: list[str],
        process_name: str,
        *,
        gpu_id: int = 0,
        ready_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.group_name = process_name
        self.ready_error = ready_error
        self.processes: list[_FakeProcess] = []
        self.spawn_env = None
        self.ready_timeout: float | None = None
        self.dead = False
        self.stage_control_endpoints = {
            f"engine@r{process_name[-1]}": f"ipc://{process_name}"
        }
        self.process_specs = [
            StageWorkerProcessSpec(
                process_name=process_name,
                stage_specs=[
                    StageLaunchConfig(
                        stage_name=f"engine@r{process_name[-1]}",
                        factory=f"{__name__}.noop_factory",
                        placement_gpu_id=gpu_id,
                        gpu_id=gpu_id,
                        recv_endpoint=f"ipc://{process_name}",
                    )
                ],
            )
        ]

    def spawn(self, ctx, process_env_overrides=None) -> None:
        del ctx
        self.spawn_env = process_env_overrides
        self.processes.append(_FakeProcess())
        self.events.append(f"spawn {self.group_name}")

    async def wait_ready(self, timeout: float) -> None:
        self.ready_timeout = timeout
        self.events.append(f"ready {self.group_name}")
        if self.ready_error is not None:
            raise self.ready_error

    def any_dead(self) -> bool:
        return self.dead

    def dead_summary(self) -> str:
        return f"{self.group_name} exited"

    def process_start_attempts(self) -> set[str]:
        return {self.group_name} if self.processes else set()

    async def shutdown(self, before_signal=None) -> None:
        del before_signal
        self.events.append(f"shutdown {self.group_name}")

    def close_control_channels(self) -> None:
        self.events.append(f"channels closed {self.group_name}")


def _patch(monkeypatch, events, groups) -> _FakeCoordinator:
    coordinator = _FakeCoordinator(events)
    monkeypatch.setattr(mp_runner, "Coordinator", lambda *a, **k: coordinator)
    monkeypatch.setattr(mp_runner, "_build_stage_groups", lambda *a, **k: groups)
    return coordinator


@pytest.mark.asyncio
async def test_no_follower_spawns_before_every_leader_is_ready(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []
    leader = _FakeGroup(events, "gen@r0")
    follower = _FakeGroup(events, "gen@r1")
    _patch(monkeypatch, events, [leader, follower])
    runner = MultiProcessPipelineRunner(_config(tmp_path))

    await runner.start(timeout=5.0)
    await runner.stop()

    assert events.index("ready gen@r0") < events.index("spawn gen@r1")


@pytest.mark.asyncio
async def test_each_wave_gets_the_whole_startup_budget(tmp_path, monkeypatch) -> None:
    """A slow leader load must not starve the follower attach behind it."""
    events: list[str] = []
    leader = _FakeGroup(events, "gen@r0")
    follower = _FakeGroup(events, "gen@r1")
    _patch(monkeypatch, events, [leader, follower])
    runner = MultiProcessPipelineRunner(_config(tmp_path))

    await runner.start(timeout=37.0)
    await runner.stop()

    assert leader.ready_timeout == 37.0
    assert follower.ready_timeout == 37.0


@pytest.mark.asyncio
async def test_a_dead_leader_stops_the_run_before_any_follower_spawns(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []
    leader = _FakeGroup(events, "gen@r0")
    leader.dead = True
    follower = _FakeGroup(events, "gen@r1")
    _patch(monkeypatch, events, [leader, follower])
    runner = MultiProcessPipelineRunner(_config(tmp_path))

    with pytest.raises(RuntimeError, match="died during startup"):
        await runner.start(timeout=5.0)

    assert "spawn gen@r1" not in events
    assert follower.spawn_env is None


@pytest.mark.asyncio
async def test_a_replica_alone_on_its_gpu_still_gets_spawned(
    tmp_path, monkeypatch
) -> None:
    """replica_devices=[0, 0, 1]: the unshared replica is routable, so it runs."""
    events: list[str] = []
    groups = [
        _FakeGroup(events, f"gen@r{index}", gpu_id=0 if index < 2 else 1)
        for index in range(3)
    ]
    _patch(monkeypatch, events, groups)
    config = _config(tmp_path)
    config.processes["gen"] = ProcessConfig(num_replicas=3, replica_devices=[0, 0, 1])
    runner = MultiProcessPipelineRunner(config)

    await runner.start(timeout=5.0)
    await runner.stop()

    assert [event for event in events if event.startswith("spawn")] == [
        "spawn gen@r0",
        "spawn gen@r2",
        "spawn gen@r1",
    ]


@pytest.mark.asyncio
async def test_cancelling_startup_without_mps_still_reaps_the_children(
    tmp_path, monkeypatch
) -> None:
    """Cancellation is a BaseException; cleanup must not hang off MPS."""
    events: list[str] = []
    leader = _FakeGroup(events, "gen@r0", ready_error=asyncio.CancelledError())
    follower = _FakeGroup(events, "gen@r1")
    _patch(monkeypatch, events, [leader, follower])
    runner = MultiProcessPipelineRunner(_config(tmp_path))

    with pytest.raises(asyncio.CancelledError):
        await runner.start(timeout=5.0)

    assert "channels closed gen@r0" in events
    assert "coordinator stop" in events


@pytest.mark.asyncio
async def test_roles_and_the_compat_flag_reach_the_spawn_environment(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []
    leader = _FakeGroup(events, "gen@r0")
    follower = _FakeGroup(events, "gen@r1")
    _patch(monkeypatch, events, [leader, follower])
    runner = MultiProcessPipelineRunner(_config(tmp_path))

    await runner.start(timeout=5.0)
    try:
        env = leader.spawn_env
        assert env is not None
        assert env["gen@r0"][ENV_WEIGHT_SHARE].startswith("leader:")
        assert env["gen@r1"][ENV_WEIGHT_SHARE].startswith("follower:")
        assert env["gen@r0"][ENV_WEIGHT_SHARE_COMPAT] == "1"
        assert env["gen@r1"][ENV_WEIGHT_SHARE_COMPAT] == "1"
        assert follower.spawn_env is env
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_mps_environment_survives_the_weight_share_merge(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []
    leader = _FakeGroup(events, "gen@r0")
    follower = _FakeGroup(events, "gen@r1")
    _patch(monkeypatch, events, [leader, follower])

    class _FakeMps:
        has_leases = False

        async def start(self) -> None:
            events.append("MPS acquire")

        def env_for_process(self, process_name: str) -> dict[str, str]:
            return {"CUDA_MPS_PIPE_DIRECTORY": f"/tmp/pipe/{process_name}"}

        async def verify(self) -> None:
            events.append("MPS verify")

        async def probe_failures(self) -> dict[str, str]:
            return {}

        async def retire_process_clients(self, process_name: str) -> set:
            return set()

        async def close(self, *, process_start_attempts=None) -> None:
            del process_start_attempts
            events.append("MPS close")

    monkeypatch.setattr(
        mp_runner, "create_for_pipeline", lambda mode, specs: _FakeMps()
    )
    config = _config(tmp_path)
    config.mps = "on"
    runner = MultiProcessPipelineRunner(config)

    await runner.start(timeout=5.0)
    try:
        env = leader.spawn_env
        assert env["gen@r0"]["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/pipe/gen@r0"
        assert env["gen@r0"][ENV_WEIGHT_SHARE].startswith("leader:")
        assert env["gen@r0"][ENV_WEIGHT_SHARE_COMPAT] == "1"
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_roles_are_planned_before_the_coordinator_binds(
    tmp_path, monkeypatch
) -> None:
    """An unshareable pipeline must fail before parent resources exist."""
    events: list[str] = []
    group = _FakeGroup(events, "gen@r0")
    coordinator = _patch(monkeypatch, events, [group])
    config = _config(tmp_path)
    config.processes["gen"] = ProcessConfig(num_replicas=2, replica_devices=[0, 1])
    runner = MultiProcessPipelineRunner(config)

    with pytest.raises(WeightShareError):
        await runner.start(timeout=5.0)

    assert "coordinator start" not in events
    assert coordinator.shutdown_calls == []


@pytest.mark.asyncio
async def test_followers_shut_down_before_their_leader(tmp_path, monkeypatch) -> None:
    events: list[str] = []
    leader = _FakeGroup(events, "gen@r0")
    follower = _FakeGroup(events, "gen@r1")
    coordinator = _patch(monkeypatch, events, [leader, follower])
    runner = MultiProcessPipelineRunner(_config(tmp_path))

    await runner.start(timeout=5.0)
    await runner.stop()

    assert events.index("shutdown gen@r1") < events.index("shutdown gen@r0")
    assert coordinator.shutdown_calls == [["engine@r1"], ["engine@r0"]]


@pytest.mark.asyncio
async def test_sharing_off_keeps_one_spawn_wave_and_one_broadcast(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []
    first = _FakeGroup(events, "gen@r0")
    second = _FakeGroup(events, "gen@r1")
    coordinator = _patch(monkeypatch, events, [first, second])
    runner = MultiProcessPipelineRunner(_config(tmp_path, weight_share="off"))

    await runner.start(timeout=5.0)
    await runner.stop()

    assert events.index("spawn gen@r1") < events.index("ready gen@r0")
    assert first.spawn_env is None
    assert coordinator.shutdown_calls == [None]
