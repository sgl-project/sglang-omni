# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import sglang_omni.pipeline.mp_runner as mp_runner
import sglang_omni.pipeline.runtime_config as runtime_config
import sglang_omni.pipeline.stage.runtime as stage_runtime
from sglang_omni.config.schema import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.profiler.event_recorder import get_recorder
from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext, FakeRelay


def noop_factory():
    return None


def failing_factory():
    raise RuntimeError("factory boom")


class _FakeControlPlane:
    def __init__(self, recv_endpoint: str):
        self.recv_endpoint = recv_endpoint


class _FakeStage:
    name = "preprocessing"

    def __init__(self, recv_endpoint: str):
        self.control_plane = _FakeControlPlane(recv_endpoint)

    async def run(self) -> None:
        await asyncio.Event().wait()


class _FakeCoordinator:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def run_completion_loop(self) -> None:
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.stopped = True


def _make_config(base_path: Path) -> PipelineConfig:
    return PipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        entry_stage="preprocessing",
        stages=[
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory=f"{__name__}.noop_factory",
                terminal=True,
            )
        ],
        endpoints=EndpointsConfig(base_path=str(base_path)),
    )


@pytest.fixture(autouse=True)
def _fake_stage_relay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stage_runtime,
        "create_relay",
        lambda relay_type, **kwargs: FakeRelay(device=kwargs.get("device", "cpu")),
    )


def test_ipc_runtime_dir_creation_and_close_contracts(tmp_path: Path) -> None:
    """Preserves IPC runtime directory creation, uniqueness, and idempotent cleanup."""
    ipc_config = _make_config(tmp_path)

    runtime_a = runtime_config.create_ipc_runtime_dir(ipc_config)
    runtime_b = runtime_config.create_ipc_runtime_dir(ipc_config)
    assert runtime_a is not None
    assert runtime_b is not None
    assert runtime_a.path != runtime_b.path

    runtime_path = runtime_a.path
    runtime_a.close()
    runtime_a.close()
    runtime_b.close()
    assert not runtime_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_prepare_pipeline_runtime_owns_or_preserves_ipc_runtime_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves owned IPC cleanup and caller-owned IPC directory preservation."""
    config = _make_config(tmp_path)

    def fail_allocate_endpoints(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime_config, "allocate_endpoints", fail_allocate_endpoints)

    with pytest.raises(RuntimeError, match="boom"):
        runtime_config.prepare_pipeline_runtime(config)
    assert list(tmp_path.iterdir()) == []

    caller_owned = runtime_config.create_ipc_runtime_dir(config)
    assert caller_owned is not None
    caller_path = caller_owned.path
    with pytest.raises(RuntimeError, match="boom"):
        runtime_config.prepare_pipeline_runtime(config, ipc_runtime_dir=caller_owned)
    assert caller_path.exists()
    caller_owned.close()
    assert list(tmp_path.iterdir()) == []


def test_prepare_pipeline_runtime_returns_managed_ipc_runtime_dir(
    tmp_path: Path,
) -> None:
    """Preserves managed IPC runtime directory ownership in runtime prep."""
    prep = runtime_config.prepare_pipeline_runtime(_make_config(tmp_path))
    runtime_dir = prep.runtime_dir
    assert runtime_dir is not None
    try:
        assert runtime_dir.path.exists()
        assert str(runtime_dir.path) in prep.endpoints["stage_preprocessing"]
    finally:
        runtime_dir.close()

    assert list(tmp_path.iterdir()) == []


def test_ipc_stage_groups_use_unique_endpoints_for_same_model_name(
    tmp_path: Path,
) -> None:
    """Preserves unique IPC endpoints across same-model pipeline instances."""
    config = _make_config(tmp_path)
    prep_a = runtime_config.prepare_pipeline_runtime(config)
    prep_b = runtime_config.prepare_pipeline_runtime(config)
    assert prep_a.runtime_dir is not None
    assert prep_b.runtime_dir is not None

    try:
        groups_a = mp_runner._build_stage_groups(
            config,
            FakeMpContext(),
            stages_cfg=prep_a.stages_cfg,
            name_map=prep_a.name_map,
            endpoints=prep_a.endpoints,
            placement_plan=prep_a.placement_plan,
            process_plan=prep_a.process_plan,
        )
        groups_b = mp_runner._build_stage_groups(
            config,
            FakeMpContext(),
            stages_cfg=prep_b.stages_cfg,
            name_map=prep_b.name_map,
            endpoints=prep_b.endpoints,
            placement_plan=prep_b.placement_plan,
            process_plan=prep_b.process_plan,
        )

        assert prep_a.endpoints["completion"] != prep_b.endpoints["completion"]
        assert groups_a[0].leader_endpoint != groups_b[0].leader_endpoint
    finally:
        prep_a.runtime_dir.close()
        prep_b.runtime_dir.close()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_cleans_runtime_dir_on_start_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves IPC runtime directory cleanup when runner startup fails."""

    class FailingCoordinator:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        async def start(self) -> None:
            raise RuntimeError("boom")

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(mp_runner, "Coordinator", FailingCoordinator)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))

    with pytest.raises(RuntimeError, match="boom"):
        await runner.start()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_cleans_spawned_groups_when_later_spawn_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves spawned process cleanup if a later stage group fails to spawn."""

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.join_count = 0
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self.terminated = True
            self._alive = False

        def kill(self) -> None:
            self.killed = True
            self._alive = False

        def join(self, timeout=None) -> None:
            del timeout
            self.join_count += 1

    class FakeGroup:
        def __init__(self, stage_name: str, *, fail_spawn: bool = False) -> None:
            self.stage_name = stage_name
            self.fail_spawn = fail_spawn
            self.process = FakeProcess() if not fail_spawn else None
            self.channels_closed = False

        @property
        def processes(self) -> list[FakeProcess]:
            return [self.process] if self.process is not None else []

        def spawn(self, ctx) -> None:
            del ctx
            if self.fail_spawn:
                raise RuntimeError(f"spawn failed for {self.stage_name}")

        async def wait_ready(self, timeout: float) -> None:
            del timeout

        def close_control_channels(self) -> None:
            self.channels_closed = True

    first_group = FakeGroup("preprocessing")
    second_group = FakeGroup("thinker", fail_spawn=True)
    monkeypatch.setattr(mp_runner, "Coordinator", _FakeCoordinator)
    monkeypatch.setattr(
        mp_runner,
        "_build_stage_groups",
        lambda *a, **k: [first_group, second_group],
    )

    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))
    with pytest.raises(RuntimeError, match="spawn failed"):
        await runner.start()

    assert first_group.process.terminated
    assert first_group.process.join_count >= 1
    assert first_group.channels_closed
    assert second_group.channels_closed
    assert list(tmp_path.iterdir()) == []


def test_startup_batches_serialize_same_gpu_groups() -> None:
    events: list[tuple[str, str]] = []

    class FakeGroup:
        def __init__(self, name: str, gpu_ids: set[int]) -> None:
            self.name = name
            self.gpu_ids = gpu_ids

        def spawn(self, ctx) -> None:
            del ctx
            events.append(("spawn", self.name))

        async def wait_ready(self, timeout: float) -> None:
            assert timeout > 0
            events.append(("ready", self.name))

    asyncio.run(
        mp_runner._spawn_and_wait_startup_batches(
            [
                FakeGroup("image_encoder", {0}),
                FakeGroup("thinker", {0}),
                FakeGroup("other_gpu", {1}),
                FakeGroup("decode", set()),
            ],
            ctx=object(),
            timeout=10,
        )
    )

    assert events == [
        ("spawn", "image_encoder"),
        ("spawn", "other_gpu"),
        ("spawn", "decode"),
        ("ready", "image_encoder"),
        ("ready", "other_gpu"),
        ("ready", "decode"),
        ("spawn", "thinker"),
        ("ready", "thinker"),
    ]


@pytest.mark.asyncio
async def test_mp_runner_startup_failure_includes_child_factory_traceback(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        name="x",
        entry_stage="preprocessing",
        stages=[
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory=f"{__name__}.failing_factory",
                terminal=True,
            )
        ],
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
    )
    runner = mp_runner.MultiProcessPipelineRunner(config)

    with pytest.raises(RuntimeError, match="factory boom"):
        await runner.start(timeout=10.0)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_stop_cleans_runtime_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves IPC runtime directory cleanup when the runner stops."""

    class FakeCoordinator:
        def __init__(
            self,
            completion_endpoint: str,
            abort_endpoint: str,
            entry_stage: str,
            terminal_stages: list[str] | None = None,
            terminal_stages_resolver=None,
        ) -> None:
            del abort_endpoint, entry_stage, terminal_stages, terminal_stages_resolver
            self.control_plane = SimpleNamespace(
                completion_endpoint=completion_endpoint
            )

        async def start(self) -> None:
            return None

        async def run_completion_loop(self) -> None:
            await asyncio.Event().wait()

        def register_stage(self, name: str, endpoint: str) -> None:
            del name, endpoint

        async def shutdown_stages(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakeGroup:
        stage_name = "preprocessing"
        leader_endpoint = "ipc://stage.sock"
        tp_size = 1
        process_count = 1
        processes: list[object] = []
        stage_control_endpoints = {"preprocessing": "ipc://stage.sock"}

        def __init__(self) -> None:
            self.shutdown_called = False

        def spawn(self, ctx) -> None:
            del ctx

        async def wait_ready(self, timeout: float) -> None:
            del timeout

        def any_dead(self) -> bool:
            return False

        def dead_summary(self) -> str:
            return "(none)"

        async def shutdown(self) -> None:
            self.shutdown_called = True

    group = FakeGroup()
    monkeypatch.setattr(mp_runner, "Coordinator", FakeCoordinator)
    monkeypatch.setattr(mp_runner, "_build_stage_groups", lambda *a, **k: [group])

    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))
    await runner.start()
    assert len([path for path in tmp_path.iterdir() if path.is_dir()]) == 1

    await runner.stop()

    assert group.shutdown_called
    assert list(tmp_path.iterdir()) == []


async def _run_launcher_with_fake_runner(
    *,
    config: PipelineConfig,
    serve_mock: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, FastAPI, SimpleNamespace]:
    app = FastAPI()
    profiler_calls = SimpleNamespace(starts=[], stops=[])

    from sglang_omni.serve import launcher

    runner_ref = None

    class FakeRunner:
        def __init__(self, pipeline_config: PipelineConfig) -> None:
            del pipeline_config
            nonlocal runner_ref
            self.coordinator = _FakeCoordinator()
            self.stage_control_endpoints = {
                "preprocessing": "ipc://stage_preprocessing.sock"
            }
            self.started = False
            self.stopped = False
            # launcher._run_server reads .prep.placement_plan / .process_plan
            # after start() to log the resolved topology. Provide empty stubs
            # that satisfy _placement_log_summary's attribute access.
            self.prep = SimpleNamespace(
                placement_plan=SimpleNamespace(gpus={}),
                process_plan=SimpleNamespace(
                    groups=(),
                    tp_stage_to_processes={},
                ),
            )
            runner_ref = self

        async def start(self, timeout: float) -> None:
            del timeout
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

        async def wait_failed(self) -> None:
            await asyncio.Future()

    class FakeProfilerControl:
        def __init__(self, stage_control_endpoints: dict[str, str]) -> None:
            del stage_control_endpoints

        async def broadcast_start(self, **kwargs) -> None:
            profiler_calls.starts.append(kwargs)

        async def broadcast_stop(self, **kwargs) -> None:
            profiler_calls.stops.append(kwargs)

    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", FakeRunner)
    monkeypatch.setattr(launcher, "ProfilerControlClient", FakeProfilerControl)
    monkeypatch.setattr(launcher, "create_app", lambda *a, **k: app)
    monkeypatch.setattr(launcher.uvicorn.Server, "serve", serve_mock)

    await launcher._run_server(config, port=8000)
    assert runner_ref is not None
    return runner_ref, app, profiler_calls


@pytest.mark.asyncio
async def test_launcher_uses_runner_and_mounts_profiler_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    server_serve = AsyncMock(return_value=None)

    runner, app, profiler_calls = await _run_launcher_with_fake_runner(
        config=config,
        serve_mock=server_serve,
        monkeypatch=monkeypatch,
    )

    assert runner.started
    assert runner.stopped
    server_serve.assert_awaited_once()
    try:
        with TestClient(app) as client:
            start_resp = client.post(
                "/start_profile",
                json={
                    "enable_torch": False,
                    "event_dir": str(tmp_path / "events"),
                },
            )
            stop_resp = client.post("/stop_profile", json={})
        assert start_resp.status_code == 200
        assert stop_resp.status_code == 200
        assert profiler_calls.starts
        assert profiler_calls.starts[0]["enable_torch"] is False
        assert profiler_calls.starts[0]["event_dir"] == str(tmp_path / "events")
        assert profiler_calls.stops == [{"run_id": None}]
    finally:
        rec = get_recorder()
        if rec.is_active():
            rec.stop()


def test_start_profile_request_only_mode_does_not_require_trace_template(
    tmp_path: Path,
) -> None:
    from sglang_omni.serve import launcher

    class FakeProfilerControl:
        def __init__(self) -> None:
            self.starts: list[dict] = []

        async def broadcast_start(self, **kwargs) -> None:
            self.starts.append(kwargs)

    app = FastAPI()
    ctl = FakeProfilerControl()
    launcher._mount_profiler_routes(app, ctl, profiler_dir=None)
    event_dir = str(tmp_path / "events")

    try:
        with TestClient(app) as client:
            resp = client.post(
                "/start_profile",
                json={"enable_torch": False, "event_dir": event_dir},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["enable_torch"] is False
        assert body["trace_path_template"] == ""
        assert body["event_dir"] == event_dir
        assert ctl.starts
        assert ctl.starts[0]["enable_torch"] is False
        assert ctl.starts[0]["trace_path_template"] == ""
        assert ctl.starts[0]["event_dir"] == event_dir
    finally:
        rec = get_recorder()
        if rec.is_active():
            rec.stop()


def test_start_profile_torch_mode_still_requires_trace_template() -> None:
    from sglang_omni.serve import launcher

    class FakeProfilerControl:
        async def broadcast_start(self, **kwargs) -> None:
            raise AssertionError("start_profile should fail before broadcasting")

    app = FastAPI()
    launcher._mount_profiler_routes(app, FakeProfilerControl(), profiler_dir=None)

    with TestClient(app) as client:
        resp = client.post("/start_profile", json={"enable_torch": True})
    assert resp.status_code == 400
    assert "trace_path_template is required" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_launcher_stops_runner_when_server_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    server_serve = AsyncMock(side_effect=RuntimeError("server failed"))

    with pytest.raises(RuntimeError, match="server failed"):
        await _run_launcher_with_fake_runner(
            config=config,
            serve_mock=server_serve,
            monkeypatch=monkeypatch,
        )

    server_serve.assert_awaited_once()


@pytest.mark.asyncio
async def test_launcher_preserves_runner_start_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)

    from sglang_omni.serve import launcher

    class FakeRunner:
        def __init__(self, pipeline_config: PipelineConfig) -> None:
            del pipeline_config

        async def start(self, timeout: float) -> None:
            del timeout
            raise RuntimeError("start failed")

        async def stop(self) -> None:
            raise AssertionError("launcher should not stop a runner that failed start")

    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", FakeRunner)

    with pytest.raises(RuntimeError, match="start failed"):
        await launcher._run_server(config, port=8000)
