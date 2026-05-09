# SPDX-License-Identifier: Apache-2.0
"""Omni V1 IPC runtime directory lifecycle tests."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI

pytest.importorskip("torch")

from sglang_omni_v1.config.compiler import (
    CompiledPipeline,
    IpcRuntimeDir,
    compile_pipeline,
    compile_pipeline_core,
    create_ipc_runtime_dir,
    prepare_pipeline_runtime,
)
from sglang_omni_v1.config.schema import EndpointsConfig, PipelineConfig, StageConfig


def noop_factory():
    return None


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
    def __init__(self):
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def run_completion_loop(self) -> None:
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.stopped = True


def _make_config(base_path: str, *, scheme: str = "ipc") -> PipelineConfig:
    return PipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        entry_stage="preprocessing",
        stages=[
            StageConfig(
                name="preprocessing",
                factory="tests.test_v1_ipc_runtime_dir.noop_factory",
                terminal=True,
            )
        ],
        endpoints=EndpointsConfig(
            scheme=scheme,
            base_path=base_path,
        ),
    )


class TestV1IpcRuntimeDir(unittest.TestCase):
    def test_ipc_runtime_dir_close_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            runtime_dir = create_ipc_runtime_dir(config)
            self.assertIsNotNone(runtime_dir)
            runtime_path = runtime_dir.path

            runtime_dir.close()
            runtime_dir.close()

            self.assertFalse(runtime_path.exists())

    def test_create_ipc_runtime_dir_returns_none_for_tcp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir, scheme="tcp")

            self.assertIsNone(create_ipc_runtime_dir(config))

    def test_ipc_runtime_dirs_are_unique_for_same_model_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)

            runtime_a = create_ipc_runtime_dir(config)
            runtime_b = create_ipc_runtime_dir(config)
            self.assertIsNotNone(runtime_a)
            self.assertIsNotNone(runtime_b)

            try:
                self.assertNotEqual(runtime_a.path, runtime_b.path)

                compiled_a = compile_pipeline_core(
                    config,
                    ipc_runtime_dir=runtime_a,
                )
                compiled_b = compile_pipeline_core(
                    config,
                    ipc_runtime_dir=runtime_b,
                )

                self.assertNotEqual(
                    compiled_a.stages[0].control_plane.recv_endpoint,
                    compiled_b.stages[0].control_plane.recv_endpoint,
                )
            finally:
                runtime_a.close()
                runtime_b.close()

    def test_compile_pipeline_rejects_unmanaged_ipc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)

            with self.assertRaisesRegex(ValueError, "does not manage IPC"):
                compile_pipeline(config)

    def test_compile_core_cleans_owned_ipc_dir_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)

            with patch(
                "sglang_omni_v1.config.compiler._compile_stage",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    compile_pipeline_core(config)

            self.assertEqual(list(Path(tmp_dir).iterdir()), [])

    def test_caller_owned_ipc_dir_is_not_removed_on_compile_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            runtime_dir = create_ipc_runtime_dir(config)
            self.assertIsNotNone(runtime_dir)
            runtime_path = runtime_dir.path

            with patch(
                "sglang_omni_v1.config.compiler._compile_stage",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    compile_pipeline_core(config, ipc_runtime_dir=runtime_dir)

            self.assertTrue(runtime_path.exists())
            runtime_dir.close()
            self.assertFalse(runtime_path.exists())

    def test_compile_core_returns_owned_runtime_dir_for_successful_ipc_compile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)

            compiled = compile_pipeline_core(config)
            runtime_dir = compiled.runtime_dir
            self.assertIsNotNone(runtime_dir)
            runtime_path = runtime_dir.path

            try:
                self.assertTrue(runtime_path.exists())
                self.assertIn(
                    str(runtime_path),
                    compiled.stages[0].control_plane.recv_endpoint,
                )
            finally:
                runtime_dir.close()

            self.assertFalse(runtime_path.exists())


class TestV1MultiProcessRunnerIpcCleanup(unittest.IsolatedAsyncioTestCase):
    def test_mp_runner_uses_unique_ipc_endpoints_for_same_model_name(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            from sglang_omni_v1.pipeline.mp_runner import _build_stage_groups

            prep_a = prepare_pipeline_runtime(config)
            prep_b = prepare_pipeline_runtime(config)
            self.assertIsNotNone(prep_a.runtime_dir)
            self.assertIsNotNone(prep_b.runtime_dir)

            try:
                groups_a = _build_stage_groups(
                    config,
                    stages_cfg=prep_a.stages_cfg,
                    name_map=prep_a.name_map,
                    endpoints=prep_a.endpoints,
                )
                groups_b = _build_stage_groups(
                    config,
                    stages_cfg=prep_b.stages_cfg,
                    name_map=prep_b.name_map,
                    endpoints=prep_b.endpoints,
                )

                self.assertNotEqual(
                    prep_a.endpoints["completion"],
                    prep_b.endpoints["completion"],
                )
                self.assertNotEqual(
                    groups_a[0].leader_endpoint,
                    groups_b[0].leader_endpoint,
                )
            finally:
                prep_a.runtime_dir.close()
                prep_b.runtime_dir.close()

            self.assertEqual(list(Path(tmp_dir).iterdir()), [])

    async def test_mp_runner_cleans_runtime_dir_on_start_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            from sglang_omni_v1.pipeline.mp_runner import MultiProcessPipelineRunner

            runner = MultiProcessPipelineRunner(config)

            with patch(
                "sglang_omni_v1.pipeline.mp_runner.Coordinator.start",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    await runner.start()

            self.assertEqual(list(Path(tmp_dir).iterdir()), [])

    async def test_mp_runner_starts_two_same_model_instances_and_cleans_on_stop(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            from sglang_omni_v1.pipeline.mp_runner import MultiProcessPipelineRunner

            runner_a = MultiProcessPipelineRunner(config)
            runner_b = MultiProcessPipelineRunner(config)

            try:
                await runner_a.start(timeout=30.0)
                await runner_b.start(timeout=30.0)

                runtime_dirs = [
                    path for path in Path(tmp_dir).iterdir() if path.is_dir()
                ]
                self.assertEqual(len(runtime_dirs), 2)
                self.assertNotEqual(
                    runner_a.coordinator.control_plane.completion_endpoint,
                    runner_b.coordinator.control_plane.completion_endpoint,
                )
                self.assertNotEqual(
                    runner_a._groups[0].leader_endpoint,
                    runner_b._groups[0].leader_endpoint,
                )
            finally:
                await runner_b.stop()
                await runner_a.stop()

            self.assertEqual(list(Path(tmp_dir).iterdir()), [])


class TestV1LauncherIpcCleanup(unittest.IsolatedAsyncioTestCase):
    async def _run_single_process_launcher_with_mocked_server(
        self,
        *,
        config: PipelineConfig,
        runtime_dir: IpcRuntimeDir,
        serve_mock: AsyncMock,
    ) -> tuple[_FakeCoordinator, FastAPI]:
        stage = _FakeStage(f"ipc://{runtime_dir.path}/stage_preprocessing.sock")
        coordinator = _FakeCoordinator()
        app = FastAPI()

        from sglang_omni_v1.serve.launcher import _run_server

        with (
            patch(
                "sglang_omni_v1.serve.launcher._find_available_port",
                return_value=8000,
            ),
            patch(
                "sglang_omni_v1.serve.launcher.compile_pipeline_core",
                return_value=CompiledPipeline(
                    coordinator=coordinator,
                    stages=[stage],
                    runtime_dir=runtime_dir,
                ),
            ) as compile_pipeline_core,
            patch(
                "sglang_omni_v1.serve.launcher.create_app",
                return_value=app,
            ) as create_app,
            patch(
                "sglang_omni_v1.serve.launcher.uvicorn.Server.serve",
                new=serve_mock,
            ),
        ):
            await _run_server(config, port=8000)

        compile_pipeline_core.assert_called_once()
        call_args = compile_pipeline_core.call_args
        called_config = (
            call_args.args[0] if call_args.args else call_args.kwargs.get("config")
        )
        self.assertIs(called_config, config)
        create_app.assert_called_once()

        return coordinator, app

    async def test_single_process_launcher_cleans_runtime_dir_on_server_exit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            runtime_dir = create_ipc_runtime_dir(config)
            self.assertIsNotNone(runtime_dir)
            runtime_path = runtime_dir.path
            server_serve = AsyncMock(return_value=None)

            coordinator, app = (
                await self._run_single_process_launcher_with_mocked_server(
                    config=config,
                    runtime_dir=runtime_dir,
                    serve_mock=server_serve,
                )
            )

            self.assertTrue(coordinator.started)
            self.assertTrue(coordinator.stopped)
            server_serve.assert_awaited_once()
            mounted_paths = {route.path for route in app.routes}
            self.assertIn("/start_profile", mounted_paths)
            self.assertIn("/stop_profile", mounted_paths)
            self.assertFalse(runtime_path.exists())

    async def test_single_process_launcher_cleans_runtime_dir_on_server_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            runtime_dir = create_ipc_runtime_dir(config)
            self.assertIsNotNone(runtime_dir)
            runtime_path = runtime_dir.path
            server_serve = AsyncMock(side_effect=RuntimeError("server failed"))

            with self.assertRaisesRegex(RuntimeError, "server failed"):
                await self._run_single_process_launcher_with_mocked_server(
                    config=config,
                    runtime_dir=runtime_dir,
                    serve_mock=server_serve,
                )

            server_serve.assert_awaited_once()
            self.assertFalse(runtime_path.exists())

    async def test_single_process_launcher_preserves_pre_start_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            runtime_dir = create_ipc_runtime_dir(config)
            self.assertIsNotNone(runtime_dir)
            runtime_path = runtime_dir.path
            coordinator = _FakeCoordinator()
            coordinator.stop = AsyncMock(side_effect=RuntimeError("stop failed"))
            stage = _FakeStage(f"ipc://{runtime_dir.path}/stage_preprocessing.sock")

            from sglang_omni_v1.serve.launcher import _run_server

            with (
                patch(
                    "sglang_omni_v1.serve.launcher._find_available_port",
                    return_value=8000,
                ),
                patch(
                    "sglang_omni_v1.serve.launcher.compile_pipeline_core",
                    return_value=CompiledPipeline(
                        coordinator=coordinator,
                        stages=[stage],
                        runtime_dir=runtime_dir,
                    ),
                ),
                patch(
                    "sglang_omni_v1.serve.launcher._collect_stage_control_endpoints",
                    side_effect=RuntimeError("bad endpoints"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "bad endpoints"):
                    await _run_server(config, port=8000)

            self.assertFalse(coordinator.started)
            coordinator.stop.assert_awaited_once()
            self.assertFalse(runtime_path.exists())
