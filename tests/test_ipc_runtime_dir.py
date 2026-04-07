# SPDX-License-Identifier: Apache-2.0
"""Lightweight IPC runtime directory lifecycle tests.

Covers per-run IPC namespace isolation (unique directory allocation)
and runtime directory cleanup on pipeline shutdown. Does not require
GPU or full model startup.

Reference: https://github.com/sgl-project/sglang-omni/issues/252

Author:
Ratish P https://github.com/Ratish1
Chenyang Zhao https://github.com/zhaochenyang20
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from sglang_omni.config import PipelineRunner, build_pipeline_runner, compile_pipeline
from sglang_omni.config.compiler import compile_pipeline_core, create_ipc_runtime_dir
from sglang_omni.config.schema import (
    EndpointsConfig,
    ExecutorConfig,
    PipelineConfig,
    StageConfig,
)

_NOOP_FACTORY = "sglang_omni.pipeline.mp_runner._noop_executor_factory"
_NOOP_GET_NEXT = "sglang_omni.pipeline.mp_runner._noop_get_next"


def _make_config(base_path: str) -> PipelineConfig:
    return PipelineConfig(
        model_path="dummy",
        entry_stage="preprocessing",
        stages=[
            StageConfig(
                name="preprocessing",
                executor=ExecutorConfig(factory=_NOOP_FACTORY, args={}),
                get_next=_NOOP_GET_NEXT,
            )
        ],
        endpoints=EndpointsConfig(
            scheme="ipc",
            base_path=base_path,
        ),
    )


class TestIpcRuntimeDir(unittest.TestCase):
    def test_ipc_runtime_dir_close_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            runtime_dir = create_ipc_runtime_dir(config)
            self.assertIsNotNone(runtime_dir)
            runtime_path = runtime_dir.path

            runtime_dir.close()
            runtime_dir.close()

            self.assertFalse(runtime_path.exists())

    def test_default_ipc_runtime_dirs_are_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)

            runtime_a = create_ipc_runtime_dir(config)
            runtime_b = create_ipc_runtime_dir(config)

            self.assertIsNotNone(runtime_a)
            self.assertIsNotNone(runtime_b)
            self.assertNotEqual(runtime_a.path, runtime_b.path)

            _coordinator_a, stages_a, _ = compile_pipeline_core(
                config,
                ipc_runtime_dir=runtime_a,
            )
            _coordinator_b, stages_b, _ = compile_pipeline_core(
                config,
                ipc_runtime_dir=runtime_b,
            )

            self.assertNotEqual(
                stages_a[0].control_plane.recv_endpoint,
                stages_b[0].control_plane.recv_endpoint,
            )

            runtime_a.close()
            runtime_b.close()

    def test_create_ipc_runtime_dir_returns_none_for_tcp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            config.endpoints.scheme = "tcp"

            self.assertIsNone(create_ipc_runtime_dir(config))

    def test_compile_pipeline_rejects_unmanaged_ipc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)

            with self.assertRaisesRegex(ValueError, "does not manage IPC runtime-dir"):
                compile_pipeline(config)

    def test_compile_core_preserves_caller_owned_ipc_dir_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            runtime_dir = create_ipc_runtime_dir(config)
            self.assertIsNotNone(runtime_dir)
            runtime_path = runtime_dir.path

            with patch(
                "sglang_omni.config.compiler._compile_stage",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    compile_pipeline_core(config, ipc_runtime_dir=runtime_dir)

            self.assertTrue(runtime_path.exists())
            runtime_dir.close()
            self.assertFalse(runtime_path.exists())


class TestPipelineRunnerIpcCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_build_pipeline_runner_cleans_runtime_dir_on_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)

            runner = build_pipeline_runner(config)

            self.assertEqual(runner.coordinator.entry_stage, "preprocessing")
            self.assertEqual(len(runner.stages), 1)
            runtime_dirs = [path for path in Path(tmp_dir).iterdir() if path.is_dir()]
            self.assertEqual(len(runtime_dirs), 1)
            runtime_path = runtime_dirs[0]
            self.assertTrue(runtime_path.exists())

            await runner.start()
            await runner.stop()

            self.assertFalse(runtime_path.exists())
            await runner.stop()

    async def test_build_pipeline_runner_cleans_runtime_dir_on_start_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            runner = build_pipeline_runner(config)

            runtime_dirs = [path for path in Path(tmp_dir).iterdir() if path.is_dir()]
            self.assertEqual(len(runtime_dirs), 1)
            runtime_path = runtime_dirs[0]

            runner.coordinator.start = AsyncMock(side_effect=RuntimeError("boom"))

            with self.assertRaisesRegex(RuntimeError, "boom"):
                await runner.start()

            self.assertFalse(runtime_path.exists())

    async def test_caller_managed_compile_path_cleans_runtime_dir_on_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)

            runtime_dir = create_ipc_runtime_dir(config)
            self.assertIsNotNone(runtime_dir)

            coordinator, stages, runtime_dir = compile_pipeline_core(
                config,
                ipc_runtime_dir=runtime_dir,
            )
            runner = PipelineRunner(coordinator, stages, ipc_runtime_dir=runtime_dir)

            runtime_dirs = [path for path in Path(tmp_dir).iterdir() if path.is_dir()]
            self.assertEqual(len(runtime_dirs), 1)
            runtime_path = runtime_dirs[0]
            self.assertTrue(runtime_path.exists())

            await runner.start()
            await runner.stop()

            self.assertFalse(runtime_path.exists())


class TestMultiProcessPipelineRunnerIpcCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_mp_runner_cleans_runtime_dir_on_start_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

            runner = MultiProcessPipelineRunner(config)

            with patch(
                "sglang_omni.pipeline.mp_runner.Coordinator.start",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    await runner.start()

            runtime_dirs = [path for path in Path(tmp_dir).iterdir() if path.is_dir()]
            self.assertEqual(runtime_dirs, [])

    async def test_mp_runner_cleans_runtime_dir_on_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

            runner = MultiProcessPipelineRunner(config)
            await runner.start(timeout=30.0)

            runtime_dirs = [path for path in Path(tmp_dir).iterdir() if path.is_dir()]
            self.assertEqual(len(runtime_dirs), 1)
            runtime_path = runtime_dirs[0]
            self.assertTrue(runtime_path.exists())

            await runner.stop()

            self.assertFalse(runtime_path.exists())


if __name__ == "__main__":
    unittest.main()
