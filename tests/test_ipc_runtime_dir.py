import tempfile
import unittest
from pathlib import Path

from sglang_omni.config import build_pipeline_runner
from sglang_omni.config.compiler import _allocate_endpoints, _prepare_ipc_runtime_dir
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
    def test_default_ipc_runtime_dirs_are_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)

            runtime_a = _prepare_ipc_runtime_dir(config)
            runtime_b = _prepare_ipc_runtime_dir(config)

            self.assertIsNotNone(runtime_a)
            self.assertIsNotNone(runtime_b)
            self.assertNotEqual(runtime_a.path, runtime_b.path)

            endpoints_a = _allocate_endpoints(
                config,
                stages=config.stages,
                ipc_base_dir=runtime_a.path,
            )
            endpoints_b = _allocate_endpoints(
                config,
                stages=config.stages,
                ipc_base_dir=runtime_b.path,
            )

            self.assertNotEqual(
                endpoints_a["stage_preprocessing"],
                endpoints_b["stage_preprocessing"],
            )

            runtime_a.close()
            runtime_b.close()


class TestPipelineRunnerIpcCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_build_pipeline_runner_cleans_runtime_dir_on_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = PipelineConfig(
                model_path="dummy",
                entry_stage="preprocessing",
                stages=[
                    StageConfig(
                        name="preprocessing",
                        executor=ExecutorConfig(
                            factory=_NOOP_FACTORY,
                            args={},
                        ),
                        get_next=_NOOP_GET_NEXT,
                    )
                ],
                endpoints=EndpointsConfig(
                    scheme="ipc",
                    base_path=tmp_dir,
                ),
            )

            coordinator, stages, runner = build_pipeline_runner(config)

            self.assertEqual(coordinator.entry_stage, "preprocessing")
            self.assertEqual(len(stages), 1)
            runtime_dirs = [path for path in Path(tmp_dir).iterdir() if path.is_dir()]
            self.assertEqual(len(runtime_dirs), 1)
            runtime_path = runtime_dirs[0]
            self.assertTrue(runtime_path.exists())

            await runner.start()
            await runner.stop()

            self.assertFalse(runtime_path.exists())


if __name__ == "__main__":
    unittest.main()
