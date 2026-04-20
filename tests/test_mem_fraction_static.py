# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import typer

from sglang_omni.cli.serve import serve
from sglang_omni.config.schema import (
    ExecutorConfig,
    MemFractionOverrideStages,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig

try:
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
    from sglang_omni.models.qwen3_omni.pipeline.stages import (
        create_sglang_thinker_executor_from_config,
    )

    _qwen3_available = True
except ImportError:
    _qwen3_available = False

_NOOP_FACTORY = "sglang_omni.pipeline.mp_runner._noop_executor_factory"
_NOOP_GET_NEXT = "sglang_omni.pipeline.mp_runner._noop_get_next"


def _make_stage(name: str, *, args: dict | None = None) -> StageConfig:
    return StageConfig(
        name=name,
        executor=ExecutorConfig(factory=_NOOP_FACTORY, args=args or {}),
        get_next=_NOOP_GET_NEXT,
    )


def _make_pipeline(
    *,
    thinker: str | None = "thinker",
    talker: str | None = "talker",
    thinker_args: dict | None = None,
    talker_args: dict | None = None,
) -> PipelineConfig:
    stages = [_make_stage("preprocessing")]
    if thinker is not None:
        stages.append(_make_stage(thinker, args=thinker_args))
    if talker is not None and talker != thinker:
        stages.append(_make_stage(talker, args=talker_args))

    return PipelineConfig(
        model_path="dummy",
        entry_stage="preprocessing",
        stages=stages,
        mem_fraction_override_stages=MemFractionOverrideStages(
            thinker=thinker,
            talker=talker,
        ),
    )


class TestMemFractionStaticOverrides(unittest.TestCase):
    def test_global_override_applies_to_all_targets_and_preserves_other_overrides(
        self,
    ) -> None:
        config = _make_pipeline(
            thinker_args={"server_args_overrides": {"cpu_offload_gb": 80}},
            talker_args={"server_args_overrides": {"enable_dp_attention": True}},
        )

        config.apply_mem_fraction_static_overrides(mem_fraction_static=0.88)

        thinker_overrides = config.stages[1].executor.args["server_args_overrides"]
        talker_overrides = config.stages[2].executor.args["server_args_overrides"]

        self.assertEqual(thinker_overrides["cpu_offload_gb"], 80)
        self.assertEqual(thinker_overrides["mem_fraction_static"], 0.88)
        self.assertTrue(talker_overrides["enable_dp_attention"])
        self.assertEqual(talker_overrides["mem_fraction_static"], 0.88)

    def test_stage_specific_overrides_take_precedence_over_global(self) -> None:
        config = _make_pipeline()

        config.apply_mem_fraction_static_overrides(
            mem_fraction_static=0.88,
            thinker_mem_fraction_static=0.83,
            talker_mem_fraction_static=0.91,
        )

        thinker_overrides = config.stages[1].executor.args["server_args_overrides"]
        talker_overrides = config.stages[2].executor.args["server_args_overrides"]

        self.assertEqual(thinker_overrides["mem_fraction_static"], 0.83)
        self.assertEqual(talker_overrides["mem_fraction_static"], 0.91)

    def test_invalid_stage_specific_override_does_not_partially_mutate_config(
        self,
    ) -> None:
        config = _make_pipeline(talker=None)

        with self.assertRaisesRegex(
            ValueError,
            "--talker-mem-fraction-static requires a pipeline with a 'talker'",
        ):
            config.apply_mem_fraction_static_overrides(
                mem_fraction_static=0.88,
                talker_mem_fraction_static=0.91,
            )

        thinker_args = config.stages[1].executor.args
        self.assertNotIn("server_args_overrides", thinker_args)

    def test_global_override_requires_declared_targets(self) -> None:
        config = _make_pipeline(thinker=None, talker=None)

        with self.assertRaisesRegex(
            ValueError,
            "--mem-fraction-static requires a pipeline with a supported",
        ):
            config.apply_mem_fraction_static_overrides(mem_fraction_static=0.88)

    def test_global_override_applies_to_single_declared_target(self) -> None:
        config = _make_pipeline(
            thinker="thinker",
            talker=None,
            thinker_args={"server_args_overrides": {"cpu_offload_gb": 80}},
        )

        config.apply_mem_fraction_static_overrides(mem_fraction_static=0.88)

        thinker_overrides = config.stages[1].executor.args["server_args_overrides"]

        self.assertEqual(thinker_overrides["cpu_offload_gb"], 80)
        self.assertEqual(thinker_overrides["mem_fraction_static"], 0.88)

    def test_override_targets_must_reference_distinct_stages(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "mem_fraction_override_stages thinker and talker must reference different stages",
        ):
            _make_pipeline(thinker="shared", talker="shared")

    def test_unknown_override_stage_is_rejected_at_construction(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "mem_fraction_override_stages references unknown stages",
        ):
            PipelineConfig(
                model_path="dummy",
                entry_stage="preprocessing",
                stages=[_make_stage("preprocessing"), _make_stage("thinker")],
                mem_fraction_override_stages=MemFractionOverrideStages(
                    thinker="ghost_stage"
                ),
            )

    def test_apply_server_args_overrides_rejects_unknown_stage(self) -> None:
        config = _make_pipeline()

        with self.assertRaisesRegex(ValueError, "Unknown stage 'nope'"):
            config.apply_server_args_overrides(stage_name="nope", overrides={})

    def test_invalid_value_range_is_rejected(self) -> None:
        config = _make_pipeline()

        for value in (-0.1, 0.0, 1.0, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "must be in the open interval",
                ):
                    config.apply_mem_fraction_static_overrides(
                        mem_fraction_static=value
                    )

    def test_model_copy_then_apply_does_not_mutate_original(self) -> None:
        config = _make_pipeline()
        copied_config = config.model_copy(update={"model_path": "other"}, deep=True)

        copied_config.apply_mem_fraction_static_overrides(mem_fraction_static=0.88)

        self.assertNotIn("server_args_overrides", config.stages[1].executor.args)


class _FakeConfigManager:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def parse_extra_args(self, args: list[str]) -> dict[str, str]:
        del args
        return {}

    def merge_config(self, extra_args: dict[str, str]) -> PipelineConfig:
        del extra_args
        return self.config


class TestServeMemFractionStatic(unittest.TestCase):
    @patch("sglang_omni.cli.serve.launch_server")
    @patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
    def test_serve_rejects_unsupported_talker_flag_before_launch(
        self,
        from_model_path,
        launch_server_mock,
    ) -> None:
        from_model_path.return_value = _FakeConfigManager(
            MingOmniPipelineConfig(model_path="dummy")
        )

        with self.assertRaises(typer.BadParameter):
            serve(
                ctx=SimpleNamespace(args=[]),
                model_path="dummy",
                config=None,
                text_only=False,
                host="0.0.0.0",
                port=8000,
                model_name=None,
                mem_fraction_static=None,
                thinker_mem_fraction_static=None,
                talker_mem_fraction_static=0.88,
                log_level="info",
            )

        launch_server_mock.assert_not_called()

    @patch("sglang_omni.cli.serve.launch_server")
    @patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
    def test_serve_rejects_invalid_mem_fraction_value_before_launch(
        self,
        from_model_path,
        launch_server_mock,
    ) -> None:
        from_model_path.return_value = _FakeConfigManager(_make_pipeline())

        with self.assertRaisesRegex(
            typer.BadParameter,
            "must be in the open interval",
        ):
            serve(
                ctx=SimpleNamespace(args=[]),
                model_path="dummy",
                config=None,
                text_only=False,
                host="0.0.0.0",
                port=8000,
                model_name=None,
                mem_fraction_static=1.5,
                thinker_mem_fraction_static=None,
                talker_mem_fraction_static=None,
                log_level="info",
            )

        launch_server_mock.assert_not_called()

    @patch("sglang_omni.cli.serve.launch_server")
    @patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
    def test_serve_applies_mem_fraction_to_copied_config_only(
        self,
        from_model_path,
        launch_server_mock,
    ) -> None:
        original_config = _make_pipeline()
        from_model_path.return_value = _FakeConfigManager(original_config)

        serve(
            ctx=SimpleNamespace(args=[]),
            model_path="other-model",
            config=None,
            text_only=False,
            host="0.0.0.0",
            port=8000,
            model_name=None,
            mem_fraction_static=0.88,
            thinker_mem_fraction_static=None,
            talker_mem_fraction_static=None,
            log_level="info",
        )

        passed_config = launch_server_mock.call_args.args[0]
        self.assertNotIn(
            "server_args_overrides", original_config.stages[1].executor.args
        )
        self.assertEqual(
            passed_config.stages[1].executor.args["server_args_overrides"][
                "mem_fraction_static"
            ],
            0.88,
        )

    @unittest.skipUnless(_qwen3_available, "qwen3_omni config not importable")
    @patch(
        "sglang_omni.models.qwen3_omni.pipeline.stages.create_sglang_thinker_executor"
    )
    @patch("sglang_omni.cli.serve.launch_server")
    @patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
    def test_serve_mem_fraction_reaches_final_server_args(
        self,
        from_model_path,
        launch_server_mock,
        create_thinker_executor_mock,
    ) -> None:
        from_model_path.return_value = _FakeConfigManager(
            Qwen3OmniPipelineConfig(model_path="dummy")
        )

        serve(
            ctx=SimpleNamespace(args=[]),
            model_path="dummy",
            config=None,
            text_only=False,
            host="0.0.0.0",
            port=8000,
            model_name=None,
            mem_fraction_static=0.88,
            thinker_mem_fraction_static=None,
            talker_mem_fraction_static=None,
            log_level="info",
        )

        passed_config = launch_server_mock.call_args.args[0]
        thinker_stage = next(
            stage for stage in passed_config.stages if stage.name == "thinker"
        )

        create_sglang_thinker_executor_from_config(
            model_path="dummy",
            **thinker_stage.executor.args,
        )

        server_args = create_thinker_executor_mock.call_args.kwargs["server_args"]
        self.assertEqual(server_args.mem_fraction_static, 0.88)
