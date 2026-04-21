# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import typer

from sglang_omni.cli.serve import serve
from sglang_omni.config.schema import ExecutorConfig, PipelineConfig, StageConfig
from sglang_omni.engines.ar.sglang_backend.server_args_builder import (
    build_sglang_server_args,
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
MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def _make_stage(name: str, *, args: dict | None = None) -> StageConfig:
    return StageConfig(
        name=name,
        executor=ExecutorConfig(factory=_NOOP_FACTORY, args=args or {}),
        get_next=_NOOP_GET_NEXT,
    )


class _MemFractionPipelineConfig(PipelineConfig):
    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker", "talker": "talker"}


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

    return _MemFractionPipelineConfig(
        model_path="dummy",
        entry_stage="preprocessing",
        stages=stages,
    )


class TestMemFractionStaticOverrides(unittest.TestCase):
    def test_apply_server_args_overrides_preserves_other_overrides(self) -> None:
        config = _make_pipeline(
            thinker_args={"server_args_overrides": {"cpu_offload_gb": 80}},
        )

        config.apply_server_args_overrides(
            stage_name="thinker",
            overrides={"mem_fraction_static": 0.88},
        )

        thinker_overrides = config.stages[1].executor.args["server_args_overrides"]

        self.assertEqual(thinker_overrides["cpu_offload_gb"], 80)
        self.assertEqual(thinker_overrides["mem_fraction_static"], 0.88)

    def test_mem_fraction_role_mapping_is_class_level_and_not_dumped(self) -> None:
        config = _make_pipeline()

        self.assertEqual(
            type(config).mem_fraction_role_to_stage(),
            {"thinker": "thinker", "talker": "talker"},
        )
        self.assertNotIn("mem_fraction_override_stages", config.model_dump())

    def test_apply_server_args_overrides_rejects_unknown_stage(self) -> None:
        config = _make_pipeline()

        with self.assertRaisesRegex(ValueError, "Unknown stage 'nope'"):
            config.apply_server_args_overrides(stage_name="nope", overrides={})

    @patch("sglang_omni.engines.ar.sglang_backend.server_args_builder.ServerArgs")
    def test_auto_mem_fraction_static_reserve_subtracts_auto_value_only(
        self, server_args_mock
    ) -> None:
        server_args_mock.return_value = SimpleNamespace(mem_fraction_static=0.929)

        server_args = build_sglang_server_args(
            model_path="dummy",
            context_length=8192,
            auto_mem_fraction_static_reserve=0.05,
        )

        self.assertEqual(server_args.mem_fraction_static, 0.879)
        self.assertNotIn("mem_fraction_static", server_args_mock.call_args.kwargs)

    @patch("sglang_omni.engines.ar.sglang_backend.server_args_builder.ServerArgs")
    def test_auto_mem_fraction_static_reserve_preserves_user_pinned_value(
        self, server_args_mock
    ) -> None:
        server_args_mock.return_value = SimpleNamespace(mem_fraction_static=0.88)

        server_args = build_sglang_server_args(
            model_path="dummy",
            context_length=8192,
            mem_fraction_static=0.88,
            auto_mem_fraction_static_reserve=0.05,
        )

        self.assertEqual(server_args.mem_fraction_static, 0.88)
        self.assertEqual(server_args_mock.call_args.kwargs["mem_fraction_static"], 0.88)


class TestH20AutoMemFractionFloor(unittest.TestCase):
    def test_h20_auto_mem_fraction_static_has_expected_floor(self) -> None:
        total_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if not 90 <= total_memory_gib <= 100:
            pytest.skip(f"H20-only check, got GPU with {total_memory_gib:.1f} GiB")

        server_args = build_sglang_server_args(MODEL_PATH, 8192)
        self.assertGreaterEqual(server_args.mem_fraction_static, 0.85)


class _FakeConfigManager:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def parse_extra_args(self, args: list[str]) -> dict[str, str]:
        del args
        return {}

    def merge_config(self, extra_args: dict[str, str]) -> PipelineConfig:
        del extra_args
        return type(self.config)(**self.config.model_dump())


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
            r"must be > 0 and < 1",
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
