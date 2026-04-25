# SPDX-License-Identifier: Apache-2.0
"""Tests for mem_fraction_static CLI and ServerArgs override behavior.

Author:
    Ratish P https://github.com/Ratish1
    Chenyang Zhao https://github.com/zhaochenyang20
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import typer

from sglang_omni.cli.serve import serve
from sglang_omni.config.schema import ExecutorConfig, PipelineConfig, StageConfig
from sglang_omni.engines.ar.sglang_backend.server_args_builder import (
    apply_encoder_mem_reserve,
    build_sglang_server_args,
)
from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig

try:
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
    from sglang_omni.models.qwen3_omni.pipeline.stages import (
        create_sglang_thinker_executor_from_config,
        create_talker_ar_executor_from_config,
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
        """Setting mem_fraction_static keeps any pre-existing server_args_overrides keys intact."""
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
        """Role-to-stage mapping lives on the class, not the instance schema dump."""
        config = _make_pipeline()

        self.assertEqual(
            type(config).mem_fraction_role_to_stage(),
            {"thinker": "thinker", "talker": "talker"},
        )
        self.assertNotIn("mem_fraction_override_stages", config.model_dump())

    def test_apply_server_args_overrides_rejects_unknown_stage(self) -> None:
        """Unknown stage names raise ValueError at the primitive boundary."""
        config = _make_pipeline()

        with self.assertRaisesRegex(ValueError, "Unknown stage 'nope'"):
            config.apply_server_args_overrides(stage_name="nope", overrides={})

    def test_apply_encoder_mem_reserve_subtracts(self) -> None:
        """apply_encoder_mem_reserve subtracts the reserve from the server_args' auto mem_fraction_static in-place."""
        server_args = SimpleNamespace(mem_fraction_static=0.929)

        apply_encoder_mem_reserve(server_args, 0.05)

        self.assertEqual(server_args.mem_fraction_static, 0.879)

    def test_apply_encoder_mem_reserve_noop_when_zero(self) -> None:
        """Reserve = 0 is a no-op; non-thinker callers (talker_ar) rely on this."""
        server_args = SimpleNamespace(mem_fraction_static=0.929)

        apply_encoder_mem_reserve(server_args, 0.0)

        self.assertEqual(server_args.mem_fraction_static, 0.929)

    def test_build_sglang_server_args_no_longer_takes_reserve_kwarg(self) -> None:
        """build_sglang_server_args is now pure ServerArgs construction; reserve lives in apply_encoder_mem_reserve."""
        import inspect

        sig = inspect.signature(build_sglang_server_args)
        self.assertNotIn("auto_mem_fraction_static_reserve", sig.parameters)


@unittest.skipUnless(_qwen3_available, "qwen3_omni config not importable")
class TestEncoderMemReserveRouting(unittest.TestCase):
    """Round-trip + fallback + validation coverage for ``--encoder-mem-reserve``."""

    def test_encoder_mem_reserve_round_trips_to_thinker_stage_args(self) -> None:
        """apply_server_args_overrides routes encoder_mem_reserve to the thinker stage args, not server_args_overrides."""
        config = Qwen3OmniPipelineConfig(model_path="dummy")

        config.apply_server_args_overrides(
            stage_name="thinker",
            overrides={"encoder_mem_reserve": 0.20},
        )

        thinker_stage = next(s for s in config.stages if s.name == "thinker")
        self.assertEqual(thinker_stage.executor.args["encoder_mem_reserve"], 0.20)
        self.assertNotIn(
            "encoder_mem_reserve",
            thinker_stage.executor.args.get("server_args_overrides", {}),
        )

    @patch(
        "sglang_omni.models.qwen3_omni.pipeline.stages.create_sglang_thinker_executor"
    )
    def test_encoder_mem_reserve_reaches_thinker_factory(
        self, create_thinker_executor_mock
    ) -> None:
        """The routed encoder_mem_reserve reaches the factory and is applied to server_args.mem_fraction_static."""
        config = Qwen3OmniPipelineConfig(model_path="dummy")
        config.apply_server_args_overrides(
            stage_name="thinker",
            overrides={"encoder_mem_reserve": 0.20},
        )
        thinker_stage = next(s for s in config.stages if s.name == "thinker")

        with patch(
            "sglang_omni.engines.ar.sglang_backend.server_args_builder.ServerArgs"
        ) as server_args_mock:
            server_args_mock.return_value = SimpleNamespace(mem_fraction_static=0.929)
            create_sglang_thinker_executor_from_config(
                model_path="dummy",
                **thinker_stage.executor.args,
            )

        server_args = create_thinker_executor_mock.call_args.kwargs["server_args"]
        # Note (Chenyang):
        # auto=0.929 minus reserve=0.20 => 0.729 after rounding.
        self.assertEqual(server_args.mem_fraction_static, 0.729)

    @patch(
        "sglang_omni.models.qwen3_omni.pipeline.stages.create_sglang_thinker_executor"
    )
    def test_factory_uses_default_reserve_when_omitted(
        self, create_thinker_executor_mock
    ) -> None:
        """Omitting encoder_mem_reserve picks up the factory signature default (0.05)."""
        with patch(
            "sglang_omni.engines.ar.sglang_backend.server_args_builder.ServerArgs"
        ) as server_args_mock:
            server_args_mock.return_value = SimpleNamespace(mem_fraction_static=0.929)
            create_sglang_thinker_executor_from_config(
                model_path="dummy",
                thinker_max_seq_len=8192,
            )

        server_args = create_thinker_executor_mock.call_args.kwargs["server_args"]
        # Note (Chenyang):
        # auto=0.929 minus default=0.05 => 0.879 after rounding.
        self.assertEqual(server_args.mem_fraction_static, 0.879)

    def test_out_of_range_encoder_mem_reserve_rejected_via_apply_overrides(
        self,
    ) -> None:
        """Out-of-range encoder_mem_reserve raises at the config boundary, before launch."""
        config = Qwen3OmniPipelineConfig(model_path="dummy")

        with self.assertRaisesRegex(
            ValueError, r"encoder_mem_reserve must be in \[0, 1\)"
        ):
            config.apply_server_args_overrides(
                stage_name="thinker",
                overrides={"encoder_mem_reserve": 5.0},
            )

    def test_thinker_executor_args_atomic_on_partial_cast_failure(self) -> None:
        """A later cast failure must not leave an earlier valid key written."""
        config = Qwen3OmniPipelineConfig(model_path="dummy")
        original_args_by_stage = {
            stage.name: dict(stage.executor.args or {})
            for stage in config.stages
            if stage.executor is not None
        }

        with self.assertRaises(ValueError):
            config.apply_server_args_overrides(
                stage_name="thinker",
                overrides={
                    "thinker_max_seq_len": 16384,
                    "encoder_mem_reserve": 5.0,
                },
            )

        current_args_by_stage = {
            stage.name: dict(stage.executor.args or {})
            for stage in config.stages
            if stage.executor is not None
        }
        self.assertEqual(current_args_by_stage, original_args_by_stage)

    def test_encoder_mem_reserve_routes_on_speech_variant(self) -> None:
        """Speech variant shares _route_thinker_executor_args and pins that."""
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
        config.apply_server_args_overrides(
            stage_name="thinker",
            overrides={"encoder_mem_reserve": 0.20},
        )

        thinker_stage = next(s for s in config.stages if s.name == "thinker")
        self.assertEqual(thinker_stage.executor.args["encoder_mem_reserve"], 0.20)

    def test_thinker_seq_len_routes_to_speech_talker(self) -> None:
        """Speech Talker must accept the same long prompt that the thinker accepts."""
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
        config.apply_server_args_overrides(
            stage_name="thinker",
            overrides={"thinker_max_seq_len": 32768},
        )

        preprocessing_stage = next(
            s for s in config.stages if s.name == "preprocessing"
        )
        thinker_stage = next(s for s in config.stages if s.name == "thinker")
        talker_stage = next(s for s in config.stages if s.name == "talker_ar")

        self.assertEqual(preprocessing_stage.executor.args["max_seq_len"], 32768)
        self.assertEqual(thinker_stage.executor.args["thinker_max_seq_len"], 32768)
        self.assertEqual(talker_stage.executor.args["talker_max_seq_len"], 32768)


class TestEncoderMemReserveFloor(unittest.TestCase):
    """``apply_encoder_mem_reserve`` raises when reserve would drop mem_fraction_static below the safe floor."""

    def test_reserve_dropping_below_floor_raises(self) -> None:
        # Note (Chenyang):
        # Simulate a tiny auto value (small-memory GPU) + an aggressive
        # reserve. 0.15 - 0.10 = 0.05 < floor 0.1 -> raise.
        server_args = SimpleNamespace(mem_fraction_static=0.15)

        with self.assertRaisesRegex(ValueError, r"below the safe floor"):
            apply_encoder_mem_reserve(server_args, 0.10)


class TestCliMemFlagMutex(unittest.TestCase):
    """Reject passing both --mem-fraction-static and --encoder-mem-reserve."""

    def test_both_flags_raises(self) -> None:
        from examples.run_qwen3_omni_server import _check_mem_flag_mutex

        with self.assertRaisesRegex(ValueError, r"mutually exclusive"):
            _check_mem_flag_mutex(0.65, 0.20)

    def test_either_flag_alone_is_ok(self) -> None:
        from examples.run_qwen3_omni_server import _check_mem_flag_mutex

        _check_mem_flag_mutex(0.65, None)
        _check_mem_flag_mutex(None, 0.20)
        _check_mem_flag_mutex(None, None)


class TestH20AutoMemFractionFloor(unittest.TestCase):
    def test_h20_auto_mem_fraction_static_has_expected_floor(self) -> None:
        """On H20 hardware, SGLang's auto-sized mem_fraction_static must sit at or above 0.85."""
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
        """serve() raises BadParameter when --talker-mem-fraction-static targets a pipeline with no talker role."""
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
        """serve() rejects mem_fraction_static outside (0, 1) at the CLI boundary, before launching."""
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
        """Overrides land on the merged config copy; the original ConfigManager instance stays untouched."""
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
        self.assertEqual(original_config.model_path, "dummy")
        self.assertEqual(passed_config.model_path, "other-model")

    @unittest.skipUnless(_qwen3_available, "qwen3_omni config not importable")
    @patch(
        "sglang_omni.models.qwen3_omni.pipeline.stages.create_sglang_thinker_executor"
    )
    @patch("sglang_omni.cli.serve.launch_server")
    @patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
    def test_serve_thinker_override_round_trips_through_factory(
        self,
        from_model_path,
        launch_server_mock,
        create_thinker_executor_mock,
    ) -> None:
        """serve() wires mem_fraction_static all the way through to the thinker factory's final ServerArgs."""
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


@unittest.skipUnless(_qwen3_available, "qwen3_omni config not importable")
class TestServeMemFractionPrecedence(unittest.TestCase):
    """Verify per-stage mem_fraction_static flags override the global fallback."""

    def _run(
        self,
        global_mfs: float | None,
        thinker_mfs: float | None,
        talker_mfs: float | None,
    ) -> PipelineConfig:
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        with (
            patch("sglang_omni.cli.serve.launch_server") as launch_mock,
            patch("sglang_omni.cli.serve.ConfigManager.from_model_path") as from_path,
        ):
            from_path.return_value = _FakeConfigManager(
                Qwen3OmniSpeechPipelineConfig(model_path="dummy")
            )
            serve(
                ctx=SimpleNamespace(args=[]),
                model_path="dummy",
                config=None,
                text_only=False,
                host="0.0.0.0",
                port=8000,
                model_name=None,
                mem_fraction_static=global_mfs,
                thinker_mem_fraction_static=thinker_mfs,
                talker_mem_fraction_static=talker_mfs,
                log_level="info",
            )
            return launch_mock.call_args.args[0]

    def _mem_fraction_static(
        self, config: PipelineConfig, stage_name: str
    ) -> float | None:
        stage = next(stage for stage in config.stages if stage.name == stage_name)
        return stage.executor.args.get("server_args_overrides", {}).get(
            "mem_fraction_static"
        )

    def test_global_only_applies_to_both_roles(self) -> None:
        """Global --mem-fraction-static alone propagates to both thinker and talker_ar."""
        config = self._run(0.80, None, None)

        self.assertEqual(self._mem_fraction_static(config, "thinker"), 0.80)
        self.assertEqual(self._mem_fraction_static(config, "talker_ar"), 0.80)

    def test_thinker_specific_overrides_global_for_thinker_only(self) -> None:
        """--thinker-mem-fraction-static wins for thinker; global remains talker's fallback."""
        config = self._run(0.80, 0.70, None)

        self.assertEqual(self._mem_fraction_static(config, "thinker"), 0.70)
        self.assertEqual(self._mem_fraction_static(config, "talker_ar"), 0.80)

    def test_talker_specific_overrides_global_for_talker_only(self) -> None:
        """--talker-mem-fraction-static wins for talker; global remains thinker's fallback."""
        config = self._run(0.80, None, 0.65)

        self.assertEqual(self._mem_fraction_static(config, "thinker"), 0.80)
        self.assertEqual(self._mem_fraction_static(config, "talker_ar"), 0.65)

    def test_both_specific_plus_global_each_stage_gets_its_specific(self) -> None:
        """When global + both per-stage flags are set, each stage receives its per-stage value."""
        config = self._run(0.80, 0.70, 0.65)

        self.assertEqual(self._mem_fraction_static(config, "thinker"), 0.70)
        self.assertEqual(self._mem_fraction_static(config, "talker_ar"), 0.65)

    def test_thinker_specific_only_does_not_bleed_into_talker(self) -> None:
        """Thinker-specific flag must not silently propagate to talker when no global or talker flag is set."""
        config = self._run(None, 0.70, None)
        talker = next(stage for stage in config.stages if stage.name == "talker_ar")

        self.assertEqual(self._mem_fraction_static(config, "thinker"), 0.70)
        self.assertNotIn("server_args_overrides", talker.executor.args)


@unittest.skipUnless(_qwen3_available, "qwen3_omni config not importable")
class TestSpeechMemFractionFactoryWiring(unittest.TestCase):
    @patch("sglang_omni.models.qwen3_omni.pipeline.stages.create_talker_ar_executor")
    @patch(
        "sglang_omni.models.qwen3_omni.pipeline.stages.create_sglang_thinker_executor"
    )
    @patch("sglang_omni.cli.serve.launch_server")
    @patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
    def test_serve_speech_overrides_round_trip_through_factories(
        self,
        from_model_path,
        launch_server_mock,
        create_thinker_executor_mock,
        create_talker_executor_mock,
    ) -> None:
        """Per-stage mem_fraction_static flags round-trip to the thinker and talker_ar factory ServerArgs on the speech path."""
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        from_model_path.return_value = _FakeConfigManager(
            Qwen3OmniSpeechPipelineConfig(model_path="dummy")
        )

        serve(
            ctx=SimpleNamespace(args=[]),
            model_path="dummy",
            config=None,
            text_only=False,
            host="0.0.0.0",
            port=8000,
            model_name=None,
            mem_fraction_static=None,
            thinker_mem_fraction_static=0.70,
            talker_mem_fraction_static=0.65,
            log_level="info",
        )

        passed_config = launch_server_mock.call_args.args[0]
        thinker_stage = next(
            stage for stage in passed_config.stages if stage.name == "thinker"
        )
        talker_stage = next(
            stage for stage in passed_config.stages if stage.name == "talker_ar"
        )

        create_sglang_thinker_executor_from_config(
            model_path="dummy",
            **thinker_stage.executor.args,
        )
        create_talker_ar_executor_from_config(
            model_path="dummy",
            **talker_stage.executor.args,
        )

        thinker_server_args = create_thinker_executor_mock.call_args.kwargs[
            "server_args"
        ]
        talker_server_args = create_talker_executor_mock.call_args.kwargs["server_args"]
        self.assertEqual(thinker_server_args.mem_fraction_static, 0.70)
        self.assertEqual(talker_server_args.mem_fraction_static, 0.65)
