# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.cli.serve import apply_tensor_parallel_engine_overrides
from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.ming_omni.config import (
    MingOmniPipelineConfig,
    MingOmniSpeechPipelineConfig,
)
from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
from sglang_omni.models.registry import (
    PIPELINE_CONFIG_REGISTRY,
    import_pipeline_configs,
)


def _stage(config: PipelineConfig, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _server_args_overrides(config: PipelineConfig, name: str) -> dict[str, object]:
    engine = _stage(config, name).engine
    return engine.overrides() if engine is not None else {}


def test_ming_config_manager_resolves_top_level_hf_architecture(monkeypatch) -> None:
    calls = []

    def fake_from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            architectures=["BailingMM2NativeForConditionalGeneration"]
        )

    monkeypatch.setattr(
        "sglang_omni.config.manager.AutoConfig.from_pretrained",
        fake_from_pretrained,
    )

    config_manager = ConfigManager.from_model_path("inclusionAI/Ming-flash-omni-2.0")

    assert calls == [(("inclusionAI/Ming-flash-omni-2.0",), {})]
    assert isinstance(config_manager.config, MingOmniSpeechPipelineConfig)
    assert config_manager.config.model_path == "inclusionAI/Ming-flash-omni-2.0"
    assert config_manager.config.terminal_stages == ["decode", "talker"]


def test_ming_config_manager_resolves_single_architecture_attribute(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sglang_omni.config.manager.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            architecture="BailingMM2NativeForConditionalGeneration"
        ),
    )

    config_manager = ConfigManager.from_model_path("dummy-ming")

    assert isinstance(config_manager.config, MingOmniSpeechPipelineConfig)


def test_ming_registry_keeps_thinker_architecture_alias() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("BailingMM2NativeForConditionalGeneration")
        is MingOmniSpeechPipelineConfig
    )
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("BailingMoeV2ForCausalLM")
        is MingOmniSpeechPipelineConfig
    )


def test_ming_hf_config_registration_does_not_import_thinker() -> None:
    import sys

    from sglang_omni.models.ming_omni import registration

    sys.modules.pop("sglang_omni.models.ming_omni.thinker", None)
    registration._ming_hf_config_registered = False

    registration.register_ming_hf_config()

    assert "sglang_omni.models.ming_omni.thinker" not in sys.modules


def test_ming_text_variant_uses_text_image_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(
        "sglang_omni.config.manager.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            architectures=["BailingMM2NativeForConditionalGeneration"]
        ),
    )

    config_manager = ConfigManager.from_model_path("dummy-ming", variant="text")

    assert isinstance(config_manager.config, MingOmniPipelineConfig)
    assert [stage.name for stage in config_manager.config.stages] == [
        "preprocessing",
        "audio_encoder",
        "image_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
    ]
    assert config_manager.config.terminal_stages == ["decode"]


def _resolve_tp(config: PipelineConfig, flags: list[tuple[str, str]]) -> PipelineConfig:
    """Apply dotted TP flags plus the derived engine overrides, as serve does."""
    merged = ConfigManager(config).merge_config(flags)
    return apply_tensor_parallel_engine_overrides(merged)


def test_ming_cli_applies_tp_gpus_and_disable_custom_all_reduce(monkeypatch) -> None:
    monkeypatch.setattr(
        "sglang_omni.cli.serve.should_disable_custom_all_reduce_for_gpus",
        lambda *args, **kwargs: True,
    )
    config = MingOmniPipelineConfig(model_path="dummy")

    resolved = _resolve_tp(
        config, [("thinker.tp_size", "4"), ("thinker.gpu", "[0, 1, 2, 3]")]
    )

    thinker = _stage(resolved, "thinker")
    assert thinker.tp_size == 4
    assert thinker.gpu == [0, 1, 2, 3]
    assert (
        _server_args_overrides(resolved, "thinker")["disable_custom_all_reduce"] is True
    )


def test_ming_cli_enables_custom_all_reduce_on_p2p_mesh(monkeypatch) -> None:
    monkeypatch.setattr(
        "sglang_omni.cli.serve.should_disable_custom_all_reduce_for_gpus",
        lambda *args, **kwargs: False,
    )
    config = MingOmniPipelineConfig(model_path="dummy")

    resolved = _resolve_tp(
        config, [("thinker.tp_size", "4"), ("thinker.gpu", "[0, 1, 2, 3]")]
    )

    assert (
        _server_args_overrides(resolved, "thinker")["disable_custom_all_reduce"]
        is False
    )


def test_hard_custom_all_reduce_disable_is_not_topology_relaxed(
    monkeypatch,
) -> None:
    from typing import ClassVar

    from sglang_omni.config import EngineStageConfig

    class HardDisableConfig(PipelineConfig):
        stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
            "thinker": EngineStageConfig,
        }

        @classmethod
        def tensor_parallel_server_args_overrides(
            cls,
            *,
            stage_name: str,
            tp_size: int,
        ) -> dict[str, object]:
            if stage_name == "thinker" and tp_size > 1:
                return {"disable_custom_all_reduce": True}
            return {}

    monkeypatch.setattr(
        "sglang_omni.cli.serve.should_disable_custom_all_reduce_for_gpus",
        lambda *args, **kwargs: False,
    )
    config = HardDisableConfig(
        model_path="dummy",
        stages=[
            EngineStageConfig(
                name="thinker",
                factory_path="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                gpu=[0, 1],
                tp_size=2,
                process="thinker",
                terminal=True,
            )
        ],
    )

    resolved = apply_tensor_parallel_engine_overrides(config)

    assert (
        _server_args_overrides(resolved, "thinker")["disable_custom_all_reduce"] is True
    )


def test_topology_gated_custom_all_reduce_reuses_topology_decision(
    monkeypatch,
) -> None:
    from typing import ClassVar

    from sglang_omni.config import EngineStageConfig

    class TwoStageTopologyGatedConfig(PipelineConfig):
        stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
            "thinker": EngineStageConfig,
            "encoder": EngineStageConfig,
        }

        @classmethod
        def tensor_parallel_server_args_overrides(
            cls,
            *,
            stage_name: str,
            tp_size: int,
        ) -> dict[str, object]:
            if stage_name in {"thinker", "encoder"} and tp_size > 1:
                return {"disable_custom_all_reduce": True}
            return {}

        @classmethod
        def topology_gated_custom_all_reduce_stages(cls) -> set[str]:
            return {"thinker", "encoder"}

    calls = []
    monkeypatch.setattr(
        "sglang_omni.cli.serve.should_disable_custom_all_reduce_for_gpus",
        lambda gpu_ids: calls.append(tuple(gpu_ids)) or False,
    )
    config = TwoStageTopologyGatedConfig(
        model_path="dummy",
        stages=[
            EngineStageConfig(
                name="thinker",
                factory_path="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                gpu=[0, 1],
                tp_size=2,
                process="thinker",
                terminal=True,
            ),
            EngineStageConfig(
                name="encoder",
                factory_path="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                gpu=[0, 1],
                tp_size=2,
                process="encoder",
                terminal=True,
            ),
        ],
    )

    resolved = apply_tensor_parallel_engine_overrides(config)

    assert calls == [(0, 1)]
    assert (
        _server_args_overrides(resolved, "thinker")["disable_custom_all_reduce"]
        is False
    )
    assert (
        _server_args_overrides(resolved, "encoder")["disable_custom_all_reduce"]
        is False
    )


def test_ming_cli_applies_image_encoder_tp_and_gpus() -> None:
    config = MingOmniPipelineConfig(model_path="dummy")

    merged = ConfigManager(config).merge_config(
        [("image_encoder.tp_size", "2"), ("image_encoder.gpu", "[4, 5]")]
    )

    image_encoder = _stage(merged, "image_encoder")
    assert image_encoder.tp_size == 2
    assert image_encoder.gpu == [4, 5]


def test_ming_cli_rejects_image_encoder_gpu_count_mismatch() -> None:
    config = MingOmniPipelineConfig(model_path="dummy")

    with pytest.raises(ValueError, match="gpu has 1 entries"):
        ConfigManager(config).merge_config(
            [("image_encoder.tp_size", "2"), ("image_encoder.gpu", "[4]")]
        )


def test_ming_cli_leaves_image_encoder_untouched_when_flags_omitted() -> None:
    config = MingOmniPipelineConfig(model_path="dummy")
    before_tp = _stage(config, "image_encoder").tp_size
    before_gpu = _stage(config, "image_encoder").gpu

    merged = ConfigManager(config).merge_config(
        [("thinker.tp_size", "2"), ("thinker.gpu", "[0, 1]")]
    )

    image_encoder = _stage(merged, "image_encoder")
    assert image_encoder.tp_size == before_tp
    assert image_encoder.gpu == before_gpu


def test_ming_cli_applies_tp_server_args_for_config_declared_tp(monkeypatch) -> None:
    """The TP-derived engine overrides fire on the resolved config, whether
    the TP setting came from a flag or from the config itself."""
    monkeypatch.setattr(
        "sglang_omni.cli.serve.should_disable_custom_all_reduce_for_gpus",
        lambda *args, **kwargs: True,
    )
    config = MingOmniPipelineConfig(model_path="dummy")
    merged = ConfigManager(config).merge_config(
        [("thinker.tp_size", "2"), ("thinker.gpu", "[0, 1]")]
    )

    resolved = apply_tensor_parallel_engine_overrides(merged)

    assert (
        _server_args_overrides(resolved, "thinker")["disable_custom_all_reduce"] is True
    )


def test_ming_cli_applies_thinker_sglang_server_args() -> None:
    config = MingOmniPipelineConfig(model_path="dummy")

    merged = ConfigManager(config).merge_config(
        [
            ("thinker.engine.mem_fraction_static", "0.80"),
            ("thinker.engine.cpu_offload_gb", "0"),
            ("thinker.engine.quantization", "fp8"),
        ]
    )

    overrides = _server_args_overrides(merged, "thinker")
    assert overrides["mem_fraction_static"] == 0.80
    assert overrides["cpu_offload_gb"] == 0
    assert overrides["quantization"] == "fp8"


def test_ming_cli_talker_gpu_targets_talker_stage() -> None:
    config = MingOmniSpeechPipelineConfig(model_path="dummy")

    merged = ConfigManager(config).merge_config(
        [
            ("thinker.tp_size", "2"),
            ("thinker.gpu", "[0, 1]"),
            ("talker.gpu", "3"),
        ]
    )

    assert _stage(merged, "thinker").gpu == [0, 1]
    assert _stage(merged, "talker").gpu == 3


def test_ming_text_cli_flag_for_a_missing_stage_names_the_real_ones() -> None:
    """The text pipeline has no talker; the dotted head refusal lists what
    does exist instead of writing into nothing."""
    config = MingOmniPipelineConfig(model_path="dummy")

    with pytest.raises(Exception, match="thinker"):
        ConfigManager(config).merge_config([("talker.gpu", "3")])


def test_qwen_cli_talker_gpu_still_targets_talker_ar_stage() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    merged = ConfigManager(config).merge_config(
        [("talker_ar.gpu", "4"), ("code2wav.gpu", "5")]
    )

    assert _stage(merged, "talker_ar").gpu == 4
    assert _stage(merged, "code2wav").gpu == 5


def test_registry_rejects_duplicate_architecture_aliases(tmp_path, monkeypatch) -> None:
    package_dir = tmp_path / "fake_models"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    for model_name, architecture in (
        ("model_a", "FakeArchA"),
        ("model_b", "FakeArchB"),
    ):
        model_dir = package_dir / model_name
        model_dir.mkdir()
        (model_dir / "__init__.py").write_text("", encoding="utf-8")
        (model_dir / "config.py").write_text(
            "\n".join(
                [
                    "from typing import ClassVar",
                    "from sglang_omni.config import PipelineConfig",
                    "",
                    f"class FakeConfig(PipelineConfig):",
                    f"    architecture: ClassVar[str] = {architecture!r}",
                    "    architecture_aliases: ClassVar[tuple[str, ...]] = ("
                    "'SharedArch',)",
                    "",
                    "EntryClass = FakeConfig",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    monkeypatch.syspath_prepend(str(tmp_path))
    import_pipeline_configs.cache_clear()

    with pytest.raises(ValueError, match="SharedArch"):
        import_pipeline_configs("fake_models", "config")

    import_pipeline_configs.cache_clear()


def test_omni_serve_builds_ming_text_config_without_launching(monkeypatch) -> None:
    monkeypatch.setattr(
        "sglang_omni.cli.serve.should_disable_custom_all_reduce_for_gpus",
        lambda *args, **kwargs: True,
    )

    from typer.testing import CliRunner

    from sglang_omni.cli import app

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "sglang_omni.config.manager.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            architectures=["BailingMM2NativeForConditionalGeneration"]
        ),
    )

    def fake_launch_server(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs

    monkeypatch.setattr("sglang_omni.cli.serve.launch_server", fake_launch_server)

    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--model-path",
            "inclusionAI/Ming-flash-omni-2.0",
            "--text-only",
            "--thinker.tp_size",
            "4",
            "--thinker.gpu",
            "[0, 1, 2, 3]",
            "--thinker.engine.cpu_offload_gb",
            "0",
            "--mem-fraction-static",
            "0.8",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--model-name",
            "ming-omni",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert type(config).__name__ == "MingOmniPipelineConfig"
    assert _stage(config, "thinker").tp_size == 4
    assert _stage(config, "thinker").gpu == [0, 1, 2, 3]
    overrides = _server_args_overrides(config, "thinker")
    assert overrides["cpu_offload_gb"] == 0
    assert overrides["disable_custom_all_reduce"] is True
    assert overrides["mem_fraction_static"] == 0.8
    assert captured["kwargs"]["host"] == "127.0.0.1"
    assert captured["kwargs"]["port"] == 8000
    assert captured["kwargs"]["model_name"] == "ming-omni"


def test_omni_serve_builds_ming_speech_config_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        "sglang_omni.cli.serve.should_disable_custom_all_reduce_for_gpus",
        lambda *args, **kwargs: True,
    )

    from typer.testing import CliRunner

    from sglang_omni.cli import app

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "sglang_omni.config.manager.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            architectures=["BailingMM2NativeForConditionalGeneration"]
        ),
    )

    def fake_launch_server(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs

    monkeypatch.setattr("sglang_omni.cli.serve.launch_server", fake_launch_server)

    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--model-path",
            "inclusionAI/Ming-flash-omni-2.0",
            "--thinker.tp_size",
            "2",
            "--thinker.gpu",
            "[0, 1]",
            "--talker.gpu",
            "3",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--model-name",
            "ming-omni",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert isinstance(config, MingOmniSpeechPipelineConfig)
    assert config.terminal_stages == ["decode", "talker"]
    assert _stage(config, "thinker").tp_size == 2
    assert _stage(config, "thinker").gpu == [0, 1]
    assert _stage(config, "talker").gpu == 3
    assert (
        _server_args_overrides(config, "thinker")["disable_custom_all_reduce"] is True
    )
    assert captured["kwargs"]["host"] == "127.0.0.1"
    assert captured["kwargs"]["port"] == 8000
    assert captured["kwargs"]["model_name"] == "ming-omni"
