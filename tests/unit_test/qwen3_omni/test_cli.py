# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer

from sglang_omni.cli.serve import serve
from sglang_omni.config import PipelineConfig, StageConfig, resolve_stage_factory_args
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechColocatedPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


class _DummyManager:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig(
            model_path="dummy",
            stages=[
                StageConfig(
                    name="stage",
                    process="pipeline",
                    factory_path="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                    terminal=True,
                )
            ],
        )

    def parse_extra_args(self, args):
        return ConfigManager(self.config).parse_extra_args(args)

    def merge_config(self, extra_args, *, extra_patches=None):
        return ConfigManager(self.config).merge_config(
            extra_args, extra_patches=extra_patches
        )


def _serve_kwargs(**overrides):
    data = dict(
        ctx=SimpleNamespace(args=[]),
        model_path="dummy",
        config=None,
        text_only=False,
        colocate=False,
        host="0.0.0.0",
        port=8000,
        model_name=None,
        mem_fraction_static=None,
        log_level="info",
    )
    data.update(overrides)
    return data


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _set_colocated_budgets(config: Qwen3OmniSpeechColocatedPipelineConfig) -> None:
    for stage_name, fraction in {
        "image_encoder": 0.05,
        "audio_encoder": 0.05,
        "thinker": 0.35,
        "talker_ar": 0.35,
        "code2wav": 0.05,
    }.items():
        _stage(config, stage_name).gpu_memory_fraction = fraction


@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_cli_colocate_requires_config(from_model_path):
    with pytest.raises(typer.BadParameter, match="requires --config"):
        serve(**_serve_kwargs(colocate=True))

    from_model_path.assert_not_called()


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_file")
def test_cli_colocate_accepts_budgeted_colocated_config(
    from_file,
    launch_server,
    capsys,
):
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    _set_colocated_budgets(config)
    from_file.return_value = _DummyManager(config)

    serve(**_serve_kwargs(config="colocated.yaml", colocate=True))

    assert "Merged Configuration" in capsys.readouterr().out
    from_file.assert_called_once_with("colocated.yaml")
    launch_server.assert_called_once()


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_file")
def test_cli_config_can_own_model_path(from_file, launch_server):
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="config-model")
    _set_colocated_budgets(config)
    from_file.return_value = _DummyManager(config)

    serve(**_serve_kwargs(config="colocated.yaml", colocate=True, model_path=None))

    launched_config = launch_server.call_args.args[0]
    assert launched_config.model_path == "config-model"


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_file")
def test_cli_model_path_overrides_config_model_path(from_file, launch_server):
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="config-model")
    _set_colocated_budgets(config)
    from_file.return_value = _DummyManager(config)

    serve(
        **_serve_kwargs(
            config="colocated.yaml",
            colocate=True,
            model_path="override-model",
        )
    )

    launched_config = launch_server.call_args.args[0]
    assert launched_config.model_path == "override-model"


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_file")
def test_cli_colocate_rejects_non_colocated_config(from_file, launch_server):
    from_file.return_value = _DummyManager(
        Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    )

    with pytest.raises(
        typer.BadParameter,
        match="Qwen3OmniSpeechColocatedPipelineConfig",
    ):
        serve(**_serve_kwargs(config="speech.yaml", colocate=True))

    launch_server.assert_not_called()


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_cli_uses_model_registry_default_by_default(from_model_path, launch_server):
    from_model_path.return_value = _DummyManager()

    serve(**_serve_kwargs())

    from_model_path.assert_called_once_with("dummy")
    launch_server.assert_called_once()


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_cli_requires_model_path_without_config(from_model_path, launch_server):
    with pytest.raises(typer.BadParameter, match="--model-path is required"):
        serve(**_serve_kwargs(model_path=None))

    from_model_path.assert_not_called()
    launch_server.assert_not_called()


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_cli_text_only_selects_text_variant(from_model_path, launch_server):
    from_model_path.return_value = _DummyManager()

    serve(**_serve_kwargs(text_only=True))

    from_model_path.assert_called_once_with("dummy", variant="text")
    launch_server.assert_called_once()


@pytest.mark.parametrize(
    ("config_cls", "text_only"),
    [
        (Qwen3OmniPipelineConfig, True),
        (Qwen3OmniSpeechPipelineConfig, False),
    ],
)
@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_cli_thinker_max_running_requests_targets_thinker_in_both_variants(
    from_model_path,
    launch_server,
    config_cls,
    text_only,
):
    from_model_path.return_value = _DummyManager(config_cls(model_path="dummy"))

    serve(
        **_serve_kwargs(
            text_only=text_only,
            ctx=SimpleNamespace(args=["--thinker.engine.max_running_requests", "16"]),
        )
    )

    launched_config = launch_server.call_args.args[0]
    assert (
        _stage(launched_config, "thinker").engine.overrides()["max_running_requests"]
        == 16
    )
    if isinstance(launched_config, Qwen3OmniSpeechPipelineConfig):
        talker_engine = _stage(launched_config, "talker_ar").engine
        assert "max_running_requests" not in (
            talker_engine.overrides() if talker_engine is not None else {}
        )


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_cli_hides_merged_config_for_normal_info_launch(
    from_model_path,
    launch_server,
    capsys,
):
    from_model_path.return_value = _DummyManager()

    serve(**_serve_kwargs())

    assert "Merged Configuration" not in capsys.readouterr().out
    launch_server.assert_called_once()


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_cli_prints_merged_config_at_debug(
    from_model_path,
    launch_server,
    capsys,
):
    from_model_path.return_value = _DummyManager()

    serve(**_serve_kwargs(log_level="debug"))

    assert "Merged Configuration" in capsys.readouterr().out
    launch_server.assert_called_once()


def test_cli_rejects_text_only_with_colocate():
    with pytest.raises(typer.BadParameter, match="--text-only"):
        serve(**_serve_kwargs(text_only=True, colocate=True))


def test_registry_resolves_qwen_colocated_config_by_class_name():
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name(
            "Qwen3OmniSpeechColocatedPipelineConfig"
        )
        is Qwen3OmniSpeechColocatedPipelineConfig
    )


@pytest.mark.parametrize(
    "config_cls",
    [Qwen3OmniPipelineConfig, Qwen3OmniSpeechPipelineConfig],
)
def test_qwen_encoder_mem_reserve_dotted_flag_targets_thinker(config_cls):
    config = config_cls(model_path="dummy")

    merged = ConfigManager(config).merge_config(
        [("thinker.factory.encoder_mem_reserve", "0.05")]
    )

    assert _stage(merged, "thinker").factory.encoder_mem_reserve == 0.05
    if isinstance(merged, Qwen3OmniSpeechPipelineConfig):
        assert _stage(merged, "talker_ar").factory.encoder_mem_reserve is None


def test_dotted_gpu_flags_move_stages(monkeypatch):
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")

    merged = ConfigManager(config).merge_config(
        [("talker_ar.gpu", "0"), ("code2wav.gpu", "0")]
    )

    assert _stage(merged, "talker_ar").gpu == 0
    assert _stage(merged, "code2wav").gpu == 0


def test_cuda_graph_dotted_flags_reach_resolved_sglang_args():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    merged = ConfigManager(config).merge_config(
        [
            ("thinker.engine.disable_cuda_graph", "true"),
            ("talker_ar.engine.disable_cuda_graph", "false"),
        ]
    )

    thinker_args = resolve_stage_factory_args(_stage(merged, "thinker"), merged)
    talker_args = resolve_stage_factory_args(_stage(merged, "talker_ar"), merged)

    assert thinker_args["server_args_overrides"]["disable_cuda_graph"] is True
    assert talker_args["server_args_overrides"]["disable_cuda_graph"] is False


def test_torch_compile_dotted_flags_reach_resolved_sglang_args():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    merged = ConfigManager(config).merge_config(
        [
            ("thinker.engine.enable_torch_compile", "true"),
            ("thinker.engine.torch_compile_max_bs", "4"),
            ("talker_ar.engine.enable_torch_compile", "false"),
            ("talker_ar.engine.torch_compile_max_bs", "2"),
        ]
    )

    thinker_args = resolve_stage_factory_args(_stage(merged, "thinker"), merged)
    talker_args = resolve_stage_factory_args(_stage(merged, "talker_ar"), merged)

    assert thinker_args["server_args_overrides"]["enable_torch_compile"] is True
    assert thinker_args["server_args_overrides"]["torch_compile_max_bs"] == 4
    assert talker_args["server_args_overrides"]["enable_torch_compile"] is False
    assert talker_args["server_args_overrides"]["torch_compile_max_bs"] == 2


def test_partial_start_default_is_on():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    talker = _stage(config, "talker_ar")
    talker_args = resolve_stage_factory_args(talker, config)
    assert talker_args["enable_partial_start"] is True


def test_partial_start_dotted_flag_can_disable_and_enable():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    disabled = ConfigManager(config).merge_config(
        [("talker_ar.factory.enable_partial_start", "false")]
    )
    talker = _stage(disabled, "talker_ar")
    assert resolve_stage_factory_args(talker, disabled)["enable_partial_start"] is False

    enabled = ConfigManager(config).merge_config(
        [("talker_ar.factory.enable_partial_start", "true")]
    )
    talker = _stage(enabled, "talker_ar")
    assert resolve_stage_factory_args(talker, enabled)["enable_partial_start"] is True


def test_partial_start_dotted_flag_rejects_a_non_boolean():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    with pytest.raises(Exception, match="bool"):
        ConfigManager(config).merge_config(
            [("talker_ar.factory.enable_partial_start", "bogus")]
        )


def test_partial_start_flag_on_a_missing_stage_names_the_real_ones():
    config = Qwen3OmniPipelineConfig(model_path="dummy")
    with pytest.raises(Exception, match="thinker"):
        ConfigManager(config).merge_config(
            [("talker_ar.factory.enable_partial_start", "true")]
        )
