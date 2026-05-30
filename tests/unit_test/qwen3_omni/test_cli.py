# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer

from sglang_omni.cli.serve import (
    apply_cuda_graph_cli_overrides,
    apply_encoder_mem_reserve_cli_override,
    apply_parallelism_cli_overrides,
    apply_partial_start_cli_overrides,
    apply_torch_compile_cli_overrides,
    serve,
)
from sglang_omni.config import PipelineConfig, StageConfig, resolve_stage_factory_args
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
                    factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                    terminal=True,
                )
            ],
        )

    def parse_extra_args(self, args):
        return {}

    def merge_config(self, extra_args):
        return self.config


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
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=None,
        encoder_mem_reserve=None,
        log_level="info",
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=None,
        code2wav_gpu=None,
        thinker_cuda_graph="default",
        talker_cuda_graph="default",
        talker_partial_start="default",
        thinker_torch_compile="default",
        talker_torch_compile="default",
        thinker_torch_compile_max_bs=None,
        talker_torch_compile_max_bs=None,
    )
    data.update(overrides)
    return data


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _set_colocated_runtime(config: Qwen3OmniSpeechColocatedPipelineConfig) -> None:
    for stage_name, fraction in {
        "image_encoder": 0.05,
        "audio_encoder": 0.05,
        "thinker": 0.35,
        "talker_ar": 0.35,
        "code2wav": 0.05,
    }.items():
        _stage(config, stage_name).runtime.resources.total_gpu_memory_fraction = (
            fraction
        )


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
    _set_colocated_runtime(config)
    from_file.return_value = _DummyManager(config)

    serve(**_serve_kwargs(config="colocated.yaml", colocate=True))

    assert "Merged Configuration" in capsys.readouterr().out
    from_file.assert_called_once_with("colocated.yaml")
    launch_server.assert_called_once()


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_file")
def test_cli_config_can_own_model_path(from_file, launch_server):
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="config-model")
    _set_colocated_runtime(config)
    from_file.return_value = _DummyManager(config)

    serve(**_serve_kwargs(config="colocated.yaml", colocate=True, model_path=None))

    launched_config = launch_server.call_args.args[0]
    assert launched_config.model_path == "config-model"


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_file")
def test_cli_model_path_overrides_config_model_path(from_file, launch_server):
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="config-model")
    _set_colocated_runtime(config)
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


def test_qwen_text_encoder_mem_reserve_still_targets_thinker():
    config = Qwen3OmniPipelineConfig(model_path="dummy")

    apply_encoder_mem_reserve_cli_override(
        config,
        encoder_mem_reserve=0.05,
        mem_fraction_static=None,
        thinker_mem_fraction_static=None,
    )

    assert _stage(config, "thinker").factory_args["encoder_mem_reserve"] == 0.05


def test_qwen_speech_encoder_mem_reserve_still_targets_thinker():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_encoder_mem_reserve_cli_override(
        config,
        encoder_mem_reserve=0.05,
        mem_fraction_static=None,
        thinker_mem_fraction_static=None,
    )

    assert _stage(config, "thinker").factory_args["encoder_mem_reserve"] == 0.05
    assert "encoder_mem_reserve" not in _stage(config, "talker_ar").factory_args


def test_qwen_text_cli_rejects_talker_gpu_with_stable_message():
    config = Qwen3OmniPipelineConfig(model_path="dummy")

    with pytest.raises(
        typer.BadParameter,
        match="--talker-gpu is not supported by Qwen3OmniPipelineConfig",
    ):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=1,
            code2wav_gpu=None,
        )


def test_speech_colocated_rejects_talker_gpu_override_to_other_gpu():
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="--talker-gpu"):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=1,
            code2wav_gpu=None,
        )


def test_speech_colocated_rejects_code2wav_gpu_override_to_other_gpu():
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="--code2wav-gpu"):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=None,
            code2wav_gpu=1,
        )


def test_speech_colocated_allows_gpu_override_to_same_gpu():
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")

    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        talker_gpu=0,
        code2wav_gpu=0,
    )

    assert next(stage for stage in config.stages if stage.name == "talker_ar").gpu == 0
    assert next(stage for stage in config.stages if stage.name == "code2wav").gpu == 0


def test_cuda_graph_cli_override_reaches_resolved_sglang_args():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_cuda_graph_cli_overrides(
        config,
        thinker_cuda_graph="off",
        talker_cuda_graph="on",
    )

    thinker = next(stage for stage in config.stages if stage.name == "thinker")
    talker = next(stage for stage in config.stages if stage.name == "talker_ar")
    thinker_args = resolve_stage_factory_args(thinker, config)
    talker_args = resolve_stage_factory_args(talker, config)

    assert thinker_args["server_args_overrides"]["disable_cuda_graph"] is True
    assert talker_args["server_args_overrides"]["disable_cuda_graph"] is False


def test_torch_compile_cli_override_reaches_resolved_sglang_args():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_torch_compile_cli_overrides(
        config,
        thinker_torch_compile="on",
        talker_torch_compile="off",
        thinker_torch_compile_max_bs=4,
        talker_torch_compile_max_bs=2,
    )

    thinker = next(stage for stage in config.stages if stage.name == "thinker")
    talker = next(stage for stage in config.stages if stage.name == "talker_ar")
    thinker_args = resolve_stage_factory_args(thinker, config)
    talker_args = resolve_stage_factory_args(talker, config)

    assert thinker_args["server_args_overrides"]["enable_torch_compile"] is True
    assert thinker_args["server_args_overrides"]["torch_compile_max_bs"] == 4
    assert talker_args["server_args_overrides"]["enable_torch_compile"] is False
    assert talker_args["server_args_overrides"]["torch_compile_max_bs"] == 2


def test_partial_start_default_is_on():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    talker = next(stage for stage in config.stages if stage.name == "talker_ar")
    talker_args = resolve_stage_factory_args(talker, config)
    assert talker_args["enable_partial_start"] is True


def test_partial_start_cli_override_can_disable_and_enable():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_partial_start_cli_overrides(config, talker_partial_start="off")
    talker = next(stage for stage in config.stages if stage.name == "talker_ar")
    assert resolve_stage_factory_args(talker, config)["enable_partial_start"] is False

    apply_partial_start_cli_overrides(config, talker_partial_start="on")
    assert resolve_stage_factory_args(talker, config)["enable_partial_start"] is True


def test_partial_start_cli_default_preserves_config_default():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    apply_partial_start_cli_overrides(config, talker_partial_start="default")
    talker = next(stage for stage in config.stages if stage.name == "talker_ar")
    assert resolve_stage_factory_args(talker, config)["enable_partial_start"] is True


def test_partial_start_cli_invalid_mode_rejected():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_partial_start_cli_overrides(config, talker_partial_start="bogus")


def test_partial_start_cli_rejects_unsupported_config_with_stable_message():
    config = Qwen3OmniPipelineConfig(model_path="dummy")
    with pytest.raises(
        typer.BadParameter,
        match="--talker-partial-start is not supported by Qwen3OmniPipelineConfig",
    ):
        apply_partial_start_cli_overrides(config, talker_partial_start="on")
