# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer

from sglang_omni_v1.cli.serve import (
    apply_cuda_graph_cli_overrides,
    apply_parallelism_cli_overrides,
    apply_torch_compile_cli_overrides,
    serve,
)
from sglang_omni_v1.config import (
    PipelineConfig,
    StageConfig,
    resolve_stage_factory_args,
)
from sglang_omni_v1.models.qwen3_omni.config import (
    Qwen3OmniSpeechColocatedPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from sglang_omni_v1.models.registry import PIPELINE_CONFIG_REGISTRY


class _DummyManager:
    def __init__(self):
        self.config = PipelineConfig(
            model_path="dummy",
            stages=[
                StageConfig(
                    name="stage",
                    factory="tests.v1_dummy_factories.dummy_factory",
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
        topology="speech",
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
        thinker_torch_compile="default",
        talker_torch_compile="default",
        thinker_torch_compile_max_bs=None,
        talker_torch_compile_max_bs=None,
    )
    data.update(overrides)
    return data


@patch("sglang_omni_v1.cli.serve.launch_server")
@patch("sglang_omni_v1.cli.serve.ConfigManager.from_model_path")
def test_v1_cli_selects_speech_colocated_topology(from_model_path, launch_server):
    from_model_path.return_value = _DummyManager()

    serve(**_serve_kwargs(topology="speech-colocated"))

    from_model_path.assert_called_once_with("dummy", variant="speech-colocated")
    launch_server.assert_called_once()


@patch("sglang_omni_v1.cli.serve.launch_server")
@patch("sglang_omni_v1.cli.serve.ConfigManager.from_model_path")
def test_v1_cli_text_only_overrides_topology(from_model_path, launch_server):
    from_model_path.return_value = _DummyManager()

    serve(**_serve_kwargs(text_only=True, topology="speech-colocated"))

    from_model_path.assert_called_once_with("dummy", variant="text")
    launch_server.assert_called_once()


def test_registry_resolves_qwen_colocated_config_by_class_name():
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name(
            "Qwen3OmniSpeechColocatedPipelineConfig"
        )
        is Qwen3OmniSpeechColocatedPipelineConfig
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
