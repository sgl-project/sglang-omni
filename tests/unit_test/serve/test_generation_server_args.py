# SPDX-License-Identifier: Apache-2.0
"""SGLang generation-stage server args role mapping."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer

from sglang_omni.cli.serve import _apply_stage_server_args_override, serve
from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.moss_tts.config import MossTTSPipelineConfig
from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig
from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.models.voxtral_tts.config import VoxtralTTSPipelineConfig

TEST_MAX_TOTAL_TOKENS = 82000

GENERATION_SERVER_ARGS = {
    "max_running_requests": 64,
    "max_total_tokens": TEST_MAX_TOTAL_TOKENS,
    "cuda_graph_max_bs": 64,
}


class _DummyManager:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def parse_extra_args(self, _args) -> dict[str, object]:
        return {}

    def merge_config(self, _extra_args: dict[str, object]) -> PipelineConfig:
        return self.config


def _serve_kwargs(**overrides) -> dict[str, object]:
    data: dict[str, object] = {
        "ctx": SimpleNamespace(args=[]),
        "model_path": "dummy",
    }
    data.update(overrides)
    return data


def _stage_args(config: PipelineConfig, stage_name: str) -> dict[str, object]:
    stage = next(s for s in config.stages if s.name == stage_name)
    return dict(stage.factory_args or {})


def _apply_generation_server_args(config: PipelineConfig) -> None:
    stage_name = type(config).generation_sglang_role_to_stage()["generation"]
    _apply_stage_server_args_override(
        config,
        stage_name=stage_name,
        updates=GENERATION_SERVER_ARGS,
        reason="SGLang generation server args override",
    )


class ExplicitGenerationPipelineConfig(PipelineConfig):
    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "custom_generation"}


class MissingGenerationStagePipelineConfig(PipelineConfig):
    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "missing_generation"}


def test_generation_server_args_use_explicit_role_map() -> None:
    config = ExplicitGenerationPipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="tts_engine",
                process="pipeline",
                factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                next="custom_generation",
            ),
            StageConfig(
                name="custom_generation",
                process="pipeline",
                factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                terminal=True,
            ),
        ],
    )
    _apply_generation_server_args(config)

    assert "server_args_overrides" not in _stage_args(config, "tts_engine")
    overrides = _stage_args(config, "custom_generation")["server_args_overrides"]
    assert overrides == GENERATION_SERVER_ARGS


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_generation_server_args_absent_preserves_pipeline_default(
    from_model_path,
    launch_server,
) -> None:
    config = HiggsTtsPipelineConfig(model_path="dummy")
    from_model_path.return_value = _DummyManager(config)

    serve(**_serve_kwargs())

    launched_config = launch_server.call_args.args[0]
    assert "server_args_overrides" not in _stage_args(launched_config, "tts_engine")


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_generation_server_args_explicit_override_reaches_generation_stage(
    from_model_path,
    launch_server,
) -> None:
    config = HiggsTtsPipelineConfig(model_path="dummy")
    from_model_path.return_value = _DummyManager(config)

    serve(
        **_serve_kwargs(
            max_running_requests=32,
            max_total_tokens=TEST_MAX_TOTAL_TOKENS,
        )
    )

    launched_config = launch_server.call_args.args[0]
    overrides = _stage_args(launched_config, "tts_engine")["server_args_overrides"]
    assert overrides == {
        "max_running_requests": 32,
        "max_total_tokens": TEST_MAX_TOTAL_TOKENS,
    }


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_generation_server_args_explicit_override_rejects_unsupported_config(
    from_model_path,
    launch_server,
) -> None:
    config = PipelineConfig(
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
    from_model_path.return_value = _DummyManager(config)

    with pytest.raises(typer.BadParameter, match="not supported"):
        serve(**_serve_kwargs(cuda_graph_max_bs=128))

    launch_server.assert_not_called()


@pytest.mark.parametrize(
    "config_cls",
    [
        HiggsTtsPipelineConfig,
        Qwen3TTSPipelineConfig,
        MossTTSPipelineConfig,
        MossTTSLocalPipelineConfig,
        S2ProPipelineConfig,
    ],
)
def test_generation_server_args_support_migrated_tts_configs(
    config_cls: type[PipelineConfig],
) -> None:
    config = config_cls(model_path="dummy")
    _apply_generation_server_args(config)

    overrides = _stage_args(config, "tts_engine")["server_args_overrides"]
    assert overrides == GENERATION_SERVER_ARGS


def test_generation_server_args_support_qwen3_omni_speech() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    _apply_generation_server_args(config)

    overrides = _stage_args(config, "talker_ar")["server_args_overrides"]
    assert overrides == GENERATION_SERVER_ARGS


def test_generation_server_args_support_voxtral_tts() -> None:
    config = VoxtralTTSPipelineConfig(model_path="dummy")
    _apply_generation_server_args(config)

    overrides = _stage_args(config, "tts_generation")["server_args_overrides"]
    assert overrides == GENERATION_SERVER_ARGS


def test_generation_server_args_declared_stage_must_exist() -> None:
    config = MissingGenerationStagePipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="tts_engine",
                process="pipeline",
                factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                terminal=True,
            )
        ],
    )

    with pytest.raises(typer.BadParameter, match="missing_generation"):
        _apply_generation_server_args(config)
