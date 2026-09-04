# SPDX-License-Identifier: Apache-2.0
"""SGLang generation-stage server args through the dotted CLI surface.

The typed per-role flags are gone; a generation engine's ServerArgs are set
with the same dotted spelling as everything else
(``--<stage>.engine.max_running_requests 64``). These tests pin that the
values reach each pipeline's generation stage -- and only that stage --
through the same merge ``sgl-omni serve`` runs.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang_omni.cli.serve import serve
from sglang_omni.config import PipelineConfig
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni.models.fun_asr.config import FunASRPipelineConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.moss_tts.config import MossTTSPipelineConfig
from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.models.voxtral_tts.config import VoxtralTTSPipelineConfig
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig

TEST_MAX_TOTAL_TOKENS = 82000

GENERATION_SERVER_ARGS = {
    "max_running_requests": 64,
    "max_total_tokens": TEST_MAX_TOTAL_TOKENS,
    "cuda_graph_max_bs": 64,
}


def _serve_kwargs(**overrides) -> dict[str, object]:
    data: dict[str, object] = {
        "ctx": SimpleNamespace(args=[]),
        "model_path": "dummy",
    }
    data.update(overrides)
    return data


def _engine_overrides(config: PipelineConfig, stage_name: str) -> dict[str, object]:
    engine = config.stage_named(stage_name).engine
    return engine.overrides() if engine is not None else {}


def _set_generation_server_args(
    config: PipelineConfig, stage_name: str
) -> PipelineConfig:
    return ConfigManager(config).merge_config(
        [
            (f"{stage_name}.engine.{key}", str(value))
            for key, value in GENERATION_SERVER_ARGS.items()
        ]
    )


@pytest.mark.parametrize(
    ("config_cls", "stage_name"),
    [
        (HiggsTtsPipelineConfig, "tts_engine"),
        (Qwen3TTSPipelineConfig, "tts_engine"),
        (MossTTSPipelineConfig, "tts_engine"),
        (MossTTSLocalPipelineConfig, "tts_engine"),
        (S2ProPipelineConfig, "tts_engine"),
        (VoxtralTTSPipelineConfig, "tts_generation"),
        (Qwen3ASRPipelineConfig, "asr"),
        (WhisperASRPipelineConfig, "asr"),
        (FunASRPipelineConfig, "asr"),
    ],
)
def test_dotted_engine_flags_reach_each_generation_stage(
    config_cls: type[PipelineConfig], stage_name: str
) -> None:
    config = config_cls(model_path="dummy")
    merged = _set_generation_server_args(config, stage_name)

    overrides = _engine_overrides(merged, stage_name)
    for key, value in GENERATION_SERVER_ARGS.items():
        assert overrides[key] == value
    # The write is per stage: no other stage picked the values up.
    for stage in merged.stages:
        if stage.name == stage_name:
            continue
        assert "max_total_tokens" not in _engine_overrides(merged, stage.name)


def test_engine_flags_on_a_non_engine_stage_are_refused() -> None:
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(ValueError, match="not an engine stage"):
        ConfigManager(config).merge_config(
            [("vocoder.engine.max_running_requests", "64")]
        )


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_serve_routes_dotted_engine_flags_to_the_stage(
    from_model_path,
    launch_server,
) -> None:
    config = HiggsTtsPipelineConfig(model_path="dummy")
    from_model_path.return_value = ConfigManager(config)

    serve(
        **_serve_kwargs(
            ctx=SimpleNamespace(
                args=[
                    "--tts_engine.engine.max_running_requests",
                    "32",
                    "--tts_engine.engine.max_total_tokens",
                    str(TEST_MAX_TOTAL_TOKENS),
                ]
            )
        )
    )

    launched_config = launch_server.call_args.args[0]
    overrides = _engine_overrides(launched_config, "tts_engine")
    assert overrides["max_running_requests"] == 32
    assert overrides["max_total_tokens"] == TEST_MAX_TOTAL_TOKENS


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_serve_without_engine_flags_preserves_pipeline_default(
    from_model_path,
    launch_server,
) -> None:
    config = HiggsTtsPipelineConfig(model_path="dummy")
    from_model_path.return_value = ConfigManager(config)

    serve(**_serve_kwargs())

    launched_config = launch_server.call_args.args[0]
    assert _engine_overrides(launched_config, "tts_engine") == {}
