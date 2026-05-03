# SPDX-License-Identifier: Apache-2.0
"""V1-native Qwen3 pipeline config assertions."""

from __future__ import annotations

import pytest
import typer

from sglang_omni_v1.cli.serve import apply_parallelism_cli_overrides
from sglang_omni_v1.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig


def _stage(config: Qwen3OmniSpeechPipelineConfig, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def test_qwen3_speech_pipeline_enables_talker_feedback() -> None:
    cfg = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    talker_stage = _stage(cfg, "talker_ar")

    assert talker_stage.factory_args["feedback_enabled"] is True


def test_qwen3_speech_pipeline_default_stage_gpus() -> None:
    cfg = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    assert _stage(cfg, "thinker").gpu == 0
    assert _stage(cfg, "talker_ar").gpu == 1
    assert _stage(cfg, "code2wav").gpu == 1
    assert cfg.gpu_placement["thinker"] == 0
    assert cfg.gpu_placement["talker_ar"] == 1
    assert cfg.gpu_placement["code2wav"] == 1


def test_qwen3_parallelism_cli_overrides_thinker_tp_and_gpus() -> None:
    cfg = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_parallelism_cli_overrides(
        cfg,
        thinker_tp_size=2,
        thinker_gpus="0,1",
        talker_gpu=2,
        code2wav_gpu=3,
    )

    assert _stage(cfg, "thinker").tp_size == 2
    assert _stage(cfg, "thinker").gpu == [0, 1]
    assert _stage(cfg, "talker_ar").gpu == 2
    assert _stage(cfg, "code2wav").gpu == 3


def test_qwen3_parallelism_cli_rejects_tp_gpu_mismatch() -> None:
    cfg = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="thinker_gpus"):
        apply_parallelism_cli_overrides(
            cfg,
            thinker_tp_size=2,
            thinker_gpus="0",
            talker_gpu=None,
            code2wav_gpu=None,
        )
