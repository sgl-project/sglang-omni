# SPDX-License-Identifier: Apache-2.0

from sglang_omni.models.minicpm_o.config import (
    MiniCPMOPipelineConfig,
    MiniCPMOSpeechPipelineConfig,
)


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _stage_index(config, name: str) -> int:
    return next(
        index for index, stage in enumerate(config.stages) if stage.name == name
    )


def test_text_pipeline_constructs_thinker_before_encoders() -> None:
    config = MiniCPMOPipelineConfig(model_path="model")

    assert _stage_index(config, "thinker") < _stage_index(config, "image_encoder")
    assert _stage_index(config, "thinker") < _stage_index(config, "audio_encoder")
    assert _stage(config, "thinker").process == "pipeline"
    assert _stage(config, "image_encoder").process == "pipeline"
    assert _stage(config, "audio_encoder").process == "pipeline"


def test_speech_pipeline_preserves_tp_and_process_boundaries() -> None:
    config = MiniCPMOSpeechPipelineConfig(model_path="model")

    assert _stage_index(config, "thinker") < _stage_index(config, "image_encoder")
    assert _stage_index(config, "thinker") < _stage_index(config, "audio_encoder")
    assert _stage(config, "thinker").process == "pipeline"
    assert _stage(config, "talker").process == "talker"
    assert _stage(config, "code2wav").process == "code2wav"
