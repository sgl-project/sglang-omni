# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.models.arkasr.config import ArkasrPipelineConfig
from sglang_omni.models.fun_asr.config import FunASRPipelineConfig
from sglang_omni.models.moss_transcribe_diarize.config import (
    MossTranscribeDiarizePipelineConfig,
)
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig


def test_pipeline_config_defaults_to_no_audio_translation_support() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="asr",
                process="asr",
                factory="tests.fake.create_stage",
                terminal=True,
            )
        ],
    )

    assert config.supports_audio_translation() is False


def test_whisper_config_supports_audio_translation() -> None:
    config = WhisperASRPipelineConfig(model_path="openai/whisper-large-v3")

    assert config.supports_audio_translation() is True


@pytest.mark.parametrize(
    "config_cls,model_path",
    [
        (Qwen3ASRPipelineConfig, "Qwen/Qwen3-ASR-1.7B"),
        (FunASRPipelineConfig, "FunAudioLLM/Fun-ASR-Nano-2512"),
        (ArkasrPipelineConfig, "BAAI/ARK-ASR-3B"),
        (
            MossTranscribeDiarizePipelineConfig,
            "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        ),
    ],
)
def test_other_asr_configs_do_not_support_audio_translation(
    config_cls: type[PipelineConfig], model_path: str
) -> None:
    config = config_cls(model_path=model_path)

    assert config.supports_audio_translation() is False


def test_runtime_overrides_cannot_enable_audio_translation() -> None:
    config = Qwen3ASRPipelineConfig(
        model_path="Qwen/Qwen3-ASR-1.7B",
        runtime_overrides={"asr": {"supports_audio_translation": True}},
    )

    assert config.supports_audio_translation() is False
