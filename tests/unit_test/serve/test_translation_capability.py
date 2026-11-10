# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_omni.models.nemotron_transcribe.config import (
    NemotronTranscribePipelineConfig,
)
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig


def test_whisper_config_supports_audio_translation() -> None:
    config = WhisperASRPipelineConfig(model_path="openai/whisper-large-v3")

    assert config.supports_audio_translation() is True


def test_nemotron_transcribe_config_supports_audio_translation() -> None:
    config = NemotronTranscribePipelineConfig(model_path="unused")

    assert config.supports_audio_translation() is True
