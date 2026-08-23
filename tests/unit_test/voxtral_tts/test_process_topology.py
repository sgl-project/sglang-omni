# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS process-boundary contracts."""

import pytest

from sglang_omni.models.voxtral_tts.config import VoxtralTTSPipelineConfig
from tests.unit_test.pipeline.helpers import build_compiled_process_topology


def test_preprocessing_to_generation_remains_process_local() -> None:
    config_data = VoxtralTTSPipelineConfig(model_path="model").model_dump()
    config_data["stages"][0]["process"] = "voxtral_preprocessing"
    config = VoxtralTTSPipelineConfig(**config_data)

    with pytest.raises(ValueError, match="Cross-process edge") as exc_info:
        build_compiled_process_topology(config)
    assert "'preprocessing' -> 'tts_generation'" in str(exc_info.value)


def test_generation_to_vocoder_remains_cross_process_safe() -> None:
    config_data = VoxtralTTSPipelineConfig(model_path="model").model_dump()
    generation, vocoder = config_data["stages"][1:]
    vocoder["process"] = "voxtral_vocoder"
    generation["runtime"]["resources"]["total_gpu_memory_fraction"] = 0.85
    vocoder["runtime"]["resources"]["total_gpu_memory_fraction"] = 0.10
    config = VoxtralTTSPipelineConfig(**config_data)

    plan = build_compiled_process_topology(config)

    assert plan.stage_to_process["tts_generation"] == "pipeline"
    assert plan.stage_to_process["vocoder"] == "voxtral_vocoder"
