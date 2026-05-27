# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-TTS Base."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.qwen3_tts"


class Qwen3TTSPipelineConfig(PipelineConfig):
    """3-stage Qwen3-TTS Base pipeline: preprocessing -> engine -> vocoder."""

    architecture: ClassVar[str] = "Qwen3TTSForConditionalGeneration"

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"gpu_id": 0, "dtype": "bfloat16"},
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"gpu_id": 0, "dtype": "bfloat16"},
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = Qwen3TTSPipelineConfig
