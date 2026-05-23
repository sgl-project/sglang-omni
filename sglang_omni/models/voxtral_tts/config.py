# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Voxtral TTS."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.models.voxtral_tts.pipeline.next_stage import (
    GENERATION_STAGE,
    PREPROCESSING_STAGE,
    VOCODER_STAGE,
)

_PKG = "sglang_omni.models.voxtral_tts.pipeline"


class VoxtralTTSPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "VoxtralTTSForConditionalGeneration"

    model_path: str
    entry_stage: str = "preprocessing"
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next=GENERATION_STAGE,
        ),
        StageConfig(
            name=GENERATION_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_generation_executor",
            factory_args={"device": "cuda:0", "max_new_tokens": 4096},
            gpu=0,
            next=VOCODER_STAGE,
        ),
        StageConfig(
            name=VOCODER_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"device": "cuda:0"},
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = VoxtralTTSPipelineConfig
