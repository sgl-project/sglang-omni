"""Pipeline configuration for Voxtral TTS."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni_v1.config import (
    ExecutorConfig,
    PipelineConfig,
    RelayConfig,
    StageConfig,
)
from sglang_omni_v1.models.voxtral_tts.pipeline.next_stage import (
    GENERATION_STAGE,
    PREPROCESSING_STAGE,
    VOCODER_STAGE,
)

_PKG = "sglang_omni_v1.models.voxtral_tts.pipeline"


class VoxtralTTSPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "VoxtralTTSForConditionalGeneration"

    model_path: str
    entry_stage: str = "preprocessing"
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory=f"{_PKG}.stages.create_preprocessing_executor",
            ),
            get_next=f"{_PKG}.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=GENERATION_STAGE,
            executor=ExecutorConfig(
                factory=f"{_PKG}.stages.create_generation_executor",
                args={
                    "device": "cuda:0",
                    "max_new_tokens": 4096,
                },
            ),
            get_next=f"{_PKG}.next_stage.generation_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=VOCODER_STAGE,
            executor=ExecutorConfig(
                factory=f"{_PKG}.stages.create_vocoder_executor",
                args={
                    "device": "cuda:0",
                },
            ),
            get_next=f"{_PKG}.next_stage.vocoder_next",
            relay=RelayConfig(device="cpu"),
        ),
    ]


EntryClass = VoxtralTTSPipelineConfig
