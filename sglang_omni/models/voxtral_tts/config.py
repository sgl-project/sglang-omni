# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Voxtral TTS."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.models.voxtral_tts.pipeline.next_stage import (
    GENERATION_STAGE,
    PREPROCESSING_STAGE,
    VOCODER_STAGE,
)

_PKG = "sglang_omni.models.voxtral_tts.pipeline"


class VoxtralTTSPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "VoxtralTTSForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        GENERATION_STAGE: EngineStageConfig,
    }

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        # Note (kaige): this payload is transport-complete, but preserve the
        # previous process-split allowlist in this PR and relax it separately.
        return frozenset({(PREPROCESSING_STAGE, GENERATION_STAGE)})

    entry_stage: str = "preprocessing"
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            next=GENERATION_STAGE,
        ),
        EngineStageConfig(
            name=GENERATION_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_generation_executor",
            factory=FactoryArgs(max_new_tokens=4096),
            gpu=0,
            next=VOCODER_STAGE,
        ),
        StageConfig(
            name=VOCODER_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = VoxtralTTSPipelineConfig
