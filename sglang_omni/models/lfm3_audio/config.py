# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration of LFM3-Audio and LFM3-Omni: preprocessing -> AR generation -> decode."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

PACKAGE = "sglang_omni.models.lfm3_audio"
PREPROCESSING_STAGE = "preprocessing"
GENERATION_STAGE = "generation"
DECODE_STAGE = "decode"


class Lfm3AudioPipelineConfig(PipelineConfig):
    """Text and 16 kHz audio in (LFM3-Omni: also images and videos); text, 24 kHz speech or both out."""

    architecture: ClassVar[str] = "Lfm3AudioForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "Lfm3OmniForConditionalGeneration",
    )
    requires_model_capabilities: ClassVar[bool] = True
    supported_speech_fields: ClassVar[frozenset[str] | None] = frozenset(
        {
            "model",
            "input",
            "voice",
            "language",
            "instructions",
            "response_format",
            "sample_rate",
            "max_new_tokens",
            "seed",
            "temperature",
            "top_k",
            "repetition_penalty",
        }
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        GENERATION_STAGE: EngineStageConfig,
    }

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        return frozenset({(PREPROCESSING_STAGE, GENERATION_STAGE)})

    entry_stage: str = PREPROCESSING_STAGE
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            next=GENERATION_STAGE,
        ),
        EngineStageConfig(
            name=GENERATION_STAGE,
            process="pipeline",
            factory_path=f"{PACKAGE}.engine_builder.create_generation_executor",
            factory=FactoryArgs(speech_sampler="seeded", enable_async_decode=True),
            gpu=0,
            next=DECODE_STAGE,
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory_path=f"{PACKAGE}.stages.create_decode_executor",
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = Lfm3AudioPipelineConfig
