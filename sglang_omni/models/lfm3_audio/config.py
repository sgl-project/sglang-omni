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
            factory=FactoryArgs(
                speech_sampler="seeded",
                enable_async_decode=True,
                # note (ruilin): each prefill drains the pending decode step and runs eagerly, which bounds the engine loop at high concurrency. Holding new requests up to 30 ms (or until 32 wait) gave 46.3 vs 40.5 req/s at concurrency 64 on SeedTTS EN, H100, and 13.1 vs 13.6 at 8.
                prefill_coalesce_requests=32,
                prefill_coalesce_wait_ms=30.0,
            ),
            # Same KV pool as the in-process default (about 2.0M tokens on an H100).
            gpu_memory_fraction=0.70,
            gpu=0,
            next=DECODE_STAGE,
        ),
        StageConfig(
            name=DECODE_STAGE,
            # note (ruilin): in the engine's process the codec shares the interpreter with the scheduler loop; on its own it served 61 vs 46 req/s at concurrency 64 on SeedTTS EN, H100 (14.9 vs 12.8 at 8, 78.7 vs 48.9 at 128).
            process="decode",
            factory_path=f"{PACKAGE}.stages.create_decode_executor",
            gpu_memory_fraction=0.15,
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = Lfm3AudioPipelineConfig
