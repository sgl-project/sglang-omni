# SPDX-License-Identifier: Apache-2.0
"""Ming 3 ASR pipeline configuration."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config.schema import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)

PACKAGE = "sglang_omni.models.ming3_asr"


def default_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="preprocessing",
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            next="thinker",
        ),
        EngineStageConfig(
            name="thinker",
            process="thinker",
            gpu=0,
            factory_path=f"{PACKAGE}.stages.create_thinker_executor",
            factory=FactoryArgs(
                max_seq_len=8192,
                max_new_tokens=256,
                pre_lm_max_batch_size=8,
                pre_lm_max_batch_wait_ms=2,
                request_build_max_workers=8,
                request_build_max_pending=32,
            ),
            engine=EngineArgs(
                mem_fraction_static=0.75,
                max_running_requests=64,
                cuda_graph_max_bs=64,
                disable_cuda_graph=False,
                enable_torch_compile=False,
            ),
            terminal=True,
        ),
    ]


class Ming3ASRPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "MingV3ASRForConditionalGeneration"
    allow_audio_chunking: ClassVar[bool] = True
    max_native_clip_s: ClassVar[float] = 30.0
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "thinker": EngineStageConfig,
    }
    model_path: str
    entry_stage: str = "preprocessing"
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=default_stages)


EntryClass = Ming3ASRPipelineConfig
