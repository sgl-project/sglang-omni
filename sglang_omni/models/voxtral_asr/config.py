# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Voxtral realtime ASR."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.voxtral_asr"


class VoxtralASRPipelineConfig(PipelineConfig):
    """Single-stage streaming ASR pipeline for Voxtral realtime checkpoints."""

    architecture: ClassVar[str] = "VoxtralRealtimeForConditionalGeneration"

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": EngineStageConfig,
    }

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        EngineStageConfig(
            name="asr",
            process="asr",
            factory_path=f"{_PKG}.stages.create_sglang_voxtral_asr_executor",
            factory=FactoryArgs(
                device="cuda:0",
                max_new_tokens=4096,
                request_build_max_workers=2,
                request_build_max_pending=16,
            ),
            engine=EngineArgs(
                max_running_requests=32,
                mem_fraction_static=0.8,
            ),
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = VoxtralASRPipelineConfig
