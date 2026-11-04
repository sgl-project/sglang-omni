# SPDX-License-Identifier: Apache-2.0
"""Single-stage configuration for Hy-SimulMT speech translation."""

from typing import ClassVar

from sglang_omni.config.schema import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    RealtimeTranscriptionConfig,
    StageConfig,
)
from sglang_omni.models.hy_simulmt.streaming import HySimulMTStreamingStrategy


class HySimulMTPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "HYV3SpeechForCausalLM"
    realtime_transcription: ClassVar[RealtimeTranscriptionConfig] = (
        RealtimeTranscriptionConfig(
            strategy_cls=HySimulMTStreamingStrategy,
            decode_interval_ms=1600,
            server_vad=False,
            max_segment_s=30.0,
        )
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": EngineStageConfig,
    }
    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        EngineStageConfig(
            name="asr",
            process="asr",
            factory_path="sglang_omni.models.hy_simulmt.stages.create_hy_simulmt_executor",
            factory=FactoryArgs(max_new_tokens=256),
            engine=EngineArgs(max_running_requests=16, context_length=8192),
            gpu=0,
            terminal=True,
        )
    ]

    def supports_audio_translation(self) -> bool:
        return True


EntryClass = HySimulMTPipelineConfig
