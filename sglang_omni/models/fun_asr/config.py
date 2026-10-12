# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    PipelineConfig,
    RealtimeTranscriptionConfig,
    StageConfig,
)

from .streaming import FunASRStreamingStrategy

_PKG = "sglang_omni.models.fun_asr"


class FunASRPipelineConfig(PipelineConfig):

    architecture: ClassVar[str] = "FunAsrNanoForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "FunASRNano",
        "FunASRForConditionalGeneration",
    )
    realtime_transcription: ClassVar[RealtimeTranscriptionConfig] = (
        RealtimeTranscriptionConfig(
            strategy_cls=FunASRStreamingStrategy,
            decode_interval_ms=720,
            server_vad=True,
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
            factory_path=f"{_PKG}.stages.create_sglang_fun_asr_executor",
            gpu=0,
            terminal=True,
        )
    ]

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, bool]:
        if stage_name == "asr":
            return {"enable_encoder_cuda_graph": True}
        else:
            pass
        return {}


EntryClass = FunASRPipelineConfig
