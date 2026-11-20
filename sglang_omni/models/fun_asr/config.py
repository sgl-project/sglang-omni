# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

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

    def stage_factory_kwargs(
        self, stage_name: str
    ) -> dict[str, bool | Literal["prefill", "decode"]]:
        if stage_name == "asr":
            return {"enable_encoder_cuda_graph": True}
        else:
            pass
        return {}


def pd_stages() -> list[StageConfig]:
    prefill = (
        FunASRPipelineConfig.model_fields["stages"].default[0].model_copy(deep=True)
    )
    prefill.terminal = False
    prefill.next = "asr_decode"
    prefill.engine = prefill.engine.model_copy(
        update={
            "disable_radix_cache": True,
            "page_size": 1,
            "disable_speculative_decoding": True,
        }
    )
    decode = prefill.model_copy(
        update={
            "name": "asr_decode",
            "process": "asr_decode",
            "gpu": 1,
            "next": None,
            "terminal": True,
        },
        deep=True,
    )
    return [prefill, decode]


class FunASRPDPipelineConfig(FunASRPipelineConfig):
    """Explicit CUDA Prefill and Decode stages for Fun-ASR-Nano."""

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": EngineStageConfig,
        "asr_decode": EngineStageConfig,
    }
    stages: list[StageConfig] = Field(default_factory=pd_stages)

    def stage_factory_kwargs(
        self, stage_name: str
    ) -> dict[str, bool | Literal["prefill", "decode"]]:
        if stage_name == "asr":
            return {"pd_role": "prefill", "enable_encoder_cuda_graph": True}
        elif stage_name == "asr_decode":
            return {"pd_role": "decode"}
        else:
            return {}


EntryClass = FunASRPipelineConfig
Variants = {"pd": FunASRPDPipelineConfig}
