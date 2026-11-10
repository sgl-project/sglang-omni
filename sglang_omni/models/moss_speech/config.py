# SPDX-License-Identifier: Apache-2.0
"""Three-stage, non-streaming MOSS-Speech configuration."""

from typing import ClassVar

from pydantic import ConfigDict, Field

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.models.moss_speech.hf_config import (
    KV_RESERVE_TOKENS,
    MAX_CONTEXT_TOKENS,
)

PACKAGE = "sglang_omni.models.moss_speech"


class MossSpeechVocoderArgs(FactoryArgs):
    model_config = ConfigDict(extra="forbid")

    codec_path: str | None = None


class MossSpeechVocoderStage(StageConfig):
    factory: MossSpeechVocoderArgs = Field(default_factory=MossSpeechVocoderArgs)


class MossSpeechPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "MossSpeechForCausalLM"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "preprocessing": MossSpeechVocoderStage,
        "ar_engine": EngineStageConfig,
        "audio_vocoder": MossSpeechVocoderStage,
    }
    entry_stage: str | None = "preprocessing"
    stages: list[StageConfig] = [
        MossSpeechVocoderStage(
            name="preprocessing",
            process="preprocessing",
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            gpu=0,
            gpu_memory_fraction=0.08,
            next="ar_engine",
        ),
        EngineStageConfig(
            name="ar_engine",
            process="ar_engine",
            factory_path=f"{PACKAGE}.stages.create_ar_engine_executor",
            engine=EngineArgs(
                max_running_requests=4,
                max_queued_requests=20,
                max_total_tokens=MAX_CONTEXT_TOKENS + KV_RESERVE_TOKENS,
                disable_cuda_graph=False,
            ),
            gpu=0,
            gpu_memory_fraction=0.72,
            next="audio_vocoder",
        ),
        MossSpeechVocoderStage(
            name="audio_vocoder",
            process="audio_vocoder",
            factory_path=f"{PACKAGE}.stages.create_audio_vocoder_executor",
            gpu=0,
            gpu_memory_fraction=0.20,
            terminal=True,
        ),
    ]


EntryClass = MossSpeechPipelineConfig
