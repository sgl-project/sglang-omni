# SPDX-License-Identifier: Apache-2.0
"""Single-stage whole-file transcription configuration for Nemotron Transcribe."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from sglang_omni.config import (
    AudioChunkingConfig,
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

PACKAGE = "sglang_omni.models.nemotron_transcribe"
ARCHITECTURE = "NeMoSpeechLMForConditionalGeneration"
MAX_AUDIO_SECONDS = 3600
DECODER_CONTEXT_TOKENS = 73728
DEFAULT_MAX_RUNNING_REQUESTS = 64
EncoderCompileMode = Literal["default", "max-autotune-no-cudagraphs"]


class NemotronTranscribeFactoryArgs(FactoryArgs):
    max_audio_seconds: float | None = Field(default=None, gt=0)
    mm_embedding_cache_size_bytes: int | None = Field(default=None, ge=0)
    chunked_prefill_size: int | None = Field(default=None, ge=1)
    encoder_max_batch_size: int | None = Field(default=None, ge=1)
    enable_pre_lm_encoder: bool | None = None
    pre_lm_cache_max_entries: int | None = Field(default=None, ge=1)
    pre_lm_cache_size_bytes: int | None = Field(default=None, ge=1)
    pre_lm_max_batch_wait_ms: int | None = Field(default=None, ge=0)
    stream_emit_interval_s: float | None = Field(default=None, ge=0)
    compile_perception: bool | None = None
    perception_cuda_graphs: bool | None = None
    perception_compile_mode: EncoderCompileMode | None = None


class NemotronTranscribeStageConfig(EngineStageConfig):
    factory: NemotronTranscribeFactoryArgs = Field(
        default_factory=NemotronTranscribeFactoryArgs
    )


class NemotronTranscribePipelineConfig(PipelineConfig):
    """Whole-file transcription. Audio up to one hour is one decoder request."""

    architecture: ClassVar[str] = ARCHITECTURE
    allow_audio_chunking: ClassVar[bool] = False
    max_native_clip_s: ClassVar[float] = float(MAX_AUDIO_SECONDS)

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": NemotronTranscribeStageConfig,
    }

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"asr": "asr"}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "asr"}

    model_path: str
    entry_stage: str = "asr"
    audio_chunking: AudioChunkingConfig = AudioChunkingConfig(
        max_audio_clip_s=float(MAX_AUDIO_SECONDS),
        max_total_audio_s=float(MAX_AUDIO_SECONDS),
        max_concurrent_chunks=1,
    )
    stages: list[StageConfig] = [
        NemotronTranscribeStageConfig(
            name="asr",
            process="asr",
            factory_path=f"{PACKAGE}.stages.create_nemotron_transcribe_executor",
            factory=NemotronTranscribeFactoryArgs(
                dtype="bfloat16",
                max_new_tokens=16384,
                context_length=DECODER_CONTEXT_TOKENS,
                max_audio_seconds=float(MAX_AUDIO_SECONDS),
                mm_embedding_cache_size_bytes=0,
                chunked_prefill_size=4096,
                encoder_max_batch_size=8,
                enable_pre_lm_encoder=True,
                pre_lm_cache_max_entries=128,
                pre_lm_cache_size_bytes=2 * 1024**3,
                pre_lm_max_batch_wait_ms=0,
                request_build_max_workers=8,
                request_build_max_pending=64,
                stream_emit_interval_s=0.05,
                compile_perception=False,
                perception_cuda_graphs=False,
                perception_compile_mode="default",
            ),
            engine=EngineArgs(
                max_running_requests=DEFAULT_MAX_RUNNING_REQUESTS,
                enable_torch_compile=False,
            ),
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = NemotronTranscribePipelineConfig
