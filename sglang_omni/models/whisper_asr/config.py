# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Whisper ASR."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    AudioChunkingConfig,
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.whisper_asr"

WHISPER_MAX_INPUT_SECONDS = 30


class WhisperASRFactoryArgs(FactoryArgs):
    """Whisper's own constructor knobs, typed like the shared ones."""

    enable_encoder_cuda_graph: bool | None = None
    enable_pre_lm_encoder: bool | None = None
    pre_lm_cache_max_entries: int | None = Field(default=None, ge=1)
    pre_lm_cache_size_bytes: int | None = Field(default=None, ge=1)
    pre_lm_max_batch_size: int | None = Field(default=None, ge=1)
    pre_lm_max_batch_wait_ms: int | None = Field(default=None, ge=0)


class WhisperASRStageConfig(EngineStageConfig):
    factory: WhisperASRFactoryArgs = Field(default_factory=WhisperASRFactoryArgs)


class WhisperASRPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for Whisper checkpoints."""

    architecture: ClassVar[str] = "WhisperForConditionalGeneration"
    audio_chunking: ClassVar[AudioChunkingConfig] = AudioChunkingConfig(
        allow_audio_chunking=True,
        max_audio_clip_s=float(WHISPER_MAX_INPUT_SECONDS),
        max_native_clip_s=float(WHISPER_MAX_INPUT_SECONDS),
        min_tail_s=1.0,
        condition_on_previous_text=False,
    )

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": WhisperASRStageConfig,
    }

    def supports_audio_translation(self) -> bool:
        """Multilingual non-turbo Whisper checkpoints translate speech to English."""
        return True

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        WhisperASRStageConfig(
            name="asr",
            process="asr",
            factory_path=f"{_PKG}.stages.create_sglang_whisper_asr_executor",
            engine=EngineArgs(max_running_requests=64),
            factory=WhisperASRFactoryArgs(
                device="cuda:0",
                # The encoder CUDA-graph replay is a documented tuning knob for
                # this pipeline; disable it when profiling eager encoder
                # execution.
                enable_encoder_cuda_graph=True,
                enable_async_decode=True,
                async_decode_min_batch_size=2,
                request_build_max_workers=8,
                request_build_max_pending=16,
                prefill_coalesce_requests=2,
                prefill_coalesce_wait_ms=6.0,
                prefill_coalesce_when_idle=True,
                prefill_coalesce_requires_pending_builds=True,
                prefill_coalesce_after_builds_during_decode=False,
                enable_pre_lm_encoder=True,
                # Note(Jeffro): Whisper is special: its input is always one
                # 30-second chunk, so the encoder output is fixed-size
                # ([1500, d_model], 3.84 MB for large-v3). That means the cache
                # size can be derived from max_entries alone; so size_bytes is
                # None here.
                pre_lm_cache_max_entries=1024,
                pre_lm_cache_size_bytes=None,
                pre_lm_max_batch_size=8,
                pre_lm_max_batch_wait_ms=0,
            ),
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = WhisperASRPipelineConfig
