# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-ASR."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    AudioChunkingConfig,
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    ResolvedAudioChunking,
    StageConfig,
)
from sglang_omni.models.qwen3_asr.audio_lengths import QWEN3_ASR_MAX_INPUT_SECONDS

_PKG = "sglang_omni.models.qwen3_asr"

QWEN3_ASR_AUDIO_CHUNKING = AudioChunkingConfig(
    max_audio_clip_s=30.0,
)

# The Torch MPS path is qualified only through this duration. Keep this
# backend limit separate from the operator's chunk size: the latter defaults
# to 30s for scheduling, while a stream can safely use the qualified 60s cap.
QWEN3_ASR_TORCH_MPS_MAX_AUDIO_SECONDS = 60.0


class Qwen3ASRFactoryArgs(FactoryArgs):
    """Qwen3-ASR's own constructor knobs, typed like the shared ones."""

    enable_pre_lm_encoder: bool | None = None
    pre_lm_cache_max_entries: int | None = Field(default=None, ge=1)
    pre_lm_cache_size_bytes: int | None = Field(default=None, ge=1)
    pre_lm_max_batch_size: int | None = Field(default=None, ge=1)
    pre_lm_max_batch_wait_ms: int | None = Field(default=None, ge=0)


class Qwen3ASRStageConfig(EngineStageConfig):
    factory: Qwen3ASRFactoryArgs = Field(default_factory=Qwen3ASRFactoryArgs)


class Qwen3ASRPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for Qwen3-ASR checkpoints."""

    architecture: ClassVar[str] = "Qwen3ASRForConditionalGeneration"
    allow_audio_chunking: ClassVar[bool] = True
    max_native_clip_s: ClassVar[float] = float(QWEN3_ASR_MAX_INPUT_SECONDS)
    audio_chunking: AudioChunkingConfig = QWEN3_ASR_AUDIO_CHUNKING

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": Qwen3ASRStageConfig,
    }

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        Qwen3ASRStageConfig(
            name="asr",
            process="asr",
            factory_path=f"{_PKG}.stages.create_sglang_qwen3_asr_executor",
            # Note (Jeffro): max_new_tokens is the floor for the per-request
            # output budget. The request builder will scale the actual budget
            # with audio duration.
            factory=Qwen3ASRFactoryArgs(
                device=None,
                max_new_tokens=128,
                enable_pre_lm_encoder=True,
                pre_lm_cache_max_entries=4096,
                pre_lm_cache_size_bytes=2 * 1024**3,
                pre_lm_max_batch_size=8,
                pre_lm_max_batch_wait_ms=0,
                request_build_max_workers=8,
                request_build_max_pending=32,
                prefill_coalesce_requests=16,
                prefill_coalesce_wait_ms=40,
                prefill_coalesce_when_idle=True,
                prefill_coalesce_requires_pending_builds=True,
                prefill_coalesce_after_builds_during_decode=True,
            ),
            engine=EngineArgs(
                max_running_requests=64,
                enable_torch_compile=True,
                torch_compile_max_bs=2,
            ),
            gpu=0,
            terminal=True,
        )
    ]

    @property
    def resolved_audio_chunking(self) -> ResolvedAudioChunking:
        from sglang.srt.utils.tensor_bridge import use_mlx

        from sglang_omni.platforms import current_platform

        policy = super().resolved_audio_chunking
        if not current_platform.is_mps() or use_mlx():
            return policy

        if policy.max_audio_clip_s > QWEN3_ASR_TORCH_MPS_MAX_AUDIO_SECONDS:
            raise ValueError(
                "Qwen3-ASR Torch MPS supports "
                "audio_chunking.max_audio_clip_s up to "
                f"{QWEN3_ASR_TORCH_MPS_MAX_AUDIO_SECONDS:g}s"
            )

        # note (yexiaodong): Torch MPS currently uses one clip shape for the
        # encoder path. Keep its native and whole-upload limits within the
        # qualified cap, while retaining the independently configurable chunk
        # size for non-streaming scheduling. MLX retains the model-native cap.
        max_total_audio_s = policy.max_total_audio_s
        if (
            max_total_audio_s is None
            or max_total_audio_s > QWEN3_ASR_TORCH_MPS_MAX_AUDIO_SECONDS
        ):
            max_total_audio_s = QWEN3_ASR_TORCH_MPS_MAX_AUDIO_SECONDS
        return replace(
            policy,
            max_native_clip_s=QWEN3_ASR_TORCH_MPS_MAX_AUDIO_SECONDS,
            max_total_audio_s=max_total_audio_s,
        )


EntryClass = Qwen3ASRPipelineConfig
