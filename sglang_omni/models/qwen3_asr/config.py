# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-ASR."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import AudioChunkingConfig, PipelineConfig, StageConfig

_PKG = "sglang_omni.models.qwen3_asr"


class Qwen3ASRPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for Qwen3-ASR checkpoints."""

    architecture: ClassVar[str] = "Qwen3ASRForConditionalGeneration"

    # The model reads the whole clip as prompt tokens (~13/audio second), so a
    # request stops fitting the engine context past ~115s. 60s keeps ~2x
    # margin; longer uploads are chunked at the serving layer.
    audio_chunking: ClassVar[AudioChunkingConfig] = AudioChunkingConfig(
        allow_audio_chunking=True,
        max_audio_clip_s=60.0,
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"asr": "asr"}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "asr"}

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        StageConfig(
            name="asr",
            process="asr",
            factory=f"{_PKG}.stages.create_sglang_qwen3_asr_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 32,
                # Note: Cap on generated tokens per request; generation just stops there, so
                # a too-small cap silently drops the transcript tail. Speech produces
                # ~5 output tokens per audio second: 128 only covered ~25s, and a 60s
                # chunk needs ~300 -- 640 doubles that. Safe to raise: context_length
                # is computed as 1500 + max_new_tokens + 8 (in stages.py), so it grows by
                # the same amount and the audio budget is untouched.
                "max_new_tokens": 640,
                "request_build_max_workers": 2,
                "request_build_max_pending": 16,
            },
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = Qwen3ASRPipelineConfig
