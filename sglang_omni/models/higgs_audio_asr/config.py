# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Higgs-Audio-v3-STT."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.higgs_audio_asr"


class HiggsAudioASRPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for higgs-audio-v3-stt checkpoints."""

    architecture: ClassVar[str] = "HiggsAudio3Model"

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        StageConfig(
            name="asr",
            process="asr",
            factory=f"{_PKG}.stages.create_sglang_higgs_audio_asr_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 32,
                "max_new_tokens": 1024,
                "request_build_max_workers": 2,
                "request_build_max_pending": 16,
            },
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = HiggsAudioASRPipelineConfig
