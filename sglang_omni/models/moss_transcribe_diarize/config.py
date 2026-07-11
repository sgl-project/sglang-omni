# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-Transcribe-Diarize."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.models.moss_transcribe_diarize import (  # noqa: F401
    hf_config as _hf_config,
)

_PKG = "sglang_omni.models.moss_transcribe_diarize"


class MossTranscribeDiarizePipelineConfig(PipelineConfig):
    """Single-stage batched ASR/diarization pipeline for MOSS-TD checkpoints."""

    architecture: ClassVar[str] = "MossTranscribeDiarizeForConditionalGeneration"

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
            factory=f"{_PKG}.stages.create_sglang_moss_transcribe_diarize_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 16,
                "encoder_cache_size_bytes": 4 * 1024**3,
                "request_build_max_workers": 2,
                "request_build_max_pending": 16,
            },
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = MossTranscribeDiarizePipelineConfig
