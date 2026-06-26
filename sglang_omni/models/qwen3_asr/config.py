# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-ASR."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.qwen3_asr"


class Qwen3ASRPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for Qwen3-ASR checkpoints."""

    architecture: ClassVar[str] = "Qwen3ASRForConditionalGeneration"

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
                "request_build_max_workers": 2,
                "request_build_max_pending": 16,
            },
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = Qwen3ASRPipelineConfig
