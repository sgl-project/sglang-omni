# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for ARK-ASR-3B."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.arkasr"


class ArkasrPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for ARK-ASR-3B checkpoints."""

    architecture: ClassVar[str] = "ArkasrForConditionalGeneration"

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        StageConfig(
            name="asr",
            process="asr",
            factory=f"{_PKG}.stages.create_sglang_arkasr_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 32,
                "max_new_tokens": 256,
                "request_build_max_workers": 2,
                "request_build_max_pending": 16,
            },
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = ArkasrPipelineConfig
