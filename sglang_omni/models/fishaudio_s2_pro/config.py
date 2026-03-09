# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for FishAudio S2-Pro TTS.

Provides the PipelineConfig subclass for registry-based discovery
(via ``sglang_omni serve --config s2pro.yaml``).
"""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import ExecutorConfig, PipelineConfig, RelayConfig, StageConfig
from sglang_omni.models.fishaudio_s2_pro.pipeline.next_stage import (
    PREPROCESSING_STAGE,
    TTS_ENGINE_STAGE,
    VOCODER_STAGE,
)

_S2_PKG = "sglang_omni.models.fishaudio_s2_pro.pipeline"


class S2ProPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "FishQwen3OmniForCausalLM"

    model_path: str
    entry_stage: str = "preprocessing"
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory=f"{_S2_PKG}.stages.create_preprocessing_executor",
            ),
            get_next=f"{_S2_PKG}.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=TTS_ENGINE_STAGE,
            executor=ExecutorConfig(
                factory=f"{_S2_PKG}.stages.create_sglang_tts_engine_executor",
                args={
                    "device": "cuda:0",
                    "max_new_tokens": 2048,
                },
            ),
            get_next=f"{_S2_PKG}.next_stage.tts_engine_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=VOCODER_STAGE,
            executor=ExecutorConfig(
                factory=f"{_S2_PKG}.stages.create_vocoder_executor",
            ),
            get_next=f"{_S2_PKG}.next_stage.vocoder_next",
            relay=RelayConfig(device="cpu"),
        ),
    ]


EntryClass = S2ProPipelineConfig
