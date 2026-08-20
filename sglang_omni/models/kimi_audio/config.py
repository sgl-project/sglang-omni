# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Kimi-Audio text generation."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.kimi_audio"


class KimiAudioPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "MoonshotKimiaForCausalLM"
    model_ids: ClassVar[tuple[str, ...]] = ("moonshotai/Kimi-Audio-7B-Instruct",)

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "generation"}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "generation"}

    model_path: str
    entry_stage: str = "generation"
    stages: ClassVar[list[StageConfig]] = [
        StageConfig(
            name="generation",
            process="generation",
            factory=f"{_PKG}.stages.create_kimi_audio_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 8,
                "max_new_tokens": 512,
                "request_build_max_workers": 2,
                "request_build_max_pending": 8,
            },
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = KimiAudioPipelineConfig

__all__ = ["KimiAudioPipelineConfig"]
