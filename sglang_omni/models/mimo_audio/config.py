# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MiMo-Audio input-to-text inference."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
)

_PKG = "sglang_omni.models.mimo_audio"
MIMO_AUDIO_TOKENIZER_PATH = "XiaomiMiMo/MiMo-Audio-Tokenizer"


def _mimo_audio_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="audio_tokenizer",
            process="audio_tokenizer",
            factory=f"{_PKG}.stages.create_audio_tokenizer_executor",
            factory_args={
                "tokenizer_path": MIMO_AUDIO_TOKENIZER_PATH,
                "device": "cuda:0",
            },
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.08)
            ),
            next="thinker",
        ),
        StageConfig(
            name="thinker",
            process="thinker",
            factory=f"{_PKG}.stages.create_thinker_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 1,
                "max_new_tokens": 256,
            },
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.92)
            ),
            terminal=True,
        ),
    ]


class MiMoAudioPipelineConfig(PipelineConfig):
    """Two-stage, single-GPU MiMo audio-input-to-text pipeline.

    The tokenizer stage converts a normalized real waveform into the official
    eight-channel RVQ representation. The thinker stage owns the SGLang Qwen2
    runtime and returns text. No output-audio stage is present.
    """

    architecture: ClassVar[str] = "MiMoAudioModel"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"tokenizer": "audio_tokenizer", "thinker": "thinker"}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "thinker"}

    model_path: str
    entry_stage: str = "audio_tokenizer"
    stages: list[StageConfig] = Field(default_factory=_mimo_audio_stages)


EntryClass = MiMoAudioPipelineConfig
