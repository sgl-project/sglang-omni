# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MiMo-V2.5-ASR transcription."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
)

_PKG = "sglang_omni.models.mimo_v25_asr"
MIMO_AUDIO_TOKENIZER_PATH = "XiaomiMiMo/MiMo-Audio-Tokenizer"


def _mimo_v25_asr_stages() -> list[StageConfig]:
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
            next="asr",
        ),
        StageConfig(
            name="asr",
            process="asr",
            factory=f"{_PKG}.stages.create_asr_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 32,
                "max_new_tokens": 256,
            },
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.92)
            ),
            terminal=True,
        ),
    ]


class MiMoV25ASRPipelineConfig(PipelineConfig):
    """Two-stage, single-GPU MiMo-V2.5-ASR transcription pipeline.

    The tokenizer stage converts a normalized waveform into eight-channel RVQ
    codes. The ASR stage runs SGLang Qwen2 decode and returns text only.
    """

    architecture: ClassVar[str] = "MiMoV2ASRForCausalLM"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"tokenizer": "audio_tokenizer", "asr": "asr"}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "asr"}

    model_path: str
    entry_stage: str = "audio_tokenizer"
    stages: list[StageConfig] = Field(default_factory=_mimo_v25_asr_stages)


EntryClass = MiMoV25ASRPipelineConfig
