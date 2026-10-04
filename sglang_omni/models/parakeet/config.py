# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for NVIDIA Parakeet ASR."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import AudioChunkingConfig, PipelineConfig, StageConfig

_PKG = "sglang_omni.models.parakeet"

# note: Parakeet's FastConformer encoder uses full attention with no fixed
# input window, so there is no native clip limit; chunking only bounds the
# quadratic attention cost and the padding a batch carries.
PARAKEET_DEFAULT_CLIP_S = 120.0


class ParakeetASRPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for Hugging Face Parakeet checkpoints."""

    architecture: ClassVar[str] = "ParakeetForTDT"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "ParakeetForRNNT",
        "ParakeetForCTC",
    )
    allow_audio_chunking: ClassVar[bool] = True
    audio_chunking: AudioChunkingConfig = AudioChunkingConfig(
        max_audio_clip_s=PARAKEET_DEFAULT_CLIP_S,
    )

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        StageConfig(
            name="asr",
            process="asr",
            factory_path=f"{_PKG}.stages.create_parakeet_asr_executor",
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = ParakeetASRPipelineConfig
