# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Parakeet TDT ASR."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig

# note (kzhou003): the encoder attends over the whole clip, so memory, not a
# position limit, bounds one pass; longer uploads are split into clips.
MAX_NATIVE_CLIP_SECONDS = 120.0


class ParakeetTDTPipelineConfig(PipelineConfig):
    """Single-stage batched transcription pipeline."""

    architecture: ClassVar[str] = "ParakeetForTDT"
    allow_audio_chunking: ClassVar[bool] = True
    max_native_clip_s: ClassVar[float] = MAX_NATIVE_CLIP_SECONDS

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        StageConfig(
            name="asr",
            process="asr",
            factory_path="sglang_omni.models.parakeet_tdt.stages.create_parakeet_tdt_executor",
            factory=FactoryArgs(
                dtype="bfloat16",
                max_batch_size=16,
                max_batch_wait_ms=5.0,
                max_audio_seconds=MAX_NATIVE_CLIP_SECONDS,
            ),
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = ParakeetTDTPipelineConfig

__all__ = ["MAX_NATIVE_CLIP_SECONDS", "ParakeetTDTPipelineConfig"]
