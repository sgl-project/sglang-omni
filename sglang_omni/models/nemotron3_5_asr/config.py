# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Nemotron 3.5 ASR."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig

_PKG = "sglang_omni.models.nemotron3_5_asr"


class Nemotron3_5ASRFactoryArgs(FactoryArgs):
    """Deployment knobs for the model-owned RNN-T stage."""

    dtype: str | None = "float32"
    num_lookahead_tokens: Literal[0, 3, 6, 13] | None = 3
    max_batch_size: int | None = Field(default=8, ge=1)
    max_batch_wait_ms: float | None = Field(default=2.0, ge=0)
    max_pending_stream_messages: int | None = Field(default=256, ge=1)


class Nemotron3_5ASRStageConfig(StageConfig):
    factory: Nemotron3_5ASRFactoryArgs = Field(
        default_factory=Nemotron3_5ASRFactoryArgs
    )


class Nemotron3_5ASRPipelineConfig(PipelineConfig):
    """Single-stage, batched offline transcription pipeline."""

    architecture: ClassVar[str] = "Nemotron3_5AsrForRNNT"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": Nemotron3_5ASRStageConfig,
    }

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        Nemotron3_5ASRStageConfig(
            name="asr",
            process="asr",
            factory_path=f"{_PKG}.stages.create_nemotron3_5_asr_executor",
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = Nemotron3_5ASRPipelineConfig


__all__ = [
    "Nemotron3_5ASRFactoryArgs",
    "Nemotron3_5ASRPipelineConfig",
    "Nemotron3_5ASRStageConfig",
]
