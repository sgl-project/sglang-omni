# SPDX-License-Identifier: Apache-2.0
"""Text-only Cosmos3-Nano pipeline configuration."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.cosmos3"

PREPROCESSING_STAGE = "preprocessing"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"


def _text_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_seq_len=8192),
            next=THINKER_STAGE,
        ),
        EngineStageConfig(
            name=THINKER_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_text_executor_from_config",
            factory=FactoryArgs(max_seq_len=8192, enable_async_decode=False),
            gpu=0,
            tp_size=1,
            next=DECODE_STAGE,
            stream_to=[DECODE_STAGE],
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class Cosmos3TextPipelineConfig(PipelineConfig):
    """Three-stage MVP: preprocessing → text AR → detokenize."""

    architecture: ClassVar[str] = "Cosmos3ForConditionalGeneration"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
    }

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {THINKER_STAGE: THINKER_STAGE}

    model_path: str
    revision: str | None = None
    stages: list[StageConfig] = Field(default_factory=_text_stages)

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        del stage_name
        return {"revision": self.revision} if self.revision is not None else {}


EntryClass = Cosmos3TextPipelineConfig

Variants = {"text": Cosmos3TextPipelineConfig}

__all__ = ["Cosmos3TextPipelineConfig", "EntryClass", "Variants"]
