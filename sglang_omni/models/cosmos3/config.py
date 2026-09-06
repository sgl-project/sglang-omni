# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Nano thinker pipeline configuration."""

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
VISION_STAGE = "vision_encoder"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"


def _thinker_stage(*, aggregate_vision: bool = False) -> StageConfig:
    stage = EngineStageConfig(
        name=THINKER_STAGE,
        process="pipeline",
        factory_path=f"{_PKG}.stages.create_sglang_text_executor_from_config",
        factory=FactoryArgs(max_seq_len=8192, enable_async_decode=False),
        gpu=0,
        tp_size=1,
        next=DECODE_STAGE,
        stream_to=[DECODE_STAGE],
        project_payload={
            DECODE_STAGE: f"{_PKG}.request_builders.project_thinker_to_decode"
        },
    )
    if aggregate_vision:
        stage.wait_for = [PREPROCESSING_STAGE, VISION_STAGE]
        stage.wait_for_fn = f"{_PKG}.request_builders.resolve_thinker_wait_sources"
        stage.merge_fn = f"{_PKG}.merge.merge_for_thinker"
    return stage


def _decode_stage() -> StageConfig:
    return StageConfig(
        name=DECODE_STAGE,
        process="pipeline",
        factory_path=f"{_PKG}.stages.create_decode_executor",
        terminal=True,
        can_accept_stream_before_payload=True,
    )


def _vision_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_seq_len=8192, enable_vision=True),
            next=[VISION_STAGE, THINKER_STAGE],
            route_fn=f"{_PKG}.request_builders.resolve_preprocessing_next_stages",
            project_payload={
                VISION_STAGE: (
                    f"{_PKG}.request_builders.project_preprocessing_to_vision_encoder"
                ),
                THINKER_STAGE: (
                    f"{_PKG}.request_builders.project_preprocessing_to_thinker"
                ),
            },
        ),
        StageConfig(
            name=VISION_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vision_encoder_executor",
            factory=FactoryArgs(device="cuda"),
            gpu=0,
            next=THINKER_STAGE,
            project_payload={
                THINKER_STAGE: (
                    f"{_PKG}.request_builders.project_vision_encoder_to_thinker"
                )
            },
        ),
        _thinker_stage(aggregate_vision=True),
        _decode_stage(),
    ]


def _text_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_seq_len=8192, enable_vision=False),
            next=THINKER_STAGE,
        ),
        _thinker_stage(),
        _decode_stage(),
    ]


class _Cosmos3PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Cosmos3ForConditionalGeneration"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
    }

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {THINKER_STAGE: THINKER_STAGE}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": THINKER_STAGE}

    model_path: str
    revision: str | None = None

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        del stage_name
        return {"revision": self.revision} if self.revision is not None else {}


class Cosmos3PipelineConfig(_Cosmos3PipelineConfig):
    """Nano pipeline with a separately scheduled Qwen3-VL vision tower."""

    stages: list[StageConfig] = Field(default_factory=_vision_stages)


class Cosmos3TextPipelineConfig(_Cosmos3PipelineConfig):
    """Tokenizer → thinker → decode variant without loading vision weights."""

    stages: list[StageConfig] = Field(default_factory=_text_stages)


EntryClass = Cosmos3PipelineConfig

Variants = {"text": Cosmos3TextPipelineConfig}

__all__ = [
    "Cosmos3PipelineConfig",
    "Cosmos3TextPipelineConfig",
    "EntryClass",
    "Variants",
]
