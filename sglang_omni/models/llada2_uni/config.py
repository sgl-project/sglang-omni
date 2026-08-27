# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for LLaDA2-Uni (Diffusion LLM)."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.llada2_uni"

PREPROCESSING_STAGE = "preprocessing"
IMAGE_STAGE = "image_encoder"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"

DEFAULT_THINKER_MAX_NEW_TOKENS = 2048


class LLaDA2UniPipelineConfig(PipelineConfig):
    """4-stage DLLM pipeline: preprocessing → image_encoder → thinker → decode."""

    architecture: ClassVar[str] = "LLaDA2MoeModelLM"

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_seq_len=8192),
            next=IMAGE_STAGE,
        ),
        StageConfig(
            name=IMAGE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_image_encoder_executor",
            factory=FactoryArgs(device="cuda"),
            gpu=0,
            next=THINKER_STAGE,
        ),
        EngineStageConfig(
            name=THINKER_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_dllm_thinker_executor_from_config",
            factory=FactoryArgs(max_seq_len=8192),
            gpu=0,
            next=DECODE_STAGE,
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
    ]


EntryClass = LLaDA2UniPipelineConfig

Variants = {
    "text": LLaDA2UniPipelineConfig,
}
