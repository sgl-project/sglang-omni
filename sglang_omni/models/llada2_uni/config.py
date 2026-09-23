# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for LLaDA2-Uni (Diffusion LLM)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from sglang_omni.config import (
    EngineArgs,
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
IMAGE_DECODE_STAGE = "image_decode"

DEFAULT_THINKER_MAX_NEW_TOKENS = 2048


class LLaDA2ImageDecoderFactoryArgs(FactoryArgs):
    backend: Literal["diffusers", "sglang"] = "diffusers"
    decode_mode: Literal["normal", "decoder-turbo"] = "normal"
    num_steps: int = Field(default=50, ge=1)
    resolution_multiplier: int = Field(default=2, ge=1)
    attention_backend: str = "torch_sdpa"


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
            gpu=0,
            next=THINKER_STAGE,
        ),
        EngineStageConfig(
            name=THINKER_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_dllm_thinker_executor_from_config",
            factory=FactoryArgs(max_seq_len=8192, dllm_algorithm="LowConfidenceCFG"),
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


class LLaDA2UniOmniPipelineConfig(LLaDA2UniPipelineConfig):
    """LLaDA text, image generation and editing with a shared thinker."""

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
            gpu=0,
            next=THINKER_STAGE,
        ),
        EngineStageConfig(
            name=THINKER_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_dllm_thinker_executor_from_config",
            factory=FactoryArgs(max_seq_len=8192, dllm_algorithm="LowConfidenceCFG"),
            engine=EngineArgs(mem_fraction_static=0.75),
            gpu=0,
            next=[THINKER_STAGE, DECODE_STAGE, IMAGE_DECODE_STAGE],
            route_fn=f"{_PKG}.request_builders.thinker_next",
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
        StageConfig(
            name=IMAGE_DECODE_STAGE,
            process=IMAGE_DECODE_STAGE,
            factory_path=f"{_PKG}.stages.create_image_decode_executor",
            factory=LLaDA2ImageDecoderFactoryArgs(),
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = LLaDA2UniOmniPipelineConfig

Variants = {
    "text": LLaDA2UniPipelineConfig,
    "omni": LLaDA2UniOmniPipelineConfig,
}
