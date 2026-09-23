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
INTERLEAVED_COLLECT_STAGE = "interleaved_collect"

DEFAULT_THINKER_MAX_NEW_TOKENS = 2048


class LLaDA2ImageDecoderFactoryArgs(FactoryArgs):
    backend: Literal["diffusers", "sglang"] = "diffusers"
    decode_mode: Literal["normal", "decoder-turbo"] = "normal"
    num_steps: int = Field(default=50, ge=1)
    resolution_multiplier: int = Field(default=2, ge=1)
    ulysses_degree: int = Field(default=1, ge=1)
    ring_degree: int = Field(default=1, ge=1)
    attention_backend: str = "torch_sdpa"
    interleaved_nonterminal: bool = False


class LLaDA2ImageDecoderStageConfig(StageConfig):
    supports_sequence_parallel: ClassVar[bool] = True
    factory: LLaDA2ImageDecoderFactoryArgs = Field(
        default_factory=LLaDA2ImageDecoderFactoryArgs
    )

    def model_post_init(self, context: object = None) -> None:
        super().model_post_init(context)
        if self.tp_size != 1:
            raise ValueError("LLaDA image decoder uses SP, not TP")
        if self.sp_size != self.factory.ulysses_degree * self.factory.ring_degree:
            raise ValueError(
                "image decoder sp_size must equal ulysses_degree * ring_degree"
            )
        if self.sp_size > 1 and self.factory.backend != "sglang":
            raise ValueError("image decoder SP requires backend='sglang'")
        if self.factory.ring_degree > 1 and self.factory.attention_backend not in {
            "fa",
            "sage_attn",
        }:
            raise ValueError("Decoder ring parallelism requires fa or sage_attn")


class LLaDA2UniPipelineConfig(PipelineConfig):
    """4-stage DLLM pipeline: preprocessing → image_encoder → thinker → decode."""

    architecture: ClassVar[str] = "LLaDA2MoeModelLM"

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
        IMAGE_DECODE_STAGE: LLaDA2ImageDecoderStageConfig,
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
        LLaDA2ImageDecoderStageConfig(
            name=IMAGE_DECODE_STAGE,
            process=IMAGE_DECODE_STAGE,
            factory_path=f"{_PKG}.stages.create_image_decode_executor",
            factory=LLaDA2ImageDecoderFactoryArgs(),
            gpu=0,
            terminal=True,
        ),
    ]


class LLaDA2UniInterleavedPipelineConfig(LLaDA2UniPipelineConfig):
    """Text-only interleaved generation with asynchronous frame decoding."""

    stages: list[StageConfig] = [
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
            factory_path=f"{_PKG}.stages.create_sglang_dllm_thinker_executor_from_config",
            factory=FactoryArgs(max_seq_len=8192, dllm_algorithm="LowConfidenceCFG"),
            engine=EngineArgs(
                mem_fraction_static=0.75,
                max_running_requests=3,
                cuda_graph_bs=[1, 2, 3, 4],
            ),
            gpu=0,
            gpu_memory_fraction=0.75,
            next=[THINKER_STAGE, IMAGE_DECODE_STAGE, INTERLEAVED_COLLECT_STAGE],
            route_fn=f"{_PKG}.request_builders.thinker_next",
            project_payload={
                stage: f"{_PKG}.interleaved.project_interleaved_payload"
                for stage in (
                    THINKER_STAGE,
                    IMAGE_DECODE_STAGE,
                    INTERLEAVED_COLLECT_STAGE,
                )
            },
        ),
        LLaDA2ImageDecoderStageConfig(
            name=IMAGE_DECODE_STAGE,
            process=IMAGE_DECODE_STAGE,
            factory_path=f"{_PKG}.stages.create_image_decode_executor",
            factory=LLaDA2ImageDecoderFactoryArgs(interleaved_nonterminal=True),
            gpu=0,
            gpu_memory_fraction=0.2,
            next=INTERLEAVED_COLLECT_STAGE,
        ),
        StageConfig(
            name=INTERLEAVED_COLLECT_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.interleaved.create_interleaved_collector_executor",
            terminal=True,
        ),
    ]


EntryClass = LLaDA2UniOmniPipelineConfig

Variants = {
    "text": LLaDA2UniPipelineConfig,
    "omni": LLaDA2UniOmniPipelineConfig,
    "interleaved": LLaDA2UniInterleavedPipelineConfig,
}
