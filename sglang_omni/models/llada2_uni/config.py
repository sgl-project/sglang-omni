# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for LLaDA2-Uni (Diffusion LLM)."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
)

_PKG = "sglang_omni.models.llada2_uni"

PREPROCESSING_STAGE = "preprocessing"
IMAGE_STAGE = "image_encoder"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"
IMAGE_DECODE_STAGE = "image_decode"

DEFAULT_THINKER_MAX_NEW_TOKENS = 2048


class LLaDA2UniPipelineConfig(PipelineConfig):
    """4-stage DLLM pipeline: preprocessing → image_encoder → thinker → decode."""

    architecture: ClassVar[str] = "LLaDA2MoeModelLM"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {THINKER_STAGE: THINKER_STAGE}

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"thinker_max_seq_len": 8192},
            runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
            next=IMAGE_STAGE,
        ),
        StageConfig(
            name=IMAGE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_image_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            next=THINKER_STAGE,
        ),
        StageConfig(
            name=THINKER_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_dllm_thinker_executor_from_config",
            factory_args={"thinker_max_seq_len": 8192},
            gpu=0,
            next=DECODE_STAGE,
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
    ]


class LLaDA2UniOmniPipelineConfig(PipelineConfig):
    """Native text/image pipeline with task-aware terminal routing."""

    architecture: ClassVar[str] = "LLaDA2MoeModelLM"
    terminal_stages_fn: str | None = f"{_PKG}.routing.resolve_terminal_stages"
    total_gpu_memory_fraction_dict: ClassVar[dict[str, float]] = {
        IMAGE_STAGE: 0.1,
        THINKER_STAGE: 0.7,
        IMAGE_DECODE_STAGE: 0.2,
    }

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {THINKER_STAGE: THINKER_STAGE}

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"thinker_max_seq_len": 8192},
            runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
            next=IMAGE_STAGE,
        ),
        StageConfig(
            name=IMAGE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_image_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(
                    total_gpu_memory_fraction=total_gpu_memory_fraction_dict[
                        IMAGE_STAGE
                    ]
                )
            ),
            next=THINKER_STAGE,
        ),
        StageConfig(
            name=THINKER_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_dllm_thinker_executor_from_config",
            factory_args={
                "thinker_max_seq_len": 8192,
                "dllm_algorithm": "LowConfidenceCFG",
                "server_args_overrides": {
                    "disable_cuda_graph": False,
                    "max_running_requests": 3,
                },
            },
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(
                    total_gpu_memory_fraction=total_gpu_memory_fraction_dict[
                        THINKER_STAGE
                    ]
                )
            ),
            next=[DECODE_STAGE, IMAGE_DECODE_STAGE],
            route_fn=f"{_PKG}.routing.thinker_next",
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
        StageConfig(
            name=IMAGE_DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_image_decode_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(
                    total_gpu_memory_fraction=total_gpu_memory_fraction_dict[
                        IMAGE_DECODE_STAGE
                    ]
                )
            ),
            terminal=True,
        ),
    ]


EntryClass = LLaDA2UniOmniPipelineConfig

Variants = {
    "text": LLaDA2UniPipelineConfig,
    "omni": LLaDA2UniOmniPipelineConfig,
}
