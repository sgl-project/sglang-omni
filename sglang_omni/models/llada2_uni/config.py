# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for LLaDA2-Uni (Diffusion LLM)."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.llada2_uni"

PREPROCESSING_STAGE = "preprocessing"
IMAGE_STAGE = "image_encoder"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"

DEFAULT_THINKER_MAX_NEW_TOKENS = 2048


class _LLaDA2UniBasePipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "LLaDA2MoeModelLM"
    tensor_parallel_disable_custom_all_reduce_stages: ClassVar[tuple[str, ...]] = (
        THINKER_STAGE,
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {THINKER_STAGE: THINKER_STAGE}


class LLaDA2UniPipelineConfig(_LLaDA2UniBasePipelineConfig):
    """4-stage DLLM pipeline: preprocessing → image_encoder → thinker → decode."""

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


class LLaDA2UniTextPipelineConfig(_LLaDA2UniBasePipelineConfig):
    """3-stage text DLLM pipeline: preprocessing → thinker → decode."""

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"thinker_max_seq_len": 8192},
            runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
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


EntryClass = LLaDA2UniPipelineConfig

Variants = {
    "text": LLaDA2UniTextPipelineConfig,
}
