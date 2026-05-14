# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for LLaDA2-Uni (Diffusion LLM)."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.llada2_uni"


class LLaDA2UniPipelineConfig(PipelineConfig):
    """4-stage DLLM pipeline: preprocessing → image_encoder → thinker → decode."""

    architecture: ClassVar[str] = "LLaDA2MoeModelLM"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker"}

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next="image_encoder",
        ),
        StageConfig(
            name="image_encoder",
            factory=f"{_PKG}.stages.create_image_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            next="thinker",
        ),
        StageConfig(
            name="thinker",
            factory=f"{_PKG}.stages.create_sglang_dllm_thinker_executor_from_config",
            factory_args={"thinker_max_seq_len": 8192},
            gpu=0,
            next="decode",
        ),
        StageConfig(
            name="decode",
            factory=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
    ]


EntryClass = LLaDA2UniPipelineConfig
