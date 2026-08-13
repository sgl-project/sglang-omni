# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for SenseNova U1 fallback paths."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
)

_PKG = "sglang_omni.models.sensenova_u1"


class SenseNovaU1PipelineConfig(PipelineConfig):
    """Single-stage VQA pipeline for the M1/M2 understanding milestones.

    The stage intentionally uses the official NEOChatModel-compatible HF path.
    Native SGLang attention for U1 requires the M2 hybrid image-span mask and is
    not claimed by this pipeline.
    """

    architecture: ClassVar[str] = "NEOChatModel"
    architecture_aliases: ClassVar[tuple[str, ...]] = ("SenseNovaU1",)

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"understanding": "u1_vqa"}

    model_path: str
    entry_stage: str = "u1_vqa"
    stages: list[StageConfig] = [
        StageConfig(
            name="u1_vqa",
            process="u1_vqa",
            factory=f"{_PKG}.stages.create_sensenova_u1_vqa_executor",
            factory_args={
                "device": "cuda:0",
                "dtype": "bfloat16",
                "max_new_tokens": 128,
                "do_sample": False,
                "attn_backend": "auto",
                "max_concurrency": 1,
            },
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.75)
            ),
            gpu=0,
            terminal=True,
        )
    ]


class SenseNovaU1FlowPipelineConfig(SenseNovaU1PipelineConfig):
    """Single-stage T2I/IT2I flow-matching pipeline for the M3 milestone."""

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "u1_flow"}

    entry_stage: str = "u1_flow"
    stages: list[StageConfig] = [
        StageConfig(
            name="u1_flow",
            process="u1_flow",
            factory=f"{_PKG}.stages.create_sensenova_u1_flow_executor",
            factory_args={
                "device": "cuda:0",
                "dtype": "bfloat16",
                "attn_backend": "auto",
                "max_concurrency": 1,
            },
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.75)
            ),
            gpu=0,
            terminal=True,
        )
    ]


class SenseNovaU1InterleavePipelineConfig(SenseNovaU1PipelineConfig):
    """Single-stage interleaved text-image generation pipeline for M4."""

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"interleave": "u1_interleave"}

    entry_stage: str = "u1_interleave"
    stages: list[StageConfig] = [
        StageConfig(
            name="u1_interleave",
            process="u1_interleave",
            factory=f"{_PKG}.stages.create_sensenova_u1_interleave_executor",
            factory_args={
                "device": "cuda:0",
                "dtype": "bfloat16",
                "attn_backend": "auto",
                "max_concurrency": 1,
            },
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.75)
            ),
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = SenseNovaU1PipelineConfig

Variants = {
    "default": SenseNovaU1PipelineConfig,
    "flow": SenseNovaU1FlowPipelineConfig,
    "interleave": SenseNovaU1InterleavePipelineConfig,
}
