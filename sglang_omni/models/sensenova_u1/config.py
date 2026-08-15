# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for native SenseNova U1 serving paths."""

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
    """Default native interleaved text-image generation pipeline."""

    architecture: ClassVar[str] = "NEOChatModel"
    architecture_aliases: ClassVar[tuple[str, ...]] = ("SenseNovaU1",)

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"interleave": "u1_interleave"}

    model_path: str
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
                "max_total_tokens": 4096,
                "eager_prefix_cache_max_entries": 4,
                "eager_decode_graph_cache_max_entries": 2,
                "eager_decode_graph_max_captures": 4,
                "eager_prefix_cache_max_tokens": 2048,
                "eager_decode_graph_max_total_tokens": 1024,
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
                "max_total_tokens": 4096,
                "eager_prefix_cache_max_entries": 4,
                "eager_decode_graph_cache_max_entries": 2,
                "eager_decode_graph_max_captures": 4,
                "eager_prefix_cache_max_tokens": 2048,
                "eager_decode_graph_max_total_tokens": 1024,
            },
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.75)
            ),
            gpu=0,
            terminal=True,
        )
    ]


class SenseNovaU1InterleavePipelineConfig(SenseNovaU1PipelineConfig):
    """Explicit alias for the default native interleave pipeline."""


class SenseNovaU1NativePipelineConfig(SenseNovaU1PipelineConfig):
    """Native language-tower load/forward diagnostic pipeline for M6."""

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"native_language": "u1_native"}

    entry_stage: str = "u1_native"
    stages: list[StageConfig] = [
        StageConfig(
            name="u1_native",
            process="u1_native",
            factory=f"{_PKG}.stages.create_sensenova_u1_native_executor",
            factory_args={
                "device": "cpu",
                "dtype": "bfloat16",
                "load_weights": False,
            },
            terminal=True,
        )
    ]


class SenseNovaU1NativeServingPipelineConfig(SenseNovaU1PipelineConfig):
    """Native SGLang language/MoT serving path for M6 attention validation."""

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"native_serving": "u1_native_serving"}

    entry_stage: str = "u1_native_serving"
    stages: list[StageConfig] = [
        StageConfig(
            name="u1_native_serving",
            process="u1_native_serving",
            factory=f"{_PKG}.stages.create_sensenova_u1_native_serving_executor",
            factory_args={
                "device": "cuda:0",
                "dtype": "bfloat16",
                "attention_backend": "triton",
                "mem_fraction_static": 0.65,
                "max_total_tokens": 4096,
                "max_running_requests": 16,
                "max_concurrency": 16,
                "max_batch_wait_ms": 10,
                "enable_cuda_graph": True,
                "cuda_graph_bs": [1, 8, 16],
                "eager_prefix_cache_max_entries": 4,
                "eager_decode_graph_cache_max_entries": 2,
                "eager_decode_graph_max_captures": 4,
                "eager_prefix_cache_max_tokens": 2048,
                "eager_decode_graph_max_total_tokens": 1024,
            },
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.70)
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
    "native": SenseNovaU1NativePipelineConfig,
    "native_serving": SenseNovaU1NativeServingPipelineConfig,
}
