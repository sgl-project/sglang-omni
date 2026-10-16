# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MiniMax Music 3."""

from __future__ import annotations

from typing import ClassVar

import torch
from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)
from sglang_omni.platforms import current_platform

from .constants import DEFAULT_DIT_CFG_SCALE, DEFAULT_DIT_STEPS

_PKG = "sglang_omni.models.minimax_music3"


def visible_gpu_count() -> int:
    return torch.get_device_module(current_platform.device_type).device_count()


class MiniMaxMusic3ARFactoryArgs(FactoryArgs):
    """AR constructor knobs shared by CUDA and native MLX."""

    mlx_model_revision: str | None = None
    enable_serial_offload: bool | None = None


class MiniMaxMusic3ARStageConfig(EngineStageConfig):
    factory: MiniMaxMusic3ARFactoryArgs = Field(
        default_factory=MiniMaxMusic3ARFactoryArgs
    )


class DitDavFactoryArgs(FactoryArgs):
    """Acoustic DIT/DAV constructor knobs, typed like the shared ones."""

    dit_steps: int | None = Field(default=None, ge=1)
    dit_cfg_scale: float | None = None
    attention_backend: str | None = None
    cache_dit: bool | None = None
    compile_acoustic: bool | None = None
    breakable_cuda_graph: bool | None = None
    mlx_model_revision: str | None = None
    enable_serial_offload: bool | None = None


class DitDavStageConfig(StageConfig):
    factory: DitDavFactoryArgs = Field(default_factory=DitDavFactoryArgs)


def stages(*, acoustic_gpu: int) -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="minimax_music3_ar",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            next="minimax_music3_ar",
        ),
        MiniMaxMusic3ARStageConfig(
            name="minimax_music3_ar",
            process="minimax_music3_ar",
            factory_path=f"{_PKG}.stages.create_ar_executor",
            factory=MiniMaxMusic3ARFactoryArgs(max_concurrency=16),
            gpu=0,
            next="dit_dav",
            stream_to=["dit_dav"],
        ),
        DitDavStageConfig(
            name="dit_dav",
            process="minimax_music3_dit_dav",
            factory_path=f"{_PKG}.stages.create_dit_dav_executor",
            factory=DitDavFactoryArgs(
                dtype="float32",
                dit_steps=DEFAULT_DIT_STEPS,
                dit_cfg_scale=DEFAULT_DIT_CFG_SCALE,
                attention_backend="torch_sdpa",
                cache_dit=False,
                compile_acoustic=True,
                breakable_cuda_graph=False,
            ),
            gpu=acoustic_gpu,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


def colocated_stages() -> list[StageConfig]:
    return stages(acoustic_gpu=0)


def two_gpu_stages() -> list[StageConfig]:
    return stages(acoustic_gpu=1)


def colocated_placement() -> PlacementConfig:
    return PlacementConfig(require_memory_fraction_for_colocation=False)


class MiniMaxMusic3PipelineConfig(PipelineConfig):
    """AR -> DIT/DAV, laid out to fit the GPUs this process can actually see."""

    architecture: ClassVar[str] = "MiniMaxMusic3ForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "minimax_music3_ar": MiniMaxMusic3ARStageConfig,
        "dit_dav": DitDavStageConfig,
    }

    stages: list[StageConfig] = Field(
        default_factory=lambda: (
            two_gpu_stages() if visible_gpu_count() >= 2 else colocated_stages()
        )
    )
    placement: PlacementConfig = Field(
        default_factory=lambda: (
            PlacementConfig() if visible_gpu_count() >= 2 else colocated_placement()
        )
    )

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        return frozenset({("preprocessing", "minimax_music3_ar")})

    @classmethod
    def stage_offload_role_to_stage(cls) -> dict[str, str]:
        return {"ar": "minimax_music3_ar", "dit": "dit_dav"}


class MiniMaxMusic3SingleGPUPipelineConfig(MiniMaxMusic3PipelineConfig):
    """Both stages on one GPU. Acoustic DIT/DAV is FP32, same as dual-GPU."""

    placement: PlacementConfig = Field(default_factory=colocated_placement)
    stages: list[StageConfig] = Field(default_factory=colocated_stages)


class MiniMaxMusic3DualGPUPipelineConfig(MiniMaxMusic3PipelineConfig):
    """DIT/DAV on a second GPU. Acoustic DIT/DAV is FP32, same as single-GPU."""

    placement: PlacementConfig = Field(default_factory=PlacementConfig)
    stages: list[StageConfig] = Field(default_factory=two_gpu_stages)


EntryClass = MiniMaxMusic3PipelineConfig

Variants = {
    "default": MiniMaxMusic3PipelineConfig,
    "single-gpu": MiniMaxMusic3SingleGPUPipelineConfig,
    "dual-gpu": MiniMaxMusic3DualGPUPipelineConfig,
}
