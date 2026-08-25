# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MiniMax Music 3."""

from __future__ import annotations

import os
from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)

from .constants import DEFAULT_DIT_CFG_SCALE, DEFAULT_DIT_STEPS

_PKG = "sglang_omni.models.minimax_music3"


def _visible_gpu_count() -> int:
    """GPUs this process could place a stage on, without initializing CUDA."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        entries = [entry.strip() for entry in visible.split(",")]
        return len([entry for entry in entries if entry and entry != "-1"])
    import torch

    return torch.cuda.device_count()


class DitDavFactoryArgs(FactoryArgs):
    """Acoustic DIT/DAV constructor knobs, typed like the shared ones."""

    dit_steps: int | None = Field(default=None, ge=1)
    dit_cfg_scale: float | None = None
    attention_backend: str | None = None
    cache_dit: bool | None = None
    compile_acoustic: bool | None = None
    breakable_cuda_graph: bool | None = None


class DitDavStageConfig(StageConfig):
    factory: DitDavFactoryArgs = Field(default_factory=DitDavFactoryArgs)


def _stages(*, acoustic_gpu: int) -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="minimax_music3_ar",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            next="minimax_music3_ar",
        ),
        EngineStageConfig(
            name="minimax_music3_ar",
            process="minimax_music3_ar",
            factory_path=f"{_PKG}.stages.create_ar_executor",
            factory=FactoryArgs(max_concurrency=16),
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


def _colocated_stages() -> list[StageConfig]:
    return _stages(acoustic_gpu=0)


def _two_gpu_stages() -> list[StageConfig]:
    return _stages(acoustic_gpu=1)


def _colocated_placement() -> PlacementConfig:
    return PlacementConfig(require_memory_fraction_for_colocation=False)


class MiniMaxMusic3PipelineConfig(PipelineConfig):
    """AR -> DIT/DAV, laid out to fit the GPUs this process can actually see."""

    architecture: ClassVar[str] = "MiniMaxMusic3ForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "minimax_music3_ar": EngineStageConfig,
        "dit_dav": DitDavStageConfig,
    }

    stages: list[StageConfig] = Field(
        default_factory=lambda: (
            _two_gpu_stages() if _visible_gpu_count() >= 2 else _colocated_stages()
        )
    )
    placement: PlacementConfig = Field(
        default_factory=lambda: (
            PlacementConfig() if _visible_gpu_count() >= 2 else _colocated_placement()
        )
    )

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        return frozenset({("preprocessing", "minimax_music3_ar")})


class MiniMaxMusic3SingleGPUPipelineConfig(MiniMaxMusic3PipelineConfig):
    """Both stages on one GPU. Acoustic DIT/DAV is FP32, same as dual-GPU."""

    placement: PlacementConfig = Field(default_factory=_colocated_placement)
    stages: list[StageConfig] = Field(default_factory=_colocated_stages)


class MiniMaxMusic3DualGPUPipelineConfig(MiniMaxMusic3PipelineConfig):
    """DIT/DAV on a second GPU. Acoustic DIT/DAV is FP32, same as single-GPU."""

    placement: PlacementConfig = Field(default_factory=PlacementConfig)
    stages: list[StageConfig] = Field(default_factory=_two_gpu_stages)


EntryClass = MiniMaxMusic3PipelineConfig

Variants = {
    "default": MiniMaxMusic3PipelineConfig,
    "single-gpu": MiniMaxMusic3SingleGPUPipelineConfig,
    "dual-gpu": MiniMaxMusic3DualGPUPipelineConfig,
}
