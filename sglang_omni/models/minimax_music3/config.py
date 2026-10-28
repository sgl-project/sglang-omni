# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MiniMax Music 3."""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path
from typing import ClassVar, Literal

import torch
import typer
from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)
from sglang_omni.config.path import ConfigPath
from sglang_omni.platforms import current_platform

from .constants import DEFAULT_DIT_CFG_SCALE, DEFAULT_DIT_STEPS

_PKG = "sglang_omni.models.minimax_music3"


def visible_gpu_count() -> int:
    return torch.get_device_module(current_platform.device_type).device_count()


class DitDavFactoryArgs(FactoryArgs):
    """Acoustic DIT/DAV constructor knobs, typed like the shared ones."""

    dit_steps: int | None = Field(default=None, ge=1)
    dit_cfg_scale: float | None = None
    attention_backend: str | None = None
    cache_dit: bool | None = None
    compile_acoustic: bool | None = None
    breakable_cuda_graph: bool | None = None
    enable_serial_offload: bool | None = None


class ArFactoryArgs(FactoryArgs):
    enable_serial_offload: bool | None = None
    serial_offload_source: Literal["mmap", "ram"] = "mmap"
    serial_offload_cache_dir: str | None = None


class ArStageConfig(EngineStageConfig):
    factory: ArFactoryArgs = Field(default_factory=ArFactoryArgs)


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
        ArStageConfig(
            name="minimax_music3_ar",
            process="minimax_music3_ar",
            factory_path=f"{_PKG}.stages.create_ar_executor",
            factory=ArFactoryArgs(max_concurrency=16),
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
        "minimax_music3_ar": ArStageConfig,
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

    def resolved_env_defaults(self) -> dict[str, str]:
        environment = super().resolved_env_defaults()
        if self.stage_named("minimax_music3_ar").factory.enable_serial_offload:
            package = find_spec("torch_memory_saver")
            if package is None or package.origin is None or torch.version.cuda is None:
                raise RuntimeError(
                    "Music3 offload requires the CUDA torch_memory_saver==0.0.10 wheel"
                )
            else:
                pass
            package_directory = Path(package.origin).parent
            cuda_major = torch.version.cuda.split(".")[0]
            pattern = f"torch_memory_saver_hook_mode_preload_cu{cuda_major}.*.so"
            libraries = [
                library
                for directory in (package_directory, package_directory.parent)
                for library in directory.glob(pattern)
            ]
            if len(libraries) != 1:
                raise RuntimeError(
                    f"Music3 offload requires exactly one allocator library matching {pattern}"
                )
            else:
                pass
            library_path = str(libraries[0])
            preload = os.environ.get("LD_PRELOAD", "")
            if "torch_memory_saver" not in preload:
                environment["LD_PRELOAD"] = (
                    f"{library_path}:{preload}" if preload else library_path
                )
            else:
                pass
        else:
            pass
        return environment

    def with_serial_offload(
        self, components: frozenset[str]
    ) -> MiniMaxMusic3PipelineConfig:
        role_to_stage = {"ar": "minimax_music3_ar", "dit": "dit_dav"}
        unknown = components - role_to_stage.keys()
        missing = role_to_stage.keys() - components
        if unknown:
            raise typer.BadParameter(
                "--stage-offload-components does not support: "
                f"{', '.join(sorted(unknown))}; supported: ar, dit"
            )
        elif missing:
            raise typer.BadParameter(
                "--stage-offload-components currently requires all of: ar, dit "
                f"(missing {', '.join(sorted(missing))})"
            )
        else:
            pass
        ar_stage = self.stage_named(role_to_stage["ar"])
        dit_stage = self.stage_named(role_to_stage["dit"])
        if ar_stage.gpu != dit_stage.gpu:
            raise typer.BadParameter(
                "--stage-offload-components ar,dit requires the "
                f"{ar_stage.name!r} and {dit_stage.name!r} stages on the same GPU "
                f"(currently {ar_stage.gpu!r} and {dit_stage.gpu!r}); use the "
                "'single-gpu' config variant"
            )
        else:
            pass
        configuration = self.model_dump()
        process = ar_stage.process or ar_stage.name
        for stage in (ar_stage, dit_stage):
            for field_name, value in (
                ("process", process),
                ("factory.enable_serial_offload", True),
            ):
                ConfigPath.parse(f"stages.{stage.name}.{field_name}", type(self)).write(
                    configuration, value
                )
        return type(self)(**configuration)


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
