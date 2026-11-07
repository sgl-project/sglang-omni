# SPDX-License-Identifier: Apache-2.0
"""AuK: preprocessing, conditioning, batched DiT sampling, and VAE decode."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import typer
from huggingface_hub.constants import HF_HUB_CACHE
from pydantic import Field, model_validator

from sglang_omni.config.schema import FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.models.auk import constants as C
from sglang_omni.platforms import current_platform
from sglang_omni.scheduling.stage_residency import offload_environment

_PKG = "sglang_omni.models.auk"
PREPROCESSING_STAGE = "preprocessing"
CONDITIONING_STAGE = "conditioning"
ENGINE_STAGE = "auk_engine"
DECODE_STAGE = "decode"


class AuKFactoryArgs(FactoryArgs):
    enable_serial_offload: bool = False
    dit_cuda_graph_capture_shapes: list[list[int]] | None = None


class AuKConditioningFactoryArgs(AuKFactoryArgs):
    serial_offload_source: Literal["mmap", "ram"] = "mmap"
    serial_offload_cache_dir: str | None = Field(
        default_factory=lambda: str(Path(HF_HUB_CACHE).parent / "auk-runtime-weights")
    )


class AuKStageConfig(StageConfig):
    factory: AuKFactoryArgs = Field(default_factory=AuKFactoryArgs)


class AuKConditioningStageConfig(AuKStageConfig):
    factory: AuKConditioningFactoryArgs = Field(
        default_factory=AuKConditioningFactoryArgs
    )


class AuKPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "AuKForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = ("AuK", "AuK-Flash")
    required_speech_reference_count: ClassVar[int | None] = None
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        CONDITIONING_STAGE: AuKConditioningStageConfig,
        ENGINE_STAGE: AuKStageConfig,
        DECODE_STAGE: AuKStageConfig,
    }

    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_concurrency=8),
            next=CONDITIONING_STAGE,
        ),
        AuKConditioningStageConfig(
            name=CONDITIONING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_conditioning_executor",
            factory=AuKConditioningFactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                text_encoder_path=C.DEFAULT_TEXT_ENCODER,
                max_batch_size=8,
                max_batch_wait_ms=10,
            ),
            gpu=0,
            next=ENGINE_STAGE,
        ),
        AuKStageConfig(
            name=ENGINE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_auk_engine_executor",
            factory=AuKFactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                nfe=C.DEFAULT_NFE,
                enable_dit_fused_qk_norm_rope=True,
                cfg_strength=C.DEFAULT_CFG_STRENGTH,
                sway_sampling_coef=C.DEFAULT_SWAY_SAMPLING_COEF,
                max_seconds=C.MAX_SECONDS,
                max_batch_size=16,
                max_batch_wait_ms=10,
                weight_dtype="bfloat16",
                enable_dit_torch_compile=True,
                enable_dit_cuda_graph=True,
            ),
            gpu=0,
            next=DECODE_STAGE,
        ),
        AuKStageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_decode_executor",
            factory=AuKFactoryArgs(
                device=current_platform.device_type, max_batch_size=4
            ),
            gpu=0,
            terminal=True,
        ),
    ]

    @model_validator(mode="after")
    def validate_offload(self) -> AuKPipelineConfig:
        gpu_stages = [
            self.stage_named(name)
            for name in (CONDITIONING_STAGE, ENGINE_STAGE, DECODE_STAGE)
        ]
        enabled = [stage.factory.enable_serial_offload for stage in gpu_stages]
        if not any(enabled):
            return self
        elif not all(enabled):
            raise ValueError(
                "AuK offload must be enabled on conditioning, auk_engine and decode together"
            )
        else:
            pass
        if len(
            {stage.gpu for stage in gpu_stages if isinstance(stage.gpu, int)}
        ) != 1 or any(
            not isinstance(stage.gpu, int) or stage.tp_size != 1 for stage in gpu_stages
        ):
            raise ValueError("AuK offload requires one shared GPU with tp_size=1")
        elif (
            len({stage.process for stage in self.stages}) != 1
            or gpu_stages[0].process is None
        ):
            raise ValueError("AuK offload requires every stage in the same process")
        elif any(process.num_replicas != 1 for process in self.processes.values()):
            raise ValueError("AuK offload requires a single process replica")
        elif any(stage.factory.device != "cuda" for stage in gpu_stages):
            raise ValueError("AuK offload requires CUDA devices")
        elif any(
            stage.factory.max_batch_size != 1
            or stage.factory.max_concurrency is not None
            for stage in gpu_stages
        ):
            raise ValueError(
                "AuK offload requires max_batch_size=1; GPU stage concurrency is fixed"
            )
        elif (
            [stage.name for stage in self.stages]
            != [PREPROCESSING_STAGE, CONDITIONING_STAGE, ENGINE_STAGE, DECODE_STAGE]
            or self.resolved_entry_stage != PREPROCESSING_STAGE
            or self.stage_named(PREPROCESSING_STAGE).next != CONDITIONING_STAGE
            or gpu_stages[0].next != ENGINE_STAGE
            or gpu_stages[1].next != DECODE_STAGE
            or not gpu_stages[2].terminal
            or any(
                stage.route_fn is not None or stage.stream_to for stage in self.stages
            )
        ):
            raise ValueError("AuK offload requires the standard linear pipeline")
        else:
            return self

    def with_serial_offload(self, components: frozenset[str]) -> AuKPipelineConfig:
        if components != {"conditioning", "dit", "vae"}:
            raise typer.BadParameter(
                "AuK --stage-offload-components requires exactly conditioning,dit,vae"
            )
        else:
            pass
        configuration = self.model_dump()
        for stage in configuration["stages"]:
            if stage["name"] in (CONDITIONING_STAGE, ENGINE_STAGE, DECODE_STAGE):
                stage["factory"].update(
                    enable_serial_offload=True, max_batch_size=1, max_batch_wait_ms=0
                )
            else:
                pass
        return type(self)(**configuration)

    def resolved_env_defaults(self) -> dict[str, str]:
        environment = super().resolved_env_defaults()
        if self.stage_named(CONDITIONING_STAGE).factory.enable_serial_offload:
            return offload_environment(environment)
        else:
            return environment


EntryClass = AuKPipelineConfig
