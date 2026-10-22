# SPDX-License-Identifier: Apache-2.0
"""Shared UMM image generation pipeline for SenseNova-U1.5."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.config.schema import FactoryArgs, FactoryDefault, StageConfig
from sglang_omni.models.sensenova_u1.admission import DEFAULT_UMM_ADMISSION_LIMITS
from sglang_omni.pipeline.umm import (
    CONTROLLER_FACTORY_PATH,
    OUTPUT_FACTORY_PATH,
    ROUTE_FN_PATH,
    STREAM_DONE_FN_PATH,
    UMMPipelineConfig,
)

DEFAULT_GENERATION_DTYPE = "bfloat16"
DEFAULT_LAYERWISE_OFFLOAD = False


class SenseNovaGenerationFactoryArgs(FactoryArgs):
    cooperative_cancellation: bool = False
    dtype: Annotated[str | None, FactoryDefault(value=DEFAULT_GENERATION_DTYPE)] = None
    enable_cache_dit: bool | None = None
    cache_dit_params: dict[str, int | float] | None = None
    dit_layerwise_offload: Annotated[
        bool | None, FactoryDefault(value=DEFAULT_LAYERWISE_OFFLOAD)
    ] = None
    dit_offload_prefetch_size: Annotated[int | None, FactoryDefault(value=1)] = Field(
        default=None, ge=0, strict=True
    )
    dit_layerwise_resident_layers: Annotated[int | None, FactoryDefault(value=0)] = (
        Field(default=None, ge=0, strict=True)
    )
    dit_layerwise_residency_policy: Annotated[
        Literal["leading", "strided"] | None, FactoryDefault(value="leading")
    ] = None
    pin_cpu_memory: Annotated[bool | None, FactoryDefault(value=False)] = None


class SenseNovaGenerationStageConfig(StageConfig):
    factory: SenseNovaGenerationFactoryArgs = Field(
        default_factory=SenseNovaGenerationFactoryArgs
    )


class SenseNovaU1PipelineConfig(UMMPipelineConfig):
    architecture: ClassVar[str] = "NEOChatModel"
    image_api_registrar: ClassVar[str] = (
        "sglang_omni.models.sensenova_u1.images_api.register_images_api"
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "generate": SenseNovaGenerationStageConfig,
    }

    model_path: str
    entry_stage: str = "orchestrator"
    stages: list[StageConfig] = [  # noqa: RUF012 -- Pydantic model default
        StageConfig(
            name="orchestrator",
            process="orchestrator",
            factory_path=CONTROLLER_FACTORY_PATH,
            factory=FactoryArgs(
                adapter="sglang_omni.models.sensenova_u1.umm.SenseNovaUMMAdapter",
                phases={"generate": "generate"},
                max_dispatches=1,
                max_outputs=10,
                max_active=8,
                max_retained_bytes=256 * 1024 * 1024,
                timeout_s=600.0,
            ),
            next=["generate", "output"],
            route_fn=ROUTE_FN_PATH,
            stream_to=["output"],
            stream_done_to_fn=STREAM_DONE_FN_PATH,
        ),
        SenseNovaGenerationStageConfig(
            name="generate",
            process="sensenova_generate",
            factory_path="sglang_omni.models.sensenova_u1.stages.create_generation_executor",
            factory=SenseNovaGenerationFactoryArgs(
                dtype=DEFAULT_GENERATION_DTYPE,
                dit_layerwise_offload=DEFAULT_LAYERWISE_OFFLOAD,
                enable_cache_dit=False,
                cooperative_cancellation=True,
            ),
            gpu=0,
            next="orchestrator",
        ),
        StageConfig(
            name="output",
            process="orchestrator",
            factory_path=OUTPUT_FACTORY_PATH,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: object = None, /) -> None:
        super().model_post_init(__context)
        generate = next(stage for stage in self.stages if stage.name == "generate")
        if generate.tp_size not in (1, 2, 4, 8):
            raise ValueError("SenseNova TP supports tp_size=1, 2, 4, or 8")
        else:
            pass
        if generate.tp_size > 1 and (generate.factory.max_batch_size or 1) != 1:
            raise ValueError("SenseNova TP requires max_batch_size=1")
        else:
            pass

        controller = resolve_stage_typed_kwargs(self.stage_named("orchestrator"))
        limits = DEFAULT_UMM_ADMISSION_LIMITS
        if controller["max_outputs"] < limits.max_images:
            raise ValueError("orchestrator.max_outputs must cover max_images")
        elif not generate.factory.cooperative_cancellation:
            raise ValueError("SenseNova UMM requires cooperative_cancellation")
        else:
            pass

        replicas = self.processes.get(generate.process or generate.name)
        if replicas is not None and replicas.num_replicas > 1:
            devices = replicas.replica_devices
            if devices is not None and len(set(devices)) != len(devices):
                raise ValueError(
                    "SenseNova-U1 DP requires distinct GPUs across replicas and TP ranks"
                )
            else:
                pass
        else:
            pass


EntryClass = SenseNovaU1PipelineConfig
