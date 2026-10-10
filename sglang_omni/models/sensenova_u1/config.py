# SPDX-License-Identifier: Apache-2.0
"""Single-stage native image generation pipeline for SenseNova-U1.5."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig


class SenseNovaGenerationFactoryArgs(FactoryArgs):
    dit_layerwise_offload: bool | None = None
    dit_offload_prefetch_size: int | None = Field(default=None, ge=0, strict=True)
    dit_layerwise_resident_layers: int | None = Field(default=None, ge=0, strict=True)
    dit_layerwise_residency_policy: Literal["leading", "strided"] | None = None
    pin_cpu_memory: bool | None = None


class SenseNovaGenerationStageConfig(StageConfig):
    factory: SenseNovaGenerationFactoryArgs = Field(
        default_factory=SenseNovaGenerationFactoryArgs
    )


class SenseNovaU1PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "NEOChatModel"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "generate": SenseNovaGenerationStageConfig,
    }

    model_path: str
    entry_stage: str = "generate"
    stages: list[StageConfig] = [  # noqa: RUF012 -- Pydantic model default
        SenseNovaGenerationStageConfig(
            name="generate",
            process="sensenova_generate",
            factory_path="sglang_omni.models.sensenova_u1.stages.create_generation_executor",
            factory=SenseNovaGenerationFactoryArgs(
                dtype="bfloat16", dit_layerwise_offload=False
            ),
            gpu=0,
            terminal=True,
        )
    ]

    def model_post_init(self, __context: object = None, /) -> None:
        super().model_post_init(__context)
        generate = next(stage for stage in self.stages if stage.name == "generate")
        if generate.tp_size != 1:
            raise ValueError("SenseNova-U1 supports DP replicas, but not TP")

        replicas = self.processes.get("sensenova_generate")
        if replicas is not None and replicas.num_replicas > 1:
            devices = replicas.replica_devices
            if devices is not None and len(set(devices)) != len(devices):
                raise ValueError(
                    "SenseNova-U1 DP requires one distinct GPU per replica"
                )


EntryClass = SenseNovaU1PipelineConfig
