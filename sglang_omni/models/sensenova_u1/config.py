# SPDX-License-Identifier: Apache-2.0
"""Single-stage native image generation pipeline for SenseNova-U1.5."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig


class SenseNovaU1PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "NEOChatModel"

    model_path: str
    entry_stage: str = "generate"
    stages: list[StageConfig] = [  # noqa: RUF012 -- Pydantic model default
        StageConfig(
            name="generate",
            process="sensenova_generate",
            factory_path="sglang_omni.models.sensenova_u1.stages.create_generation_executor",
            factory=FactoryArgs(dtype="bfloat16"),
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
