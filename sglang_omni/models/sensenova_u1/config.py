# SPDX-License-Identifier: Apache-2.0
"""Single-stage native T2I pipeline for SenseNova-U1.5."""

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


EntryClass = SenseNovaU1PipelineConfig
