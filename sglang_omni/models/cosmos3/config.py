# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 generation configuration backed by SGLang multimodal_gen."""

from typing import ClassVar

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig


class Cosmos3PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Cosmos3ForConditionalGeneration"
    native_media_stage: ClassVar[str] = "generation"
    native_media_factory_path: ClassVar[str] = (
        "sglang_omni.models.cosmos3.media.prepare_native_media_app"
    )
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "Cosmos3EdgeForConditionalGeneration",
    )

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="generation",
            process="generation",
            factory_path="sglang_omni.models.cosmos3.stages.create_generation_scheduler",
            allow_child_processes=True,
            factory=FactoryArgs(),
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = Cosmos3PipelineConfig


class Cosmos3ReasonerPipelineConfig(Cosmos3PipelineConfig):
    native_media_stage: ClassVar[str | None] = None
    stages: list[StageConfig] = [
        StageConfig(
            name="reasoner",
            process="reasoner",
            factory_path="sglang_omni.models.cosmos3.reasoner.create_reasoner_scheduler",
            allow_child_processes=True,
            factory=FactoryArgs(max_concurrency=8),
            gpu=0,
            terminal=True,
        )
    ]


Variants = {
    "text": Cosmos3ReasonerPipelineConfig,
    "generation": Cosmos3PipelineConfig,
}
