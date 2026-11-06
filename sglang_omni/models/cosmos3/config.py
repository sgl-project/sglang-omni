# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 generation configuration backed by SGLang multimodal_gen."""

from typing import ClassVar

from sglang_omni.config import EngineStageConfig, FactoryArgs, StageConfig
from sglang_omni.native_generation.config import NativeGenerationPipelineConfig


class Cosmos3PipelineConfig(NativeGenerationPipelineConfig):
    architecture: ClassVar[str] = "Cosmos3ForConditionalGeneration"


EntryClass = Cosmos3PipelineConfig


class Cosmos3ReasonerPipelineConfig(Cosmos3PipelineConfig):
    # engine.* keys and --mem-fraction-static reach the native SRT ServerArgs.
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "reasoner": EngineStageConfig
    }
    stages: list[StageConfig] = [
        EngineStageConfig(
            name="reasoner",
            process="reasoner",
            factory_path=(
                "sglang_omni.scheduling.native_chat_scheduler."
                "create_native_chat_scheduler"
            ),
            allow_child_processes=True,
            factory=FactoryArgs(stage_name="reasoner"),
            gpu=0,
            terminal=True,
        )
    ]


Variants = {
    "text": Cosmos3ReasonerPipelineConfig,
    "generation": Cosmos3PipelineConfig,
    # Imported only when selected, so other models load none of its modules.
    "umm": "sglang_omni.models.cosmos3.umm_config:Cosmos3UMMPipelineConfig",
}
