# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 generation configuration backed by SGLang multimodal_gen."""

from collections.abc import Mapping
from typing import ClassVar

from sglang_omni.config import EngineStageConfig, FactoryArgs, StageConfig
from sglang_omni.native_generation.config import NativeGenerationPipelineConfig
from sglang_omni.platforms import current_platform

# Measured on an Arc Pro B60: the understanding tower loads 16.33 GiB on a
# 23.91 GiB card, where native's own automatic fraction of 0.591 leaves nothing
# for the KV cache and the Reasoner refuses to start. A stage budget, so a
# configured one replaces it and null returns to native's fraction.
XPU_REASONER_GPU_MEMORY_FRACTION = 0.80


class Cosmos3PipelineConfig(NativeGenerationPipelineConfig):
    architecture: ClassVar[str] = "Cosmos3ForConditionalGeneration"

    def model_post_init(self, __context: object = None) -> None:
        reasoner = next(
            (stage for stage in self.stages if stage.name == "reasoner"), None
        )
        # A configuration rebuilt from written stages keeps what it wrote, null included.
        if (
            reasoner is not None
            and "stages" not in self.model_fields_set
            and current_platform.is_xpu()
        ):
            reasoner.gpu_memory_fraction = XPU_REASONER_GPU_MEMORY_FRACTION
        else:
            pass
        super().model_post_init(__context)

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        if stage_name == "generation" and current_platform.is_xpu():
            # Native selects this streaming on CUDA only, so without it the
            # stage moves the whole DiT onto the card and the first request
            # runs out of device memory. A card that holds the DiT resident
            # clears the selection with a null in the same config entry.
            return {"server_args_overrides": {"layerwise_offload_components": ["dit"]}}
        else:
            return super().stage_factory_kwargs(stage_name)


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
