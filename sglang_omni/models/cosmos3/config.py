# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 generation configuration backed by SGLang multimodal_gen."""

from collections.abc import Mapping
from typing import ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.config.schema import stage_process_name
from sglang_omni.native_generation import ROUTE_PREFIXES
from sglang_omni.platforms import current_platform

# Native keeps the DiT and the VAE resident while a card holds at least this
# much memory and streams the DiT below it, but makes that choice on CUDA only.
# Restated from native's Cosmos3 deployment thresholds, at the larger of the
# two so a card that may be too small streams.
RESIDENT_DIT_MIN_DEVICE_MEMORY_GIB = 120


def card_total_memory_bytes(device_id: int) -> int:
    """Total memory of one accelerator, or zero when the card cannot be read."""
    try:
        return int(current_platform.get_device_total_memory(device_id))
    except RuntimeError:
        return 0


# Measured on an Arc Pro B60: the understanding tower loads 16.33 GiB on a
# 23.91 GiB card, where native's own automatic fraction of 0.591 leaves nothing
# for the KV cache and the Reasoner refuses to start. A stage budget, so a
# configured one replaces it and null returns to native's fraction.
XPU_REASONER_GPU_MEMORY_FRACTION = 0.80


class Cosmos3PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Cosmos3ForConditionalGeneration"

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="generation",
            process="generation",
            factory_path="sglang_omni.native_generation.stage.create_generation_scheduler",
            allow_child_processes=True,
            factory=FactoryArgs(),
            gpu=0,
            terminal=True,
            # The stage serves native SGLang's media routes from its own process.
            http_routes=list(ROUTE_PREFIXES),
        )
    ]

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
        if stage_name != "generation" or not current_platform.is_xpu():
            return super().stage_factory_kwargs(stage_name)
        else:
            pass
        stage = self.stage_named(stage_name)
        process = self.processes.get(stage_process_name(stage))
        # Every card the stage occupies, in the order placement accounts for
        # them: the GPUs native runs on, then the replica devices that replace
        # the declared placement, then the declared one.
        if stage.runtime_gpu_ids is not None:
            devices = set(stage.runtime_gpu_ids)
        elif process is not None and process.replica_devices is not None:
            devices = set(process.replica_devices)
        elif stage.gpu is None:
            devices = set()
        else:
            devices = set(stage.gpu if isinstance(stage.gpu, list) else [stage.gpu])
        # An unreadable card counts as too small to hold the DiT.
        smallest_card_bytes = min(
            (card_total_memory_bytes(device_id) for device_id in devices), default=0
        )
        if smallest_card_bytes >= RESIDENT_DIT_MIN_DEVICE_MEMORY_GIB * (1 << 30):
            return super().stage_factory_kwargs(stage_name)
        else:
            # Native selects the streaming it needs on CUDA only, so without
            # this the stage moves the whole DiT onto the card and the first
            # request runs out of device memory.
            return {"server_args_overrides": {"layerwise_offload_components": ["dit"]}}


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
