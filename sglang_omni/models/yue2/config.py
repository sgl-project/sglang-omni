# SPDX-License-Identifier: Apache-2.0
"""YuE2 pipeline configuration: SGLang AR stage followed by the acoustic stage."""

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)

PACKAGE = "sglang_omni.models.yue2"


class Yue2AcousticFactoryArgs(FactoryArgs):
    nar_cuda_graph: bool = False
    fuse_projections: bool = False
    fused_kernels: bool = False
    vae_path: str = "m-a-p/YuE2-Vae"
    vae_core_frames: int = Field(default=512, gt=0)
    vae_halo_frames: int = Field(default=16, ge=0)


class Yue2AcousticStageConfig(StageConfig):
    factory: Yue2AcousticFactoryArgs = Field(default_factory=Yue2AcousticFactoryArgs)


class Yue2PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "YuE2ForCausalLM"
    requires_model_capabilities: ClassVar[bool] = True
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "yue2_ar": EngineStageConfig,
        "yue2_acoustic": Yue2AcousticStageConfig,
    }
    stages: list[StageConfig] = Field(
        default_factory=lambda: [
            EngineStageConfig(
                name="yue2_ar",
                process="yue2_ar",
                factory_path=f"{PACKAGE}.stages.create_ar_executor",
                factory=FactoryArgs(max_concurrency=16),
                gpu=0,
                next="yue2_acoustic",
            ),
            Yue2AcousticStageConfig(
                name="yue2_acoustic",
                process="yue2_acoustic",
                factory_path=f"{PACKAGE}.stages.create_acoustic_executor",
                gpu=0,
                terminal=True,
            ),
        ]
    )
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )


EntryClass = Yue2PipelineConfig
