# SPDX-License-Identifier: Apache-2.0
"""Single-worker YuE2 pipeline configuration."""

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig


class Yue2FactoryArgs(FactoryArgs):
    ar_cuda_graph: bool = False
    nar_cuda_graph: bool = False
    sampling_cuda_graph: bool = False
    fuse_projections: bool = False
    fused_kernels: bool = False
    max_batch_size: int = Field(default=1, gt=0)
    max_batch_wait_ms: int = Field(default=25, ge=0)
    batch_graph_cache_size: int = Field(default=2, gt=0)
    vae_path: str = "m-a-p/YuE2-Vae"
    vae_core_frames: int = Field(default=1024, gt=0)
    vae_halo_frames: int = Field(default=16, ge=0)


class Yue2StageConfig(StageConfig):
    factory: Yue2FactoryArgs = Field(default_factory=Yue2FactoryArgs)


class Yue2PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "YuE2ForCausalLM"
    requires_model_capabilities: ClassVar[bool] = True
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "synthesis": Yue2StageConfig,
    }
    stages: list[StageConfig] = Field(
        default_factory=lambda: [
            Yue2StageConfig(
                name="synthesis",
                process="synthesis",
                factory_path="sglang_omni.models.yue2.stages.create_synthesis_executor",
                gpu=0,
                terminal=True,
            )
        ]
    )


EntryClass = Yue2PipelineConfig
