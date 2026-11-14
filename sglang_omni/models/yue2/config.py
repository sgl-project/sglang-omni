# SPDX-License-Identifier: Apache-2.0
"""YuE2 pipeline configuration: SGLang AR stage followed by the acoustic stage."""

import atexit
import functools
import shutil
import tempfile
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


@functools.cache
def shared_weights_directory() -> str:
    """One private directory per launching process, where the AR stage publishes its weight handles."""
    directory = tempfile.mkdtemp(prefix="sglang-omni-yue2-")
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    return directory


class Yue2AcousticFactoryArgs(FactoryArgs):
    fuse_projections: bool = False
    fused_kernels: bool = False
    vae_path: str = "m-a-p/YuE2-Vae"
    vae_core_frames: int = Field(default=512, gt=0)
    vae_halo_frames: int = Field(default=16, ge=0)
    share_ar_weights: bool = True


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

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, str]:
        del stage_name
        replicated = any(
            process.num_replicas > 1 for process in self.processes.values()
        )
        ar_stage, acoustic_stage = self.stage_named("yue2_ar"), self.stage_named(
            "yue2_acoustic"
        )
        if replicated or ar_stage.gpu != acoustic_stage.gpu:
            return {}
        else:
            return {"shared_weights_dir": shared_weights_directory()}


EntryClass = Yue2PipelineConfig
