# SPDX-License-Identifier: Apache-2.0
"""Stage placement and deployment configuration for native duplex inference."""

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    PipelineConfig,
    StageConfig,
)

PKG = "sglang_omni.models.minicpm_o.native_stages"


def stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="perception",
            process="perception",
            gpu=0,
            gpu_memory_fraction=0.12,
            factory_path=f"{PKG}.create_perception_scheduler",
            next="thinker",
        ),
        EngineStageConfig(
            name="thinker",
            process="thinker",
            gpu=0,
            gpu_memory_fraction=0.52,
            factory_path=f"{PKG}.create_thinker_scheduler",
            next="talker",
            engine=EngineArgs(disable_cuda_graph=True),
        ),
        EngineStageConfig(
            name="talker",
            process="talker",
            gpu=0,
            gpu_memory_fraction=0.15,
            factory_path=f"{PKG}.create_talker_scheduler",
            next="speech",
            engine=EngineArgs(disable_cuda_graph=True),
        ),
        StageConfig(
            name="speech",
            process="speech",
            gpu=0,
            gpu_memory_fraction=0.15,
            factory_path=f"{PKG}.create_speech_scheduler",
            terminal=True,
        ),
    ]


class MiniCPMODuplexPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "MiniCPMO"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "thinker": EngineStageConfig,
        "talker": EngineStageConfig,
    }
    model_path: str
    reference_audio: str | None = None
    entry_stage: str = "perception"
    stages: list[StageConfig] = Field(default_factory=stages)

    realtime_deployment_factory: ClassVar[str] = (
        "sglang_omni.models.minicpm_o.session_adapters.build_realtime_deployment"
    )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, str | None]:
        if stage_name in {"perception", "speech"}:
            return {"reference_audio": self.reference_audio}
        else:
            return super().stage_factory_kwargs(stage_name)


EntryClass = MiniCPMODuplexPipelineConfig
