# SPDX-License-Identifier: Apache-2.0
"""Opt-in configuration: offline VoiceChat keeps its existing topology."""

from typing import ClassVar

from pydantic import Field

from sglang_omni.config.schema import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)

PREFIX = "sglang_omni.models.nemotron_voicechat.duplex_stages"
STAGES = ["perception", "thinker", "talker", "code2wav"]


def stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="perception",
            process="perception",
            gpu=0,
            factory_path=f"{PREFIX}.create_perception",
            next="thinker",
        ),
        EngineStageConfig(
            name="thinker",
            process="thinker",
            gpu=0,
            factory_path=f"{PREFIX}.create_thinker",
            factory=FactoryArgs(dtype="bfloat16"),
            engine=EngineArgs(mem_fraction_static=0.40),
            next="talker",
        ),
        EngineStageConfig(
            name="talker",
            process="talker",
            gpu=0,
            factory_path=f"{PREFIX}.create_talker",
            factory=FactoryArgs(dtype="bfloat16"),
            engine=EngineArgs(mem_fraction_static=0.25),
            next="code2wav",
        ),
        StageConfig(
            name="code2wav",
            process="codec",
            gpu=0,
            factory_path=f"{PREFIX}.create_codec",
            terminal=True,
        ),
    ]


class NemotronVoiceChatDuplexPipelineConfig(PipelineConfig):
    realtime_deployment_factory: ClassVar[str] = (
        "sglang_omni.models.nemotron_voicechat.realtime.deployment"
    )
    model_path: str
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "thinker": EngineStageConfig,
        "talker": EngineStageConfig,
    }
    stages: list[StageConfig] = Field(default_factory=stages)
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
