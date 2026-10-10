# SPDX-License-Identifier: Apache-2.0
"""Opt-in configuration: offline VoiceChat keeps its existing topology."""

from typing import ClassVar

from pydantic import Field

from sglang_omni.config.schema import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    GraphConfig,
    GraphEdgeConfig,
    GraphNodeConfig,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)
from sglang_omni.models.nemotron_voicechat.tools import RANDOM_NUMBER_TOOL

PREFIX = "sglang_omni.models.nemotron_voicechat.duplex_stages"
STAGES = ["perception", "thinker", "talker", "code2wav"]
TOOL_STAGE = "tool"


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


def tool_stages() -> list[StageConfig]:
    """The duplex chain with tool prompting, plus a CPU tool node."""
    duplex_stages = stages()
    for stage in duplex_stages:
        if stage.name == "thinker":
            stage.factory = FactoryArgs(
                dtype="bfloat16", tool_definitions=[RANDOM_NUMBER_TOOL]
            )
        else:
            pass
    return [
        *duplex_stages,
        StageConfig(
            name=TOOL_STAGE,
            process=TOOL_STAGE,
            factory_path="sglang_omni.scheduling.tool_session.create_tool_scheduler",
            factory=FactoryArgs(
                tools={
                    "generate_random_number": "sglang_omni.models.nemotron_voicechat.tools.generate_random_number"
                }
            ),
            terminal=True,
        ),
    ]


def tool_graph() -> GraphConfig:
    return GraphConfig(
        nodes={
            "voicechat": GraphNodeConfig(stages=STAGES),
            "tool": GraphNodeConfig(stages=[TOOL_STAGE]),
        },
        inputs={"audio": ["voicechat"]},
        output="voicechat",
        edges=[
            GraphEdgeConfig(source="voicechat", target="tool", modality="tool_call"),
            GraphEdgeConfig(
                source="tool", target="voicechat", modality="tool_response"
            ),
        ],
    )


class NemotronVoiceChatToolPipelineConfig(NemotronVoiceChatDuplexPipelineConfig):
    """VoiceChat and an external tool node as one two-node session graph."""

    stages: list[StageConfig] = Field(default_factory=tool_stages)
    graph: GraphConfig | None = Field(default_factory=tool_graph)
