# SPDX-License-Identifier: Apache-2.0
"""Opt-in configuration: offline VoiceChat keeps its existing topology."""

from collections.abc import Mapping
from typing import ClassVar

from pydantic import Field, JsonValue, model_validator

from sglang_omni.admission import REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
from sglang_omni.config.schema import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    SessionControlEdgeConfig,
    SessionDataEdgeConfig,
    SessionNodeConfig,
    StageConfig,
)
from sglang_omni.config.tools import (
    ToolRegistry,
    tool_definitions,
    tool_implementations,
)
from sglang_omni.models.nemotron_voicechat.payload_types import VOICECHAT_FORMAT
from sglang_omni.proto.session import GRAPH_INPUT
from sglang_omni.proto.streams import TOOL_CALL, TOOL_RESPONSE

PREFIX = "sglang_omni.models.nemotron_voicechat.duplex_stages"
STAGES = ["perception", "thinker", "talker", "code2wav"]
TOOL_STAGE = "tool"
VOICECHAT_NODE = "voicechat"
DEFAULT_MAX_SESSIONS = 1
CONTEXT_LENGTH = 6144
THINKER_MEMORY_FRACTION = 0.40
TALKER_MEMORY_FRACTION = 0.25


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
            engine=EngineArgs(mem_fraction_static=THINKER_MEMORY_FRACTION),
            next="talker",
        ),
        EngineStageConfig(
            name="talker",
            process="talker",
            gpu=0,
            factory_path=f"{PREFIX}.create_talker",
            factory=FactoryArgs(dtype="bfloat16"),
            engine=EngineArgs(mem_fraction_static=TALKER_MEMORY_FRACTION),
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
    max_sessions: int = Field(default=DEFAULT_MAX_SESSIONS, ge=1)
    stages: list[StageConfig] = Field(default_factory=stages)
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, JsonValue]:
        request_slots = (
            self.max_sessions + REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
        )
        if stage_name in {"perception", "code2wav"}:
            return {"max_open_sessions": self.max_sessions}
        elif stage_name in {"thinker", "talker"}:
            engine = self.stage_named(stage_name).engine
            server_args_overrides: dict[str, JsonValue] = {
                "max_running_requests": request_slots
            }
            default_memory_fraction = (
                THINKER_MEMORY_FRACTION
                if stage_name == "thinker"
                else TALKER_MEMORY_FRACTION
            )
            if (
                engine.kv_cache_bytes is None
                and engine.max_total_tokens is None
                and engine.mem_fraction_static == default_memory_fraction
            ):
                server_args_overrides["max_total_tokens"] = (
                    request_slots * CONTEXT_LENGTH
                )
            else:
                pass
            return {"server_args_overrides": server_args_overrides}
        else:
            return dict(super().stage_factory_kwargs(stage_name))


def tool_stages() -> list[StageConfig]:
    """The duplex chain with a CPU tool stage configured at deployment time."""
    return [
        *stages(),
        StageConfig(
            name=TOOL_STAGE,
            process=TOOL_STAGE,
            factory_path="sglang_omni.models.tools.session.create_tool_scheduler",
            terminal=True,
        ),
    ]


def tool_session_nodes() -> dict[str, SessionNodeConfig]:
    return {
        VOICECHAT_NODE: SessionNodeConfig(
            stages=STAGES,
            inputs=[
                # note (Junnan Li): A real-time model has no use for frames that waited behind newer ones.
                SessionDataEdgeConfig(source=GRAPH_INPUT, overflow="drop_oldest"),
                # note (Junnan Li): A response queued behind the user's audio reaches the model late enough that it calls the tool again.
                SessionControlEdgeConfig(
                    kind="control",
                    source=TOOL_STAGE,
                    event=TOOL_RESPONSE,
                ),
            ],
        ),
        TOOL_STAGE: SessionNodeConfig(
            stages=[TOOL_STAGE], inputs=[f"{VOICECHAT_NODE}:{TOOL_CALL}"]
        ),
    }


class NemotronVoiceChatToolPipelineConfig(NemotronVoiceChatDuplexPipelineConfig):
    """VoiceChat and a tool stage as one two-node Session graph.

    The thinker's function channel writes a call; the codec stage emits it as
    a tool_call chunk, the tool stage answers with a tool_response chunk on a
    control edge, past the queued audio, and the perception stage hands it to
    the thinker's function channel with the next audio frame.
    """

    stages: list[StageConfig] = Field(default_factory=tool_stages)
    session_nodes: dict[str, SessionNodeConfig] = Field(
        default_factory=tool_session_nodes
    )
    session_publish: dict[str, list[str]] = Field(
        default_factory=lambda: {VOICECHAT_NODE: [VOICECHAT_FORMAT]}
    )

    tools: ToolRegistry = Field(default_factory=dict)
    tool_names: list[str] | None = None

    @model_validator(mode="after")
    def check_tool_registry(self) -> "NemotronVoiceChatToolPipelineConfig":
        for name in self.tools if self.tool_names is None else self.tool_names:
            if name not in self.tools:
                raise ValueError(f"unknown tool {name!r}")
            else:
                pass
        for stage, argument in (("thinker", "tool_definitions"), (TOOL_STAGE, "tools")):
            if argument in self.stage_named(stage).factory.model_dump():
                raise ValueError(
                    f"declare tools in the pipeline tools registry, not {stage}.factory.{argument}"
                )
            else:
                pass
        return self

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        kwargs = dict(super().stage_factory_kwargs(stage_name))
        if stage_name == "thinker":
            kwargs["tool_definitions"] = tool_definitions(self.tools, self.tool_names)
        elif stage_name == TOOL_STAGE:
            kwargs["tools"] = tool_implementations(self.tools, self.tool_names)
        else:
            pass
        return kwargs
