# SPDX-License-Identifier: Apache-2.0
"""Opt-in configuration: offline VoiceChat keeps its existing topology."""

from typing import ClassVar

from pydantic import Field, JsonValue

from sglang_omni.admission import REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
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
DEFAULT_MAX_SESSIONS = 1
THINKER_CONTEXT_LENGTH = 8192
THINKER_MEMORY_FRACTION = 0.40
TALKER_CONTEXT_LENGTH = 4096
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
                context_length = engine.model_extra.get(
                    "context_length",
                    (
                        THINKER_CONTEXT_LENGTH
                        if stage_name == "thinker"
                        else TALKER_CONTEXT_LENGTH
                    ),
                )
                server_args_overrides["max_total_tokens"] = (
                    request_slots * context_length
                )
            else:
                pass
            return {"server_args_overrides": server_args_overrides}
        else:
            return dict(super().stage_factory_kwargs(stage_name))
