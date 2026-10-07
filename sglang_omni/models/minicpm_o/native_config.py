# SPDX-License-Identifier: Apache-2.0
"""Stage placement and deployment configuration for native duplex inference."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from sglang_omni.admission import REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    PipelineConfig,
    StageConfig,
)

PKG = "sglang_omni.models.minicpm_o.native_stages"
DEFAULT_MAX_SESSIONS = 2
DEFAULT_SPEECH_STATE_BYTES_PER_SESSION = 2 << 30


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
            factory_path="sglang_omni.models.minicpm_o.stages.create_sglang_session_talker_executor_from_config",
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


class MiniCPMODuplexSampling(BaseModel):
    """Deployment defaults for the thinker duplex sampler; a session may override them."""

    model_config = ConfigDict(extra="forbid")

    greedy: bool = False
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_k: int = Field(default=20, ge=-1)
    top_p: float = Field(default=0.8, gt=0, le=1)
    repetition_penalty: float = Field(default=1.05, ge=1)
    listen_prob_scale: float = Field(default=1.0, ge=0)
    force_listen_count: int = Field(default=3, ge=0)
    max_new_tokens_per_unit: int = Field(default=20, ge=1)
    repetition_window_size: int = Field(default=512, ge=1)
    talker_temperature: float = Field(default=0.8, ge=0, le=2)
    talker_repetition_penalty: float = Field(default=1.05, ge=1)


class MiniCPMODuplexVision(BaseModel):
    """Per-unit image limits; each frame costs one overview tile plus its slices."""

    model_config = ConfigDict(extra="forbid")

    max_frames_per_unit: int = Field(default=4, ge=1)
    max_tiles_per_unit: int = Field(default=10, ge=1)
    max_slice_nums: int = Field(default=1, ge=1)
    max_slice_nums_limit: int = Field(default=9, ge=1, le=9)

    @model_validator(mode="after")
    def check_limits(self) -> "MiniCPMODuplexVision":
        if self.max_slice_nums > self.max_slice_nums_limit:
            raise ValueError("max_slice_nums exceeds max_slice_nums_limit")
        elif self.max_slice_nums_limit > 1 and (
            self.max_tiles_per_unit < self.max_slice_nums_limit + 1
        ):
            raise ValueError("max_tiles_per_unit cannot fit one frame at the limit")
        else:
            return self


class MiniCPMODuplexPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "MiniCPMO"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "thinker": EngineStageConfig,
        "talker": EngineStageConfig,
    }
    model_path: str
    reference_audio: str | None = None
    max_sessions: int = Field(default=DEFAULT_MAX_SESSIONS, ge=1)
    speech_state_bytes_per_session: int = Field(
        default=DEFAULT_SPEECH_STATE_BYTES_PER_SESSION, ge=1
    )
    sampling: MiniCPMODuplexSampling = Field(default_factory=MiniCPMODuplexSampling)
    vision: MiniCPMODuplexVision = Field(default_factory=MiniCPMODuplexVision)
    entry_stage: str = "perception"
    stages: list[StageConfig] = Field(default_factory=stages)

    realtime_deployment_factory: ClassVar[str] = (
        "sglang_omni.models.minicpm_o.session_adapters.build_realtime_deployment"
    )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, JsonValue]:
        if stage_name in {"perception", "speech"}:
            kwargs: dict[str, JsonValue] = {
                "reference_audio": self.reference_audio,
                "max_open_sessions": self.max_sessions,
            }
            if stage_name == "speech":
                kwargs["max_state_bytes_per_session"] = (
                    self.speech_state_bytes_per_session
                )
            else:
                pass
            return kwargs
        elif stage_name in {"thinker", "talker"}:
            return {
                "server_args_overrides": {
                    "max_running_requests": self.max_sessions
                    + REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
                }
            }
        else:
            return super().stage_factory_kwargs(stage_name)


EntryClass = MiniCPMODuplexPipelineConfig
