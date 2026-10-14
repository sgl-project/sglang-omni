# SPDX-License-Identifier: Apache-2.0
"""Stage placement and deployment configuration for native duplex inference."""

from typing import ClassVar, Literal

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
THINKER_CONTEXT_LENGTH = 8192
THINKER_GPU_MEMORY_FRACTION = 0.52
TALKER_CONTEXT_LENGTH = 4096


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
            gpu_memory_fraction=THINKER_GPU_MEMORY_FRACTION,
            factory_path=f"{PKG}.create_thinker_scheduler",
            next="talker",
            # note (Junnan Li): Compiling every decode graph batch size adds minutes to startup.
            engine=EngineArgs(enable_torch_compile=False),
        ),
        EngineStageConfig(
            name="talker",
            process="talker",
            gpu=0,
            gpu_memory_fraction=0.15,
            factory_path="sglang_omni.models.minicpm_o.stages.create_sglang_session_talker_executor_from_config",
            next="speech",
            engine=EngineArgs(enable_torch_compile=False),
        ),
        StageConfig(
            name="speech",
            process="speech",
            gpu=0,
            gpu_memory_fraction=0.15,
            factory_path=f"{PKG}.create_speech_scheduler",
            terminal=True,
            # note (Junnan Li): Ragged vocoder batches fragment the allocator; the thinker and talker share memory over CUDA IPC, so only this process opts in.
            # note (Junnan Li): Padded HiFT shapes can exceed cuDNN's default 10000-plan cache at large max_sessions; the set is bounded, so the cache is unbounded.
            env={
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT": "0",
            },
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


class MiniCPMODuplexSpeech(BaseModel):
    """Vocoder settings of the speech stage; the defaults are the checkpoint's own duplex loop."""

    model_config = ConfigDict(extra="forbid")

    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    enable_dit_torch_compile: bool = False
    n_timesteps: int = Field(default=10, ge=1)


DEFAULT_SPEECH_SETTINGS = MiniCPMODuplexSpeech()


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
    speech: MiniCPMODuplexSpeech = Field(default_factory=MiniCPMODuplexSpeech)
    entry_stage: str = "perception"
    stages: list[StageConfig] = Field(default_factory=stages)

    realtime_deployment_factory: ClassVar[str] = (
        "sglang_omni.models.minicpm_o.session_adapters.build_realtime_deployment"
    )
    is_realtime_only: ClassVar[bool] = True

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, JsonValue]:
        request_slots = (
            self.max_sessions + REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
        )
        if stage_name in {"perception", "speech"}:
            kwargs: dict[str, JsonValue] = {
                "reference_audio": self.reference_audio,
                "max_open_sessions": self.max_sessions,
            }
            if stage_name == "speech":
                kwargs["max_state_bytes_per_session"] = (
                    self.speech_state_bytes_per_session
                )
                kwargs["dtype"] = self.speech.dtype
                kwargs["enable_dit_torch_compile"] = (
                    self.speech.enable_dit_torch_compile
                )
                kwargs["n_timesteps"] = self.speech.n_timesteps
            else:
                pass
            return kwargs
        elif stage_name == "thinker":
            kwargs = {"server_args_overrides": {"max_running_requests": request_slots}}
            stage = self.stage_named(stage_name)
            # note (Junnan Li): A thinker memory size the deployment writes wins; a fraction counts as written when it differs from the default.
            if (
                stage.gpu_memory_fraction == THINKER_GPU_MEMORY_FRACTION
                and stage.engine.kv_cache_bytes is None
                and stage.engine.mem_fraction_static is None
                and stage.engine.max_total_tokens is None
            ):
                kwargs["kv_cache_tokens"] = (
                    request_slots
                    * stage.engine.model_extra.get(
                        "context_length", THINKER_CONTEXT_LENGTH
                    )
                )
            else:
                pass
            return kwargs
        elif stage_name == "talker":
            server_args_overrides: dict[str, JsonValue] = {
                "max_running_requests": request_slots
            }
            engine = self.stage_named(stage_name).engine
            if (
                engine.kv_cache_bytes is None
                and engine.mem_fraction_static is None
                and engine.max_total_tokens is None
            ):
                server_args_overrides["max_total_tokens"] = (
                    request_slots * TALKER_CONTEXT_LENGTH
                )
            else:
                pass
            return {"server_args_overrides": server_args_overrides}
        else:
            return super().stage_factory_kwargs(stage_name)


EntryClass = MiniCPMODuplexPipelineConfig
