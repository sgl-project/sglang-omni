# SPDX-License-Identifier: Apache-2.0
"""The PersonaPlex pipelines on one GPU, the LM under SGLang.

The default pipeline answers a whole recording; the session variant holds
full-duplex sessions over /v1/realtime.
"""

from collections.abc import Mapping
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from sglang_omni.admission import REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
from sglang_omni.config.schema import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)
from sglang_omni.models.personaplex.architecture import (
    DEFAULT_AUDIO_TEMPERATURE,
    DEFAULT_AUDIO_TOP_K,
)
from sglang_omni.models.personaplex.hf_config import PERSONAPLEX_ARCH

MODEL_STAGES_PREFIX = "sglang_omni.models.personaplex.stages"
PREPROCESSING_STAGE = "preprocessing"
MIMI_ENCODE_STAGE = "mimi_encode"
LM_STAGE = "lm"
CODE2WAV_STAGE = "code2wav"
# The LM's concurrency when lm.engine.max_running_requests is unset.
DEFAULT_LM_MAX_RUNNING_REQUESTS = 1


class PersonaPlexDepformerFactoryArgs(FactoryArgs):
    depformer_cuda_graph_batch_sizes: list[PositiveInt] = Field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32]
    )


class PersonaPlexLMFactoryArgs(PersonaPlexDepformerFactoryArgs):
    enable_async_decode: bool = False


class PersonaPlexLMStageConfig(EngineStageConfig):
    factory: PersonaPlexLMFactoryArgs = Field(default_factory=PersonaPlexLMFactoryArgs)


class PersonaPlexSessionLMStageConfig(EngineStageConfig):
    factory: PersonaPlexDepformerFactoryArgs = Field(
        default_factory=PersonaPlexDepformerFactoryArgs
    )


class PersonaPlexCode2WavFactoryArgs(FactoryArgs):
    """Streaming decode: num_decode_slots streams resident at once, up to
    max_batch_size of them per step, device graphs for chunks of up to
    max_graph_frames frames (0 turns the graphs off).

    num_decode_slots is derived from the LM's concurrency unless set.
    """

    num_decode_slots: int | None = Field(default=None, ge=1)
    max_batch_size: int = Field(default=8, ge=1)
    max_graph_frames: int | None = Field(default=None, ge=0)
    chunk_ramp_frames: list[Annotated[PositiveInt, Field(strict=True)]] = Field(
        default_factory=lambda: [1, 4], min_length=1
    )


class PersonaPlexCode2WavStageConfig(StageConfig):
    factory: PersonaPlexCode2WavFactoryArgs = Field(
        default_factory=PersonaPlexCode2WavFactoryArgs
    )


def personaplex_stages_factory() -> list[StageConfig]:
    return [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_preprocessing_executor",
            next=MIMI_ENCODE_STAGE,
        ),
        StageConfig(
            name=MIMI_ENCODE_STAGE,
            process="pipeline",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_mimi_encode_executor",
            gpu=0,
            next=LM_STAGE,
        ),
        PersonaPlexLMStageConfig(
            name=LM_STAGE,
            process="lm",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_lm_executor",
            factory=PersonaPlexLMFactoryArgs(dtype="bfloat16"),
            gpu=0,
            engine=EngineArgs(mem_fraction_static=0.3),
            next=["decode", CODE2WAV_STAGE],
            stream_to=[CODE2WAV_STAGE],
        ),
        StageConfig(
            name="decode",
            process="pipeline",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_decode_executor",
            terminal=True,
        ),
        PersonaPlexCode2WavStageConfig(
            name=CODE2WAV_STAGE,
            process="lm",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_code2wav_executor",
            factory=PersonaPlexCode2WavFactoryArgs(),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


def personaplex_session_stages_factory() -> list[StageConfig]:
    """One linear route, the only shape a session unit takes."""
    return [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_session_preprocessing_executor",
            next=MIMI_ENCODE_STAGE,
        ),
        StageConfig(
            name=MIMI_ENCODE_STAGE,
            process="pipeline",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_session_mimi_encode_executor",
            gpu=0,
            next=LM_STAGE,
        ),
        PersonaPlexSessionLMStageConfig(
            name=LM_STAGE,
            process="lm",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_session_lm_executor",
            factory=PersonaPlexDepformerFactoryArgs(dtype="bfloat16"),
            gpu=0,
            engine=EngineArgs(mem_fraction_static=0.3),
            next=CODE2WAV_STAGE,
        ),
        StageConfig(
            name=CODE2WAV_STAGE,
            process="lm",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_session_code2wav_executor",
            gpu=0,
            terminal=True,
        ),
    ]


class PersonaPlexPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = PERSONAPLEX_ARCH
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        LM_STAGE: PersonaPlexLMStageConfig,
        CODE2WAV_STAGE: PersonaPlexCode2WavStageConfig,
    }

    model_path: str
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=personaplex_stages_factory)

    def model_post_init(self, __context: object = None) -> None:
        super().model_post_init(__context)
        # Note (nightdecode): An unset num_decode_slots stays unset and is
        # derived at launch in stage_factory_kwargs; writing the derivation
        # back would make it look pinned on the next merge. Only a pinned
        # value is checked, on every rebuild.
        num_decode_slots = self.pinned_num_decode_slots()
        lm_max_running = self.lm_max_running_requests()
        if num_decode_slots is not None and num_decode_slots < lm_max_running:
            raise ValueError(
                f"{CODE2WAV_STAGE}.factory.num_decode_slots={num_decode_slots} is "
                f"below the LM's max_running_requests={lm_max_running}: every "
                "running LM request needs a code2wav decode slot of its own. "
                "Omit num_decode_slots to derive it from the LM."
            )
        else:
            pass

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        if stage_name == CODE2WAV_STAGE:
            # Note (nightdecode): Match LM concurrency. Streams whose lifetimes
            # overlap while tail audio drains wait for a decode slot.
            return {
                "num_decode_slots": self.lm_max_running_requests(),
                "max_graph_frames": max(
                    self.stage_named(CODE2WAV_STAGE).factory.chunk_ramp_frames
                ),
            }
        else:
            return super().stage_factory_kwargs(stage_name)

    def pinned_num_decode_slots(self) -> int | None:
        stage = self.stage_or_none(CODE2WAV_STAGE)
        value = (
            getattr(stage.factory, "num_decode_slots", None)
            if stage is not None
            else None
        )
        return int(value) if value is not None else None

    def lm_max_running_requests(self) -> int:
        """The LM's concurrency: its engine setting, else the stage default."""
        stage = self.stage_or_none(LM_STAGE)
        engine = getattr(stage, "engine", None)
        value = engine.max_running_requests if engine is not None else None
        return int(value) if value is not None else DEFAULT_LM_MAX_RUNNING_REQUESTS

    def stage_or_none(self, stage_name: str) -> StageConfig | None:
        return next((stage for stage in self.stages if stage.name == stage_name), None)


class PersonaPlexSessionSampling(BaseModel):
    """Deployment sampling for sessions; unset fields keep the offline defaults,
    and a session may override any field."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0)
    top_k: int | None = Field(default=None, ge=-1)
    top_p: float | None = Field(default=None, gt=0, le=1)
    min_p: float | None = Field(default=None, ge=0, le=1)
    repetition_penalty: float | None = Field(default=None, gt=0)
    audio_temperature: float | None = Field(default=None, ge=0)
    audio_top_k: int | None = Field(default=None, ge=-1)
    seed: int | None = None


class PersonaPlexSessionPipelineConfig(PersonaPlexPipelineConfig):
    realtime_deployment_factory: ClassVar[str | None] = (
        "sglang_omni.models.personaplex.realtime.create_realtime_deployment"
    )
    is_realtime_only: ClassVar[bool] = True
    # Note (wilsonzheng0327): Async LM decode and offline code2wav factory fields
    # have no session counterpart.
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        LM_STAGE: PersonaPlexSessionLMStageConfig
    }

    max_sessions: int = Field(default=1, ge=1)
    # Note (Junnan Li): None keeps the offline default voice and role prompt.
    voice: str | None = None
    instructions: str | None = None
    sampling: PersonaPlexSessionSampling = Field(
        default_factory=PersonaPlexSessionSampling
    )
    stages: list[StageConfig] = Field(
        default_factory=personaplex_session_stages_factory
    )

    def model_post_init(self, __context: object = None) -> None:
        super().model_post_init(__context)
        lm_max_running = self.lm_max_running_requests()
        if lm_max_running < self.lm_request_slots():
            raise ValueError(
                f"{LM_STAGE}.engine.max_running_requests={lm_max_running} is below "
                f"max_sessions + {REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV}"
                f"={self.lm_request_slots()}: the session bridge admits a session's "
                "first unit only while a slot stays free for sessions holding KV."
            )
        else:
            pass

    def lm_request_slots(self) -> int:
        return self.max_sessions + REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV

    def lm_max_running_requests(self) -> int:
        """The LM's concurrency: its engine setting, else the session slots."""
        engine = self.stage_named(LM_STAGE).engine
        value = engine.max_running_requests
        return int(value) if value is not None else self.lm_request_slots()

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        if stage_name == LM_STAGE:
            return {
                "audio_temperature": (
                    self.sampling.audio_temperature
                    if self.sampling.audio_temperature is not None
                    else DEFAULT_AUDIO_TEMPERATURE
                ),
                "audio_top_k": (
                    self.sampling.audio_top_k
                    if self.sampling.audio_top_k is not None
                    else DEFAULT_AUDIO_TOP_K
                ),
                "server_args_overrides": {
                    "max_running_requests": self.lm_request_slots()
                },
            }
        else:
            return {"max_open_sessions": self.max_sessions}


EntryClass = PersonaPlexPipelineConfig

Variants = {
    "offline": PersonaPlexPipelineConfig,
    "session": PersonaPlexSessionPipelineConfig,
}
