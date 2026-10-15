# SPDX-License-Identifier: Apache-2.0
"""The PersonaPlex pipeline: five stages on one GPU, the LM under SGLang."""

from collections.abc import Mapping
from typing import Annotated, ClassVar

from pydantic import Field, PositiveInt

from sglang_omni.config.schema import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)
from sglang_omni.models.personaplex.hf_config import PERSONAPLEX_ARCH

MODEL_STAGES_PREFIX = "sglang_omni.models.personaplex.stages"
PREPROCESSING_STAGE = "preprocessing"
LM_STAGE = "lm"
CODE2WAV_STAGE = "code2wav"
# The LM's concurrency when lm.engine.max_running_requests is unset.
DEFAULT_LM_MAX_RUNNING_REQUESTS = 1


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
            next="mimi_encode",
        ),
        StageConfig(
            name="mimi_encode",
            process="pipeline",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_mimi_encode_executor",
            gpu=0,
            next=LM_STAGE,
        ),
        EngineStageConfig(
            name=LM_STAGE,
            process="lm",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_lm_executor",
            factory=FactoryArgs(dtype="bfloat16"),
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


class PersonaPlexPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = PERSONAPLEX_ARCH
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        LM_STAGE: EngineStageConfig,
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


EntryClass = PersonaPlexPipelineConfig
