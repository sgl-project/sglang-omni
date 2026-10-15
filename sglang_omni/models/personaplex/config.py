# SPDX-License-Identifier: Apache-2.0
"""The PersonaPlex pipeline: five stages on one GPU, the LM under SGLang."""

from typing import Annotated, ClassVar

from pydantic import Field, NonNegativeInt, PositiveInt

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


class PersonaPlexCode2WavFactoryArgs(FactoryArgs):
    num_decode_slots: NonNegativeInt = 8
    max_graph_frames: NonNegativeInt | None = None
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

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, int]:
        if stage_name == CODE2WAV_STAGE:
            return {
                "max_graph_frames": max(
                    self.stage_named(CODE2WAV_STAGE).factory.chunk_ramp_frames
                )
            }
        else:
            return {}


EntryClass = PersonaPlexPipelineConfig
