# SPDX-License-Identifier: Apache-2.0
"""The PersonaPlex pipeline: five stages on one GPU, the LM under SGLang."""

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
from sglang_omni.models.personaplex.hf_config import PERSONAPLEX_ARCH

MODEL_STAGES_PREFIX = "sglang_omni.models.personaplex.stages"
PREPROCESSING_STAGE = "preprocessing"
MIMI_ENCODE_STAGE = "mimi_encode"
LM_STAGE = "lm"
CODE2WAV_STAGE = "code2wav"
# A full-duplex call visits these in order, one 80 ms unit at a time.
REALTIME_STAGES = (PREPROCESSING_STAGE, MIMI_ENCODE_STAGE, LM_STAGE, CODE2WAV_STAGE)
REALTIME_MAX_CALLS = 1


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
        StageConfig(
            name=CODE2WAV_STAGE,
            process="lm",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_code2wav_executor",
            factory=FactoryArgs(num_decode_slots=8, max_graph_frames=1),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class PersonaPlexPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = PERSONAPLEX_ARCH
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        LM_STAGE: EngineStageConfig
    }

    model_path: str
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=personaplex_stages_factory)


EntryClass = PersonaPlexPipelineConfig
