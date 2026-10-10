# SPDX-License-Identifier: Apache-2.0
"""The PersonaPlex pipeline: five stages on one GPU, the LM under SGLang."""

from typing import ClassVar

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


class PersonaPlexLMFactoryArgs(FactoryArgs):
    depformer_cuda_graph_batch_sizes: list[PositiveInt] | None = None


class PersonaPlexLMStageConfig(EngineStageConfig):
    factory: PersonaPlexLMFactoryArgs = Field(default_factory=PersonaPlexLMFactoryArgs)


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
        PersonaPlexLMStageConfig(
            name=LM_STAGE,
            process="lm",
            factory_path=f"{MODEL_STAGES_PREFIX}.create_lm_executor",
            factory=PersonaPlexLMFactoryArgs(
                dtype="bfloat16", depformer_cuda_graph_batch_sizes=[1, 2, 4, 8]
            ),
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
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class PersonaPlexPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = PERSONAPLEX_ARCH
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        LM_STAGE: PersonaPlexLMStageConfig
    }

    model_path: str
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=personaplex_stages_factory)


EntryClass = PersonaPlexPipelineConfig
