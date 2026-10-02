# SPDX-License-Identifier: Apache-2.0
"""EasyMagpie TTS pipeline configuration.

Three-stage pipeline: preprocessing -> tts_engine -> vocoder.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.easymagpie_tts"


def stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            gpu=0,
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(dtype="float16"),
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            gpu=0,
            terminal=True,
        ),
    ]


class EasyMagpieTTSPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "EasyMagpieTTSForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = Field(default_factory=stages)


EntryClass = EasyMagpieTTSPipelineConfig
