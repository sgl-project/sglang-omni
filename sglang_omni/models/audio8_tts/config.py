# SPDX-License-Identifier: Apache-2.0
"""Audio8 TTS pipeline with a dense Slow AR and a whole-utterance codec."""

from typing import ClassVar

from pydantic import Field

from sglang_omni.config.schema import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)

ARCHITECTURE = "ArkttsModel"
PACKAGE = "sglang_omni.models.audio8_tts"
TTS_ENGINE_STAGE = "tts_engine"


class Audio8TTSPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = ARCHITECTURE
    requires_model_capabilities: ClassVar[bool] = True
    speech_reference_text_required: ClassVar[bool] = True
    speech_request_validator_factory: ClassVar[str] = (
        f"{PACKAGE}.speech_request.Audio8SpeechRequestValidator"
    )
    supported_speech_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "model",
            "input",
            "voice",
            "response_format",
            "sample_rate",
            "speed",
            "stream",
            "stream_format",
            "references",
            "ref_audio",
            "ref_text",
            "max_new_tokens",
            "temperature",
            "top_k",
            "top_p",
            "seed",
        }
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        TTS_ENGINE_STAGE: EngineStageConfig,
    }

    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            gpu=0,
            factory_path=f"{PACKAGE}.frontend.create_preprocessing_executor",
            next=TTS_ENGINE_STAGE,
        ),
        EngineStageConfig(
            name=TTS_ENGINE_STAGE,
            process="pipeline",
            gpu=0,
            factory_path=f"{PACKAGE}.engine.create_sglang_tts_engine_executor",
            factory=FactoryArgs(max_new_tokens=2048),
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            gpu=0,
            factory_path=f"{PACKAGE}.frontend.create_vocoder_executor",
            terminal=True,
        ),
    ]


EntryClass = Audio8TTSPipelineConfig
