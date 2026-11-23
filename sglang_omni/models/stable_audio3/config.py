# SPDX-License-Identifier: Apache-2.0
"""Stable Audio 3 pipeline: request validation, then batched generation."""

from typing import ClassVar

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.models.stable_audio3.request_builders import GENERATION_STAGE

PACKAGE = "sglang_omni.models.stable_audio3"


class StableAudio3PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "StableAudio3ForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True
    speech_request_validator_factory: ClassVar[str | None] = (
        f"{PACKAGE}.speech_validation.StableAudio3SpeechRequestValidator"
    )
    supported_speech_fields: ClassVar[frozenset[str] | None] = frozenset(
        {
            "model",
            "input",
            "voice",
            "response_format",
            "sample_rate",
            "stream",
            "seed",
            "stage_params",
        }
    )
    stages: list[StageConfig] = Field(
        default_factory=lambda: [
            StageConfig(
                name="preprocessing",
                process="stable_audio3",
                factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
                next=GENERATION_STAGE,
            ),
            StageConfig(
                name=GENERATION_STAGE,
                process="stable_audio3",
                factory_path=f"{PACKAGE}.stages.create_generation_executor",
                factory=FactoryArgs(max_batch_size=8, max_batch_wait_ms=5),
                gpu=0,
                terminal=True,
            ),
        ]
    )


EntryClass = StableAudio3PipelineConfig
