# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Audar-TTS-V1 Turbo."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.audar_tts"


def _stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next="reference_encoder",
        ),
        StageConfig(
            name="reference_encoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_reference_encoder_executor",
            gpu=0,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_tts_engine_executor",
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            gpu=0,
            terminal=True,
        ),
    ]


class AudarTTSPipelineConfig(PipelineConfig):
    """Audar-TTS-V1 Turbo GGUF pipeline."""

    architecture: ClassVar[str] = "AudarTTSForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = ("AudarTTS",)
    requires_model_capabilities: ClassVar[bool] = True
    required_speech_reference_count: ClassVar[int | None] = 1
    speech_reference_text_required: ClassVar[bool] = True
    additional_speech_languages: ClassVar[frozenset[str]] = frozenset({"Arabic"})

    model_path: str
    stages: list[StageConfig] = Field(default_factory=_stages)

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = AudarTTSPipelineConfig
