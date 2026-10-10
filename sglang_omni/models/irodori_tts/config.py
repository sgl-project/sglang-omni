# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Irodori-TTS v4.1 Small."""

from typing import ClassVar

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.platforms import current_platform

PACKAGE_PATH = "sglang_omni.models.irodori_tts"


class IrodoriTTSPipelineConfig(PipelineConfig):
    """Single-stage Irodori-TTS speech synthesis pipeline."""

    architecture: ClassVar[str] = "IrodoriTTSForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = ("IrodoriTTS",)
    requires_model_capabilities: ClassVar[bool] = True
    speech_reference_text_required: ClassVar[bool] = False
    additional_speech_languages: ClassVar[frozenset[str]] = frozenset({"Japanese"})

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="synthesis",
            process="pipeline",
            factory_path=f"{PACKAGE_PATH}.stages.create_irodori_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                model_precision="bf16",
                codec_precision="fp32",
                codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                max_seconds=30.0,
                max_batch_size=16,
                max_batch_wait_ms=8,
                max_batch_tokens=4096,
            ),
            gpu=0,
            terminal=True,
        )
    ]

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = IrodoriTTSPipelineConfig
