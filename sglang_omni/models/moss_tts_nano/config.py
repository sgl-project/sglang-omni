# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS-Nano."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, PipelineConfig, StageConfig


class MossTTSNanoPipelineConfig(PipelineConfig):
    """CPU-first MOSS-TTS-Nano pipeline."""

    architecture: ClassVar[str] = "MossTTSNanoForCausalLM"
    requires_model_capabilities: ClassVar[bool] = True
    max_speech_input_chars: ClassVar[int | None] = None
    additional_speech_languages: ClassVar[frozenset[str]] = frozenset(
        {
            "Arabic",
            "Czech",
            "Danish",
            "Greek",
            "Hebrew",
            "Hungarian",
            "Persian (Farsi)",
            "Polish",
            "Swedish",
            "Turkish",
        }
    )

    stages: list[StageConfig] = Field(
        default_factory=lambda: [
            StageConfig(
                name="tts",
                process="pipeline",
                factory_path=(
                    "sglang_omni.models.moss_tts_nano.stages.create_tts_executor"
                ),
                factory=FactoryArgs(
                    device="cpu", dtype="float32", attention_backend="auto"
                ),
                terminal=True,
            )
        ]
    )

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = MossTTSNanoPipelineConfig
