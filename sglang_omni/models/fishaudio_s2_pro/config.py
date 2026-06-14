# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for FishAudio S2-Pro TTS."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.fishaudio_s2_pro"


class S2ProPipelineConfig(PipelineConfig):
    """3-stage TTS pipeline: preprocessing → tts_engine → vocoder."""

    architecture: ClassVar[str] = "FishQwen3OmniForCausalLM"

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"device": "cuda:0", "max_new_tokens": 2048},
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = S2ProPipelineConfig
