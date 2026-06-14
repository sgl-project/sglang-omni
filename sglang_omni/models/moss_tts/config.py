# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS Delay."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.moss_tts"


class MossTTSPipelineConfig(PipelineConfig):
    """MOSS-TTS Delay pipeline: preprocessing -> AR engine -> vocoder."""

    architecture: ClassVar[str] = "MossTTSDelayModel"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "MossTTSDelay",
        "MossTTSDelayForConditionalGeneration",
        "MossTTSDelayWithCodec",
        "MossTTSDelayWithCodecModel",
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

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
            factory_args={"gpu_id": 0, "dtype": "bfloat16"},
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"gpu_id": 0, "dtype": "float32"},
            gpu=0,
            terminal=True,
        ),
    ]

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = MossTTSPipelineConfig
