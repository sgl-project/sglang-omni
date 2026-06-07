# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS Local (depth transformer)."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.moss_tts_local"
# Reuse the moss_tts (Delay) preprocessing factory — identical processor.
_MOSS_TTS_PKG = "sglang_omni.models.moss_tts"


class MossTTSLocalPipelineConfig(PipelineConfig):
    """MOSS-TTS Local pipeline: preprocessing -> depth-transformer AR -> vocoder."""

    architecture: ClassVar[str] = "MossTTSLocalModel"
    architecture_aliases: ClassVar[tuple[str, ...]] = ("MossTTSLocal",)

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
            factory=f"{_MOSS_TTS_PKG}.stages.create_preprocessing_executor",
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


EntryClass = MossTTSLocalPipelineConfig
