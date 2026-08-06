# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Fun-CosyVoice3."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.fun_cosyvoice3"


class FunCosyVoice3PipelineConfig(PipelineConfig):
    """3-stage Fun-CosyVoice3 pipeline: preprocessing -> tts_engine -> vocoder."""

    architecture: ClassVar[str] = "FunCosyVoice3SGLangModel"
    # This checkpoint has no built-in speaker presets, so public requests need
    # one reference clip for speaker conditioning.
    required_speech_reference_count: ClassVar[int | None] = 1
    speech_reference_text_excludes_instructions: ClassVar[bool] = True

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "tts_engine"}

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def process_safe_edges(cls) -> frozenset[tuple[str, str]]:
        # note: preprocessing -> tts_engine is excluded because preprocessing
        # stores prepared requests in module-level registries that the AR
        # engine builder reads in-process. The vocoder loads its own flow+hift
        # models and reads audio_codes from the payload.
        return frozenset({("tts_engine", "vocoder")})

    @classmethod
    def process_edge_resources(
        cls,
    ) -> dict[tuple[str, str], dict[str, float]]:
        return {
            ("tts_engine", "vocoder"): {
                "tts_engine": 0.85,
                "vocoder": 0.10,
            }
        }

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
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = FunCosyVoice3PipelineConfig
