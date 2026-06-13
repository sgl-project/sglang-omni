# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Zyphra ZONOS2 TTS."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.zonos2"


class ZONOS2PipelineConfig(PipelineConfig):
    """4-stage ZONOS2 pipeline skeleton.

    The integration is intentionally limited to configuration and registration.
    Runtime stage factories raise a clear ``NotImplementedError`` until the
    ZONOS2 text frontend, speaker conditioning, multi-codebook MoE decode, and
    DAC vocoder are ported from the reference implementation.
    """

    architecture: ClassVar[str] = "Zonos2ForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "ZONOS2ForConditionalGeneration",
        "Zonos2TTSForConditionalGeneration",
    )

    model_path: str
    entry_stage: str = "text_frontend"
    stages: list[StageConfig] = [
        StageConfig(
            name="text_frontend",
            process="pipeline",
            factory=f"{_PKG}.stages.create_text_frontend_executor",
            next="speaker_embedding",
        ),
        StageConfig(
            name="speaker_embedding",
            process="pipeline",
            factory=f"{_PKG}.stages.create_speaker_embedding_executor",
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
            factory=f"{_PKG}.stages.create_dac_vocoder_executor",
            factory_args={"gpu_id": 0, "dtype": "bfloat16"},
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = ZONOS2PipelineConfig
