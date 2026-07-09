# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS Delay."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.moss_tts"
_DEFAULT_MOSS_AUDIO_TOKENIZER = "OpenMOSS-Team/MOSS-Audio-Tokenizer"


class MossTTSPipelineConfig(PipelineConfig):
    """MOSS-TTS Delay pipeline: preprocessing -> AR engine -> vocoder."""

    architecture: ClassVar[str] = "MossTTSDelayModel"
    requires_model_capabilities: ClassVar[bool] = True
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

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "tts_engine"}

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={
                "codec_model_path": _DEFAULT_MOSS_AUDIO_TOKENIZER,
                "device": "cuda:0",
                "dtype": "float32",
            },
            gpu=0,
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
            factory_args={
                "codec_model_path": _DEFAULT_MOSS_AUDIO_TOKENIZER,
                "dtype": "float32",
            },
            gpu=0,
            terminal=True,
        ),
    ]

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = MossTTSPipelineConfig
