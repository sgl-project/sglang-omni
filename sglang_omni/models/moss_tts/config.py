# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS Delay."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.moss_tts"
_REF_AUDIO_CACHE_MAX_ITEMS = 8192
_REF_AUDIO_CACHE_MAX_BYTES = 64 * 1024 * 1024


class MossTTSPreprocessingFactoryArgs(FactoryArgs):
    """Reference-encode cache knobs, typed like the shared ones."""

    compute_dtype: str | None = None
    ref_audio_cache: bool | None = None
    ref_audio_cache_max_items: int | None = Field(default=None, ge=1)
    ref_audio_cache_max_bytes: int | None = Field(default=None, ge=1)


class MossTTSPreprocessingStageConfig(StageConfig):
    factory: MossTTSPreprocessingFactoryArgs = Field(
        default_factory=MossTTSPreprocessingFactoryArgs
    )


class MossTTSVocoderFactoryArgs(FactoryArgs):
    compute_dtype: str | None = None


class MossTTSVocoderStageConfig(StageConfig):
    factory: MossTTSVocoderFactoryArgs = Field(
        default_factory=MossTTSVocoderFactoryArgs
    )


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

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "preprocessing": MossTTSPreprocessingStageConfig,
        "tts_engine": EngineStageConfig,
        "vocoder": MossTTSVocoderStageConfig,
    }

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        # Note (Akazaakane): preprocessing publishes into the module-level
        # PreparedRequestQueue that the AR stage pops in-process.
        return frozenset({("preprocessing", "tts_engine")})

    stages: list[StageConfig] = [
        MossTTSPreprocessingStageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=MossTTSPreprocessingFactoryArgs(
                compute_dtype="bfloat16",
                ref_audio_cache=True,
                ref_audio_cache_max_items=_REF_AUDIO_CACHE_MAX_ITEMS,
                ref_audio_cache_max_bytes=_REF_AUDIO_CACHE_MAX_BYTES,
            ),
            gpu=0,
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        MossTTSVocoderStageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=MossTTSVocoderFactoryArgs(
                dtype="float32", compute_dtype="bfloat16"
            ),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = MossTTSPipelineConfig
