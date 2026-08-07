# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS-Realtime."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import (
    PipelineConfig,
    SGLangServerArgsConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
)

_PKG = "sglang_omni.models.moss_tts_realtime"


def _stages() -> list[StageConfig]:
    runtime = StageRuntimeConfig(
        resources=StageResourceConfig(total_gpu_memory_fraction=0.92),
        sglang_server_args=SGLangServerArgsConfig(mem_fraction_static=None),
    )
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            gpu=0,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"dtype": "bfloat16", "codec_mem_reserve": 0.22},
            runtime=runtime,
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


class MossTTSRealtimePipelineConfig(PipelineConfig):
    """Single-GPU MOSS-TTS-Realtime pipeline."""

    architecture: ClassVar[str] = "MossTTSRealtime"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "MossTTSRealtimeForConditionalGeneration",
    )
    requires_model_capabilities: ClassVar[bool] = True
    additional_speech_languages: ClassVar[frozenset[str]] = frozenset(
        {
            "Arabic",
            "Czech",
            "Danish",
            "Dutch",
            "Greek",
            "Hebrew",
            "Hungarian",
            "Italian",
            "Korean",
            "Persian (Farsi)",
            "Polish",
            "Portuguese",
            "Russian",
            "Swedish",
            "Turkish",
        }
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
    stages: list[StageConfig] = Field(default_factory=_stages)
    ref_audio_cache_max_items: int = 256
    ref_audio_cache_max_bytes: int = 64 * 1024 * 1024
    stream_chunk_frames: int = 6
    initial_chunk_frames: int = 1

    def model_post_init(self, __context: Any = None, /) -> None:
        super().model_post_init(__context)
        if self.ref_audio_cache_max_items < 1:
            raise ValueError("ref_audio_cache_max_items must be >= 1")
        if self.ref_audio_cache_max_bytes < 1:
            raise ValueError("ref_audio_cache_max_bytes must be >= 1")
        if self.stream_chunk_frames < 1:
            raise ValueError("stream_chunk_frames must be >= 1")
        if not 1 <= self.initial_chunk_frames <= self.stream_chunk_frames:
            raise ValueError("initial_chunk_frames must be in [1, stream_chunk_frames]")
        for stage in self.stages:
            if stage.factory.endswith("create_preprocessing_executor"):
                stage.factory_args.setdefault(
                    "cache_max_items", self.ref_audio_cache_max_items
                )
                stage.factory_args.setdefault(
                    "cache_max_bytes", self.ref_audio_cache_max_bytes
                )
            if stage.factory.endswith("create_vocoder_executor"):
                stage.factory_args.setdefault(
                    "stream_chunk_frames", self.stream_chunk_frames
                )
                stage.factory_args.setdefault(
                    "initial_chunk_frames", self.initial_chunk_frames
                )

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = MossTTSRealtimePipelineConfig
