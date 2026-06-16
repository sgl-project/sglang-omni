# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS Local (v1.5)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    PipelineConfig,
    SGLangServerArgsConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
)

_PKG = "sglang_omni.models.moss_tts_local"
# Note (Ratish): in the default single-process topology preprocessing loads before AR, so its
# codec memory is included by process-scoped SGLang accounting
# the reserve is for the later vocoder codec instance and runtime headroom
_COLOCATED_TOTAL_GPU_MEMORY_FRACTION = 0.90
_COLOCATED_CODEC_MEM_RESERVE = 0.05
_AR_MEM_FRACTION_STATIC = 0.85


def _stages(*, codec_device: str, colocated: bool) -> list[StageConfig]:
    tts_engine_runtime = StageRuntimeConfig(
        resources=StageResourceConfig(
            total_gpu_memory_fraction=(
                _COLOCATED_TOTAL_GPU_MEMORY_FRACTION if colocated else None
            )
        ),
        sglang_server_args=SGLangServerArgsConfig(
            mem_fraction_static=None if colocated else _AR_MEM_FRACTION_STATIC
        ),
    )
    tts_engine_args = {"gpu_id": 0, "dtype": "bfloat16"}
    if colocated:
        tts_engine_args["codec_mem_reserve"] = _COLOCATED_CODEC_MEM_RESERVE

    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={
                "device": codec_device,
                "ref_audio_cache": True,
                "ref_audio_cache_max_items": 8192,
                "ref_audio_cache_max_bytes": 64 * 1024 * 1024,
            },
            gpu=0,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args=tts_engine_args,
            runtime=tts_engine_runtime,
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"device": codec_device},
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class MossTTSLocalPipelineConfig(PipelineConfig):
    """Single-GPU MOSS-TTS Local pipeline."""

    architecture: ClassVar[str] = "MossTTSLocalModel"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "MossTTSLocal",
        "MossTTSLocalForConditionalGeneration",
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    model_path: str
    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:0", colocated=True)
    )

    def supports_uploaded_voice_references(self) -> bool:
        return True


class MossTTSLocalColocatedPipelineConfig(MossTTSLocalPipelineConfig):
    """Backward-compatible alias for the default single-GPU pipeline."""

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:0", colocated=True)
    )


class MossTTSLocalSplitPipelineConfig(MossTTSLocalPipelineConfig):
    """Two-GPU variant that places codec work on the second visible GPU."""

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:1", colocated=False)
    )


EntryClass = MossTTSLocalPipelineConfig

Variants = {
    "default": MossTTSLocalPipelineConfig,
    "colocated": MossTTSLocalColocatedPipelineConfig,
    "split": MossTTSLocalSplitPipelineConfig,
}
