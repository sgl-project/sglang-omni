# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS Delay."""

from __future__ import annotations

import os
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
_COLOCATED_PREPROCESSING_GPU_MEMORY_FRACTION = 0.10
_COLOCATED_AR_GPU_MEMORY_FRACTION = 0.72
_COLOCATED_VOCODER_GPU_MEMORY_FRACTION = 0.18


def uses_rocm_wsl_dxg() -> bool:
    """Return whether PyTorch HIP can select the WSL DXG device path."""
    try:
        import torch
    except ImportError:
        return False
    return (
        torch.version.hip is not None
        and os.environ.get("HSA_ENABLE_DXG_DETECTION") != "0"
        and os.path.exists("/dev/dxg")
    )


def resolve_vocoder_cuda_graph(vocoder_cuda_graph: bool | None) -> bool:
    """Resolve the platform default and reject an unsafe DXG opt-in."""
    if not uses_rocm_wsl_dxg():
        return True if vocoder_cuda_graph is None else vocoder_cuda_graph
    else:
        pass
    if vocoder_cuda_graph is True:
        raise ValueError(
            "MOSS-TTS Local vocoder CUDA graphs cannot be enabled on ROCm "
            "WSL/DXG because HIP graph capture can abort the process; omit "
            "vocoder_cuda_graph or set it to false"
        )
    else:
        pass
    return False


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


def stages(*, colocated: bool) -> list[StageConfig]:
    # Note (Jiaxin Deng): the vocoder's Python decode loop shares the interpreter
    # with the AR scheduler when both live in one process; on H200 isolating it
    # lifts single-replica throughput by ~70% at every concurrency cap.
    return [
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
            gpu_memory_fraction=(
                _COLOCATED_PREPROCESSING_GPU_MEMORY_FRACTION if colocated else None
            ),
            gpu=0,
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu_memory_fraction=(
                _COLOCATED_AR_GPU_MEMORY_FRACTION if colocated else None
            ),
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        MossTTSVocoderStageConfig(
            name="vocoder",
            process="vocoder" if colocated else "pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=MossTTSVocoderFactoryArgs(
                dtype="float32", compute_dtype="bfloat16"
            ),
            gpu_memory_fraction=(
                _COLOCATED_VOCODER_GPU_MEMORY_FRACTION if colocated else None
            ),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class MossTTSPipelineConfig(PipelineConfig):
    """MOSS-TTS Delay pipeline: preprocessing -> AR engine -> vocoder."""

    architecture: ClassVar[str] = "MossTTSDelayModel"
    requires_model_capabilities: ClassVar[bool] = True
    max_speech_input_chars: ClassVar[int | None] = None
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

    stages: list[StageConfig] = Field(default_factory=lambda: stages(colocated=True))

    def supports_uploaded_voice_references(self) -> bool:
        return True


class MossTTSSingleProcessPipelineConfig(MossTTSPipelineConfig):
    """All three stages in one process, the layout before the vocoder split."""

    stages: list[StageConfig] = Field(default_factory=lambda: stages(colocated=False))


EntryClass = MossTTSPipelineConfig

Variants = {
    "default": MossTTSPipelineConfig,
    "single_process": MossTTSSingleProcessPipelineConfig,
}
