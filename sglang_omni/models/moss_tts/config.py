# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS Delay."""

from __future__ import annotations

import os
from typing import ClassVar, Literal

from pydantic import Field

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend

_PKG = "sglang_omni.models.moss_tts"
_REF_AUDIO_CACHE_MAX_ITEMS = 8192
_REF_AUDIO_CACHE_MAX_BYTES = 64 * 1024 * 1024
_COLOCATED_PREPROCESSING_GPU_MEMORY_FRACTION = 0.10
# note (Zhuofeng Li): dense prefill graphs need headroom beyond the KV budget.
_COLOCATED_AR_GPU_MEMORY_FRACTION = 0.60
_COLOCATED_VOCODER_GPU_MEMORY_FRACTION = 0.18
# note (Zhuofeng Li): exact token counts preserve the eager prefill shape.
COLOCATED_PREFILL_CUDA_GRAPH_MAX_TOKENS = 512


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


class MossTTSEngineFactoryArgs(FactoryArgs):
    async_decode_mode: Literal["all", "nonstream"] | None = None


class MossTTSEngineStageConfig(EngineStageConfig):
    factory: MossTTSEngineFactoryArgs = Field(default_factory=MossTTSEngineFactoryArgs)


class MossTTSVocoderFactoryArgs(FactoryArgs):
    compute_dtype: str | None = None
    segment_graph_max_frames: int = Field(default=0, ge=0)


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
        MossTTSEngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=MossTTSEngineFactoryArgs(
                dtype="bfloat16",
                async_decode_mode="nonstream" if colocated else None,
            ),
            engine=(
                EngineArgs(
                    cuda_graph_backend_prefill=CudaGraphBackend.BREAKABLE,
                    cuda_graph_bs_prefill=list(
                        range(1, COLOCATED_PREFILL_CUDA_GRAPH_MAX_TOKENS + 1)
                    ),
                )
                if colocated
                else EngineArgs()
            ),
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
                dtype="float32",
                compute_dtype="bfloat16",
                segment_graph_max_frames=128 if colocated else 0,
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
        "tts_engine": MossTTSEngineStageConfig,
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
