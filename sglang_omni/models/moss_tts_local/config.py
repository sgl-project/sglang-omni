# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS Local (v1.5)."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.utils.cpu import bounded_intraop_threads

_PKG = "sglang_omni.models.moss_tts_local"
# Keep reference encoding with AR so process-scoped SGLang accounting includes
# its codec allocation. The vocoder is isolated: its Python-heavy packed decode
# otherwise stalls the AR scheduler thread under ordinary serving concurrency.
_COLOCATED_PREPROCESSING_GPU_MEMORY_FRACTION = 0.15
_COLOCATED_AR_GPU_MEMORY_FRACTION = 0.67
_COLOCATED_VOCODER_GPU_MEMORY_FRACTION = 0.18
_AR_MEM_FRACTION_STATIC = 0.85
_REF_AUDIO_CACHE_MAX_ITEMS = 8192
_REF_AUDIO_CACHE_MAX_BYTES = 64 * 1024 * 1024
_PREPROCESSING_MAX_CONCURRENCY = 16
_MAX_PIPELINE_INTRAOP_THREADS = 8


def _stages(*, codec_device: str, colocated: bool) -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(
                device=codec_device,
                max_concurrency=_PREPROCESSING_MAX_CONCURRENCY,
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
            engine=EngineArgs(
                mem_fraction_static=None if colocated else _AR_MEM_FRACTION_STATIC
            ),
            gpu_memory_fraction=(
                _COLOCATED_AR_GPU_MEMORY_FRACTION if colocated else None
            ),
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="vocoder" if colocated else "pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(device=codec_device),
            gpu_memory_fraction=(
                _COLOCATED_VOCODER_GPU_MEMORY_FRACTION if colocated else None
            ),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class MossTTSLocalPipelineConfig(PipelineConfig):
    """Single-GPU MOSS-TTS Local pipeline."""

    architecture: ClassVar[str] = "MossTTSLocalModel"
    requires_model_capabilities: ClassVar[bool] = True
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "MossTTSLocal",
        "MossTTSLocalForConditionalGeneration",
    )
    additional_speech_languages: ClassVar[frozenset[str]] = frozenset(
        {
            "Cantonese",
            "Arabic",
            "Czech",
            "Danish",
            "Dutch",
            "Finnish",
            "Greek",
            "Hebrew",
            "Hindi",
            "Hungarian",
            "Macedonian",
            "Malay",
            "Persian (Farsi)",
            "Polish",
            "Romanian",
            "Swahili",
            "Swedish",
            "Tagalog",
            "Thai",
            "Turkish",
            "Vietnamese",
        }
    )

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        # Note (Akazaakane): preprocessing publishes prepared requests into a
        # module-level PreparedRequestQueue that the AR stage pops in-process.
        return frozenset({("preprocessing", "tts_engine")})

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:0", colocated=True)
    )

    # Streaming-vocoder CUDA-graph knobs, passed to the vocoder factory via
    # stage_factory_kwargs. Default on, fail-safe to eager;
    # cuda_graph_frames=None captures exact T=1..stream_chunk.
    cuda_graph: bool = True
    cuda_graph_frames: list[int] | None = None
    cuda_graph_min_free_gb: float = 3.0
    ref_audio_cache: bool = True
    ref_audio_cache_max_items: int = _REF_AUDIO_CACHE_MAX_ITEMS
    ref_audio_cache_max_bytes: int = _REF_AUDIO_CACHE_MAX_BYTES

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name == "preprocessing":
            return {
                "ref_audio_cache": self.ref_audio_cache,
                "ref_audio_cache_max_items": self.ref_audio_cache_max_items,
                "ref_audio_cache_max_bytes": self.ref_audio_cache_max_bytes,
            }
        if stage_name == "tts_engine":
            engine_stage = self.stage_named("tts_engine")
            if engine_stage.gpu_memory_fraction is not None:
                # Colocated layouts budget the codec reserve through the
                # per-stage fractions instead of the engine-side reserve.
                return {"codec_mem_reserve": 0.0}
            return {}
        if stage_name == "vocoder":
            return {
                "cuda_graph": self.cuda_graph,
                "cuda_graph_frames": self.cuda_graph_frames,
                "cuda_graph_min_free_gb": self.cuda_graph_min_free_gb,
            }
        return {}

    def resolved_env_defaults(self) -> dict[str, str]:
        preprocessing = next(
            (stage for stage in self.stages if stage.name == "preprocessing"),
            None,
        )
        if preprocessing is None:
            return dict(self.env_defaults)
        configured_workers = preprocessing.factory.max_concurrency
        preprocessing_workers = max(
            int(
                configured_workers
                if configured_workers is not None
                else _PREPROCESSING_MAX_CONCURRENCY
            ),
            1,
        )
        # Stage processes must inherit this before importing Torch/OpenMP-backed
        # libraries. Calling torch.set_num_threads() inside preprocessing is too
        # late for the separately spawned AR and vocoder processes. Derived at
        # launch so a rebuilt config re-derives from the resolved concurrency;
        # a written env_defaults entry wins.
        derived = {
            "OMP_NUM_THREADS": str(
                bounded_intraop_threads(
                    worker_count=preprocessing_workers,
                    max_threads=_MAX_PIPELINE_INTRAOP_THREADS,
                )
            )
        }
        return {**derived, **self.env_defaults}

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        if self.ref_audio_cache_max_items < 1:
            raise ValueError(
                "ref_audio_cache_max_items must be >= 1; got "
                f"{self.ref_audio_cache_max_items}"
            )
        if self.ref_audio_cache_max_bytes < 1:
            raise ValueError(
                "ref_audio_cache_max_bytes must be >= 1; got "
                f"{self.ref_audio_cache_max_bytes}"
            )
        if self.cuda_graph_min_free_gb < 0:
            raise ValueError(
                "cuda_graph_min_free_gb must be >= 0 (0 disables the VRAM headroom "
                f"guard); got {self.cuda_graph_min_free_gb}"
            )
        if self.cuda_graph_frames is not None:
            if not self.cuda_graph_frames:
                raise ValueError(
                    "cuda_graph_frames must be non-empty; set `cuda_graph: false` to "
                    "disable graphs, or leave it null to use the default capture set"
                )
            invalid = [t for t in self.cuda_graph_frames if t < 1]
            if invalid:
                raise ValueError(
                    f"cuda_graph_frames entries must be positive ints (>= 1); got {invalid}"
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

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        # Note (Akazaakane): split mode declares gpu=0 for placement while running the
        # codec on cuda:1, so the colocated fractions do not describe this topology.
        # Splitting stays unsupported here until the split variant declares its own.
        return frozenset({("preprocessing", "tts_engine"), ("tts_engine", "vocoder")})

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:1", colocated=False)
    )


EntryClass = MossTTSLocalPipelineConfig

Variants = {
    "default": MossTTSLocalPipelineConfig,
    "colocated": MossTTSLocalColocatedPipelineConfig,
    "split": MossTTSLocalSplitPipelineConfig,
}
