# SPDX-License-Identifier: Apache-2.0
"""Deployment configuration for MOSS-TTS-Realtime."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.moss_tts_realtime"

DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

# Colocated layouts budget the single pipeline process through the engine
# stage's gpu_memory_fraction; the engine derives its codec memory reserve from
# that budget (see engine_builder._derive_colocated_codec_memory_budget).
_COLOCATED_TOTAL_GPU_MEMORY_FRACTION = 0.90
_AR_MEM_FRACTION_STATIC = 0.85
_REF_AUDIO_CACHE_MAX_ITEMS = 8192
_REF_AUDIO_CACHE_MAX_BYTES = 64 * 1024 * 1024
_MAX_CUDA_GRAPH_FRAMES = 25


class MossTTSRealtimeResourceLimits(BaseModel):
    """Bounded ownership policy for realtime sessions and turns.

    Enforcement lands in the protocol/session/scheduler tasks. Keeping the
    values in one validated object makes those later paths share an identical
    admission contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_sessions: int = 64
    max_held_sessions: int = 64
    max_active_turns: int = 16
    max_pending_text_tokens: int = 4096
    max_pending_text_bytes: int = 256 * 1024
    max_input_updates: int = 8192
    terminal_tombstone_limit: int = 8192
    input_idle_timeout_s: float = 30.0
    turn_timeout_s: float = 600.0
    session_idle_ttl_s: float = 300.0

    def model_post_init(self, __context: Any = None) -> None:
        positive_int_fields = (
            "max_sessions",
            "max_held_sessions",
            "max_active_turns",
            "max_pending_text_tokens",
            "max_pending_text_bytes",
            "max_input_updates",
            "terminal_tombstone_limit",
        )
        for field_name in positive_int_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        positive_float_fields = (
            "input_idle_timeout_s",
            "turn_timeout_s",
            "session_idle_ttl_s",
        )
        for field_name in positive_float_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")

        if self.max_held_sessions > self.max_sessions:
            raise ValueError("max_held_sessions cannot exceed max_sessions")
        if self.max_active_turns > self.max_sessions:
            raise ValueError("max_active_turns cannot exceed max_sessions")


def _stages(*, codec_device: str, colocated: bool) -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(device=codec_device),
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
                _COLOCATED_TOTAL_GPU_MEMORY_FRACTION if colocated else None
            ),
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
            can_accept_stream_before_payload=True,
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(device=codec_device),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class MossTTSRealtimePipelineConfig(PipelineConfig):
    """Default single-GPU MOSS-TTS-Realtime pipeline."""

    architecture: ClassVar[str] = "MossTTSRealtime"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "MossTTSRealtimeModel",
        "MossTTSRealtimeForConditionalGeneration",
    )
    coordinator_factory: ClassVar[str] = (
        "sglang_omni.pipeline.realtime_coordinator.RealtimeCoordinator"
    )
    requires_model_capabilities: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    codec_model_path: str = DEFAULT_MOSS_TTS_REALTIME_CODEC_MODEL
    realtime_input_stage: str = "tts_engine"
    cuda_graph: bool = True
    cuda_graph_frames: list[int] | None = None
    cuda_graph_min_free_gb: float = 3.0
    vocoder_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    ref_audio_cache: bool = True
    ref_audio_cache_max_items: int = _REF_AUDIO_CACHE_MAX_ITEMS
    ref_audio_cache_max_bytes: int = _REF_AUDIO_CACHE_MAX_BYTES
    limits: MossTTSRealtimeResourceLimits = Field(
        default_factory=MossTTSRealtimeResourceLimits
    )
    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:0", colocated=True)
    )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name == "preprocessing":
            return {
                "codec_model_path": self.codec_model_path,
                "ref_audio_cache": self.ref_audio_cache,
                "ref_audio_cache_max_items": self.ref_audio_cache_max_items,
                "ref_audio_cache_max_bytes": self.ref_audio_cache_max_bytes,
            }
        if stage_name == "tts_engine":
            return {
                "codec_model_path": self.codec_model_path,
                **self.limits.model_dump(),
            }
        if stage_name == "vocoder":
            return {
                "codec_model_path": self.codec_model_path,
                # Slots are leased per realtime session, not per turn: a held
                # (warm) session keeps its codec streaming state across turns.
                # Size the pool so every held session plus every active turn
                # can own one slot without contention.
                "stream_slots": (
                    self.limits.max_held_sessions + self.limits.max_active_turns
                ),
                "session_idle_ttl_s": self.limits.session_idle_ttl_s,
                "cuda_graph": self.cuda_graph,
                "cuda_graph_frames": self.cuda_graph_frames,
                "cuda_graph_min_free_gb": self.cuda_graph_min_free_gb,
                "dtype": self.vocoder_dtype,
            }
        return {}

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        if not self.codec_model_path.strip():
            raise ValueError("codec_model_path must not be empty")
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
            raise ValueError("cuda_graph_min_free_gb must be non-negative")
        if self.cuda_graph_frames is not None:
            if not self.cuda_graph_frames:
                raise ValueError("cuda_graph_frames must not be empty")
            invalid = [
                frame
                for frame in self.cuda_graph_frames
                if not 1 <= frame <= _MAX_CUDA_GRAPH_FRAMES
            ]
            if invalid:
                raise ValueError(
                    "cuda_graph_frames entries must be integers in "
                    f"[1, {_MAX_CUDA_GRAPH_FRAMES}], "
                    f"got {invalid}"
                )
        stage_by_name = {stage.name: stage for stage in self.stages}
        input_stage = stage_by_name.get(self.realtime_input_stage)
        if input_stage is None:
            raise ValueError(
                f"realtime_input_stage {self.realtime_input_stage!r} is not defined"
            )
        if not input_stage.can_accept_stream_before_payload:
            raise ValueError(
                "realtime_input_stage must accept updates before its ordinary payload"
            )

    def supports_uploaded_voice_references(self) -> bool:
        return True

    def build_speech_realtime_handler(self) -> Any:
        from sglang_omni.models.moss_tts_realtime.speech_ws import (
            create_moss_tts_realtime_speech_ws_handler,
        )
        from sglang_omni.models.moss_tts_realtime.text_delta import (
            load_moss_tts_realtime_text_tokenizer,
        )

        tokenizer = load_moss_tts_realtime_text_tokenizer(self.model_path)
        return create_moss_tts_realtime_speech_ws_handler(
            tokenizer=tokenizer,
            limits=self.limits,
            realtime_input_stage=self.realtime_input_stage,
        )


class MossTTSRealtimeColocatedPipelineConfig(MossTTSRealtimePipelineConfig):
    """Explicit alias for the default colocated topology."""

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:0", colocated=True)
    )


class MossTTSRealtimeSplitPipelineConfig(MossTTSRealtimePipelineConfig):
    """Two-GPU topology with codec work targeting the second visible GPU."""

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:1", colocated=False)
    )


EntryClass = MossTTSRealtimePipelineConfig

Variants = {
    "default": MossTTSRealtimePipelineConfig,
    "colocated": MossTTSRealtimeColocatedPipelineConfig,
    "split": MossTTSRealtimeSplitPipelineConfig,
}
