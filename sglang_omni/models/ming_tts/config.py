# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Ming-Omni-TTS 16B."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.ming_tts"

PREPROCESSING_STAGE = "preprocessing"
REFERENCE_ENCODE_STAGE = "reference_encode"
TTS_ENGINE_STAGE = "tts_engine"
AUDIO_DECODE_STAGE = "audio_decode"

MING_TTS_AUDIO_DECODE_MAX_BATCH_SIZE = 1
MING_TTS_AUDIO_DECODE_MAX_BATCH_WAIT_MS = 0
MING_TTS_DEFAULT_STREAM_SLOTS = 1
MING_TTS_DEFAULT_MAX_DECODE_STEPS_CAP = 256
MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES = 2
MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES = 4
MING_TTS_DEFAULT_STREAMING_CUDA_GRAPH = False


def _validate_ming_tts_pipeline_contract(
    config: "MingTTSPipelineConfig",
) -> dict[str, StageConfig]:
    """Validate the fixed Ming payload and streaming topology."""

    expected_next = {
        PREPROCESSING_STAGE: REFERENCE_ENCODE_STAGE,
        REFERENCE_ENCODE_STAGE: TTS_ENGINE_STAGE,
        TTS_ENGINE_STAGE: AUDIO_DECODE_STAGE,
        AUDIO_DECODE_STAGE: None,
    }
    stages = {stage.name: stage for stage in config.stages}
    if set(stages) != set(expected_next):
        raise ValueError(
            "Ming-Omni-TTS requires exactly the stages "
            f"{sorted(expected_next)}; got {sorted(stages)}"
        )
    if config.resolved_entry_stage != PREPROCESSING_STAGE:
        raise ValueError(f"Ming-Omni-TTS entry_stage must be {PREPROCESSING_STAGE!r}")

    for stage_name, next_stage in expected_next.items():
        stage = stages[stage_name]
        next_targets = (
            []
            if stage.next is None
            else [stage.next] if isinstance(stage.next, str) else list(stage.next)
        )
        expected_targets = [] if next_stage is None else [next_stage]
        expected_terminal = next_stage is None
        if next_targets != expected_targets or stage.terminal != expected_terminal:
            raise ValueError(
                f"Ming-Omni-TTS stage {stage_name!r} must set "
                f"next={expected_targets!r}, terminal={expected_terminal}; got "
                f"next={next_targets!r}, terminal={stage.terminal}"
            )

    tts_engine = stages[TTS_ENGINE_STAGE]
    if list(tts_engine.stream_to) != [AUDIO_DECODE_STAGE]:
        raise ValueError(
            "Ming-Omni-TTS tts_engine stream_to must include 'audio_decode' as "
            f"its only target, got {list(tts_engine.stream_to)!r}"
        )
    for stage_name in (
        PREPROCESSING_STAGE,
        REFERENCE_ENCODE_STAGE,
        AUDIO_DECODE_STAGE,
    ):
        if stages[stage_name].stream_to:
            raise ValueError(
                f"Ming-Omni-TTS stage {stage_name!r} stream_to must be empty"
            )

    audio_decode = stages[AUDIO_DECODE_STAGE]
    if not audio_decode.can_accept_stream_before_payload:
        raise ValueError(
            "Ming-Omni-TTS audio_decode must set "
            "can_accept_stream_before_payload=true because "
            "tts_engine sends stream data and stream_done before the terminal payload"
        )

    dynamic_fields = [
        f"{stage.name}.{field_name}"
        for stage in config.stages
        for field_name in (
            "route_fn",
            "wait_for",
            "wait_for_fn",
            "merge_fn",
            "stream_done_to_fn",
            "project_payload",
        )
        if getattr(stage, field_name)
    ]
    if config.terminal_stages_fn is not None:
        dynamic_fields.append("terminal_stages_fn")
    if dynamic_fields:
        raise ValueError(
            "Ming-Omni-TTS fixed pipeline must not configure dynamic data-flow "
            f"fields: {sorted(dynamic_fields)}"
        )
    return stages


def validate_ming_tts_audio_decode_batch_config(
    *,
    max_batch_size: int,
    max_batch_wait_ms: int,
) -> None:
    if (
        isinstance(max_batch_size, bool)
        or not isinstance(max_batch_size, int)
        or max_batch_size != MING_TTS_AUDIO_DECODE_MAX_BATCH_SIZE
    ):
        raise ValueError(
            "Ming-Omni-TTS audio_decode currently supports "
            f"max_batch_size={MING_TTS_AUDIO_DECODE_MAX_BATCH_SIZE} only; "
            "cross-request non-streaming AudioVAE batching is not implemented, "
            f"got {max_batch_size!r}"
        )
    if (
        isinstance(max_batch_wait_ms, bool)
        or not isinstance(max_batch_wait_ms, int)
        or max_batch_wait_ms != MING_TTS_AUDIO_DECODE_MAX_BATCH_WAIT_MS
    ):
        raise ValueError(
            "Ming-Omni-TTS audio_decode currently supports "
            f"max_batch_wait_ms={MING_TTS_AUDIO_DECODE_MAX_BATCH_WAIT_MS} only; "
            "cross-request non-streaming AudioVAE batching is not implemented, got "
            f"{max_batch_wait_ms!r}"
        )


def validate_ming_tts_audio_decode_stream_slots(stream_slots: int) -> None:
    if (
        isinstance(stream_slots, bool)
        or not isinstance(stream_slots, int)
        or stream_slots <= 0
    ):
        raise ValueError(
            "Ming-Omni-TTS audio_decode stream_slots must be a positive integer, "
            f"got {stream_slots!r}"
        )


def validate_ming_tts_audio_decode_cadence_config(
    *,
    initial_chunk_patches: int,
    steady_chunk_patches: int,
) -> None:
    if (
        isinstance(initial_chunk_patches, bool)
        or not isinstance(initial_chunk_patches, int)
        or initial_chunk_patches <= 0
    ):
        raise ValueError(
            "Ming-Omni-TTS initial_chunk_patches must be a positive integer"
        )
    if (
        isinstance(steady_chunk_patches, bool)
        or not isinstance(steady_chunk_patches, int)
        or steady_chunk_patches <= 0
    ):
        raise ValueError(
            "Ming-Omni-TTS steady_chunk_patches must be a positive integer"
        )


class MingTTSPreprocessingFactoryArgs(FactoryArgs):
    max_decode_steps_cap: int | None = Field(default=None, gt=0, strict=True)


class MingTTSPreprocessingStageConfig(StageConfig):
    factory: MingTTSPreprocessingFactoryArgs = Field(
        default_factory=MingTTSPreprocessingFactoryArgs
    )


class MingTTSAudioDecodeFactoryArgs(FactoryArgs):
    streaming_cuda_graph: bool | None = Field(default=None, strict=True)
    initial_chunk_patches: int | None = Field(default=None, gt=0, strict=True)
    steady_chunk_patches: int | None = Field(default=None, gt=0, strict=True)
    stream_slots: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description=(
            "Concurrent streaming AudioVAE state slots, fixed transition width, "
            "and nonblocking stream-chunk collector cap"
        ),
    )
    max_batch_size: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Cross-request non-streaming AudioVAE batch size; currently 1",
    )
    max_batch_wait_ms: int | None = Field(
        default=None,
        ge=0,
        strict=True,
        description="Cross-request non-streaming AudioVAE batch wait; currently 0",
    )


class MingTTSAudioDecodeStageConfig(StageConfig):
    factory: MingTTSAudioDecodeFactoryArgs = Field(
        default_factory=MingTTSAudioDecodeFactoryArgs
    )


class MingTTSPipelineConfig(PipelineConfig):
    """Ming-Omni-TTS pipeline.

    preprocessing -> reference_encode -> tts_engine -> audio_decode.
    The reference stage is kept as a fixed cheap/no-op boundary for text-only
    requests so reference-conditioned requests use the same serving graph.
    """

    architecture: ClassVar[str] = "BailingMMNativeForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        PREPROCESSING_STAGE: MingTTSPreprocessingStageConfig,
        TTS_ENGINE_STAGE: EngineStageConfig,
        AUDIO_DECODE_STAGE: MingTTSAudioDecodeStageConfig,
    }

    entry_stage: str = PREPROCESSING_STAGE
    stages: list[StageConfig] = [
        MingTTSPreprocessingStageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=MingTTSPreprocessingFactoryArgs(
                max_decode_steps_cap=MING_TTS_DEFAULT_MAX_DECODE_STEPS_CAP,
            ),
            next=REFERENCE_ENCODE_STAGE,
        ),
        StageConfig(
            name=REFERENCE_ENCODE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_reference_encode_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=0,
            next=TTS_ENGINE_STAGE,
        ),
        EngineStageConfig(
            name=TTS_ENGINE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=0,
            next=AUDIO_DECODE_STAGE,
            stream_to=[AUDIO_DECODE_STAGE],
        ),
        MingTTSAudioDecodeStageConfig(
            name=AUDIO_DECODE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_audio_decode_executor",
            factory=MingTTSAudioDecodeFactoryArgs(
                dtype="bfloat16",
                initial_chunk_patches=MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES,
                steady_chunk_patches=MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES,
                streaming_cuda_graph=MING_TTS_DEFAULT_STREAMING_CUDA_GRAPH,
                stream_slots=MING_TTS_DEFAULT_STREAM_SLOTS,
                max_batch_size=MING_TTS_AUDIO_DECODE_MAX_BATCH_SIZE,
                max_batch_wait_ms=MING_TTS_AUDIO_DECODE_MAX_BATCH_WAIT_MS,
            ),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        stages = _validate_ming_tts_pipeline_contract(self)
        preprocessing = stages[PREPROCESSING_STAGE]
        audio_decode = stages[AUDIO_DECODE_STAGE]

        # The value rules (positive ints) are static Field declarations on
        # the typed groups; only the model's cross-value contracts remain.
        del preprocessing
        decode_extra = audio_decode.factory.model_extra or {}
        if "decode_mode" in decode_extra:
            raise ValueError(
                "Ming-Omni-TTS audio_decode no longer supports 'decode_mode'. "
                "Non-streaming requests always use full-sequence AudioVAE "
                "decode; streaming requests use incremental decode. Remove "
                "'decode_mode' from the audio_decode factory group."
            )
        max_batch_wait_ms = audio_decode.factory.max_batch_wait_ms
        if max_batch_wait_ms is None:
            max_batch_wait_ms = MING_TTS_AUDIO_DECODE_MAX_BATCH_WAIT_MS
        stream_slots = audio_decode.factory.stream_slots
        if stream_slots is None:
            stream_slots = MING_TTS_DEFAULT_STREAM_SLOTS
        validate_ming_tts_audio_decode_stream_slots(stream_slots)
        validate_ming_tts_audio_decode_batch_config(
            max_batch_size=(
                audio_decode.factory.max_batch_size
                if audio_decode.factory.max_batch_size is not None
                else MING_TTS_AUDIO_DECODE_MAX_BATCH_SIZE
            ),
            max_batch_wait_ms=max_batch_wait_ms,
        )

        for stage in self.stages:
            if stage.name != TTS_ENGINE_STAGE:
                if stage.tp_size != 1:
                    raise ValueError(
                        "Ming-Omni-TTS supports tensor parallelism only on "
                        f"{TTS_ENGINE_STAGE!r}; stage {stage.name!r} has "
                        f"tp_size={stage.tp_size}."
                    )
                continue


EntryClass = MingTTSPipelineConfig
