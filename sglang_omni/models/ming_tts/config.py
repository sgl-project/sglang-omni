# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Ming-Omni-TTS 16B."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.ming_tts"

PREPROCESSING_STAGE = "preprocessing"
REFERENCE_ENCODE_STAGE = "reference_encode"
TTS_ENGINE_STAGE = "tts_engine"
AUDIO_DECODE_STAGE = "audio_decode"

MING_TTS_AUDIO_DECODE_MAX_BATCH_SIZE = 1
MING_TTS_AUDIO_DECODE_MAX_BATCH_WAIT_MS = 0
MING_TTS_DEFAULT_MAX_DECODE_STEPS_CAP = 256
MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES = 2
MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES = 4


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
) -> tuple[int, int]:
    if (
        isinstance(max_batch_size, bool)
        or not isinstance(max_batch_size, int)
        or max_batch_size != MING_TTS_AUDIO_DECODE_MAX_BATCH_SIZE
    ):
        raise ValueError(
            "Ming-Omni-TTS audio_decode currently supports "
            f"max_batch_size={MING_TTS_AUDIO_DECODE_MAX_BATCH_SIZE} only; "
            "cross-request AudioVAE batching is not implemented yet, got "
            f"{max_batch_size!r}"
        )
    if (
        isinstance(max_batch_wait_ms, bool)
        or not isinstance(max_batch_wait_ms, int)
        or max_batch_wait_ms != MING_TTS_AUDIO_DECODE_MAX_BATCH_WAIT_MS
    ):
        raise ValueError(
            "Ming-Omni-TTS audio_decode currently supports "
            f"max_batch_wait_ms={MING_TTS_AUDIO_DECODE_MAX_BATCH_WAIT_MS} only "
            "while cross-request AudioVAE batching is disabled, got "
            f"{max_batch_wait_ms!r}"
        )
    return max_batch_size, max_batch_wait_ms


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


class MingTTSPipelineConfig(PipelineConfig):
    """Ming-Omni-TTS pipeline.

    preprocessing -> reference_encode -> tts_engine -> audio_decode.
    The reference stage is kept as a fixed cheap/no-op boundary for text-only
    requests so reference-conditioned requests use the same serving graph.
    """

    architecture: ClassVar[str] = "BailingMMNativeForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": TTS_ENGINE_STAGE}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": TTS_ENGINE_STAGE}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": TTS_ENGINE_STAGE}

    @classmethod
    def isolation_role_to_stage(cls) -> dict[str, str]:
        return {"vocoder": AUDIO_DECODE_STAGE}

    @classmethod
    def process_safe_edges(cls) -> frozenset[tuple[str, str]]:
        return frozenset({(TTS_ENGINE_STAGE, AUDIO_DECODE_STAGE)})

    @classmethod
    def process_edge_resources(
        cls,
    ) -> dict[tuple[str, str], dict[str, float]]:
        return {
            (TTS_ENGINE_STAGE, AUDIO_DECODE_STAGE): {
                REFERENCE_ENCODE_STAGE: 0.08,
                TTS_ENGINE_STAGE: 0.72,
                AUDIO_DECODE_STAGE: 0.12,
            }
        }

    model_path: str
    entry_stage: str = PREPROCESSING_STAGE
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={
                "max_decode_steps_cap": MING_TTS_DEFAULT_MAX_DECODE_STEPS_CAP
            },
            next=REFERENCE_ENCODE_STAGE,
        ),
        StageConfig(
            name=REFERENCE_ENCODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_reference_encode_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            next=TTS_ENGINE_STAGE,
        ),
        StageConfig(
            name=TTS_ENGINE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            next=AUDIO_DECODE_STAGE,
            stream_to=[AUDIO_DECODE_STAGE],
        ),
        StageConfig(
            name=AUDIO_DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_audio_decode_executor",
            factory_args={
                "dtype": "bfloat16",
                "initial_chunk_patches": MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES,
                "steady_chunk_patches": MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES,
                "max_batch_size": MING_TTS_AUDIO_DECODE_MAX_BATCH_SIZE,
                "max_batch_wait_ms": MING_TTS_AUDIO_DECODE_MAX_BATCH_WAIT_MS,
            },
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

        preprocessing_overrides = self.runtime_overrides.get(PREPROCESSING_STAGE, {})
        if "max_decode_steps_cap" in preprocessing_overrides:
            raise ValueError(
                "Ming-Omni-TTS max_decode_steps_cap is owned by "
                "preprocessing.factory_args, not runtime_overrides"
            )
        max_decode_steps_cap = preprocessing.factory_args.setdefault(
            "max_decode_steps_cap", MING_TTS_DEFAULT_MAX_DECODE_STEPS_CAP
        )
        if max_decode_steps_cap is not None and (
            isinstance(max_decode_steps_cap, bool)
            or not isinstance(max_decode_steps_cap, int)
            or max_decode_steps_cap <= 0
        ):
            raise ValueError(
                "Ming-Omni-TTS preprocessing.factory_args.max_decode_steps_cap "
                "must be a positive integer or null"
            )

        audio_decode_overrides = self.runtime_overrides.get(AUDIO_DECODE_STAGE, {})
        for field_name in ("initial_chunk_patches", "steady_chunk_patches"):
            if field_name in audio_decode_overrides:
                raise ValueError(
                    f"Ming-Omni-TTS {field_name} is owned by "
                    "audio_decode.factory_args, not runtime_overrides"
                )
        initial_chunk_patches = audio_decode.factory_args.setdefault(
            "initial_chunk_patches", MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES
        )
        steady_chunk_patches = audio_decode.factory_args.setdefault(
            "steady_chunk_patches", MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES
        )
        validate_ming_tts_audio_decode_cadence_config(
            initial_chunk_patches=initial_chunk_patches,
            steady_chunk_patches=steady_chunk_patches,
        )
        if "decode_mode" in audio_decode.factory_args or (
            "decode_mode" in audio_decode_overrides
        ):
            raise ValueError(
                "Ming-Omni-TTS audio_decode no longer supports 'decode_mode'. "
                "Non-streaming requests always use full-sequence AudioVAE decode; "
                "streaming requests use incremental decode. Remove 'decode_mode' "
                "from audio_decode factory_args/runtime_overrides."
            )
        validate_ming_tts_audio_decode_batch_config(
            max_batch_size=audio_decode_overrides.get(
                "max_batch_size",
                audio_decode.factory_args.get(
                    "max_batch_size", MING_TTS_AUDIO_DECODE_MAX_BATCH_SIZE
                ),
            ),
            max_batch_wait_ms=audio_decode_overrides.get(
                "max_batch_wait_ms",
                audio_decode.factory_args.get(
                    "max_batch_wait_ms", MING_TTS_AUDIO_DECODE_MAX_BATCH_WAIT_MS
                ),
            ),
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

            if stage.tp_size <= 0:
                raise ValueError(
                    "Ming-Omni-TTS tts_engine tp_size must be positive; "
                    f"got tp_size={stage.tp_size}."
                )
            if stage.tp_size == 1:
                continue
            if not isinstance(stage.gpu, list):
                raise ValueError(
                    "Ming-Omni-TTS tts_engine tensor parallelism requires "
                    "gpu=[rank0_gpu, rank1_gpu, ...]."
                )
            if len(stage.gpu) != stage.tp_size:
                raise ValueError(
                    "Ming-Omni-TTS tts_engine TP GPU list length must match "
                    f"tp_size; got gpu={stage.gpu!r}, tp_size={stage.tp_size}."
                )


EntryClass = MingTTSPipelineConfig
