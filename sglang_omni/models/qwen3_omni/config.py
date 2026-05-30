# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-Omni."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, PlacementConfig, StageConfig

_PKG = "sglang_omni.models.qwen3_omni"
_PLACEMENT_POLICY = f"{_PKG}.placement.Qwen3OmniPlacementPolicy"
MIN_PARTIAL_START_CHUNKS = 3

# SGLang reads this when DeepGEMM compile utilities are imported. Qwen AR
# stages can first hit some dense FP8 shapes after readiness; disable all-M
# precompile so that miss does not become a long post-ready compile session.
# FIXME (Ratish): Replace this with a bounded/pre-ready SGLang DeepGEMM compile
# policy once that exists outside import-time environment globals.
_DEEPGEMM_PRECOMPILE_ENV_DEFAULTS = {"SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0"}


def _preprocessing_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name="preprocessing",
        process=process,
        factory=f"{_PKG}.stages.create_preprocessing_executor",
        factory_args={"thinker_max_seq_len": 8192},
        runtime_arg_map={
            "max_seq_len": "thinker_max_seq_len",
            "video_fps": "video_fps",
        },
        next=["image_encoder", "audio_encoder", "mm_aggregate"],
        route_fn=f"{_PKG}.request_builders.resolve_preprocessing_next_stages",
        project_payload={
            "image_encoder": (
                f"{_PKG}.request_builders.project_preprocessing_to_image_encoder"
            ),
            "audio_encoder": (
                f"{_PKG}.request_builders.project_preprocessing_to_audio_encoder"
            ),
            "mm_aggregate": (
                f"{_PKG}.request_builders.project_preprocessing_to_mm_aggregate"
            ),
        },
    )


def _image_encoder_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="image_encoder",
        process=process,
        factory=f"{_PKG}.stages.create_image_encoder_executor",
        factory_args={"device": "cuda", "dtype": None},
        gpu=gpu,
        next="mm_aggregate",
        project_payload={
            "mm_aggregate": f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
        },
    )


def _audio_encoder_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="audio_encoder",
        process=process,
        factory=f"{_PKG}.stages.create_audio_encoder_executor",
        factory_args={"device": "cuda", "dtype": None},
        gpu=gpu,
        next="mm_aggregate",
        project_payload={
            "mm_aggregate": f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
        },
    )


def _aggregate_stage(*, process: str, speech_enabled: bool = False) -> StageConfig:
    # Route the merged payload to talker_ar so partial-start can fire — the
    # policy hook needs the new_request before `stream_done` arrives.
    if speech_enabled:
        return StageConfig(
            name="mm_aggregate",
            process=process,
            factory=f"{_PKG}.stages.create_aggregate_executor",
            wait_for=["preprocessing", "image_encoder", "audio_encoder"],
            wait_for_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources",
            merge_fn=f"{_PKG}.merge.merge_for_thinker",
            next=["thinker", "talker_ar"],
            route_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_next_stages",
            project_payload={
                "talker_ar": (
                    f"{_PKG}.request_builders.project_mm_aggregate_to_talker_ar"
                ),
            },
        )
    return StageConfig(
        name="mm_aggregate",
        process=process,
        factory=f"{_PKG}.stages.create_aggregate_executor",
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        wait_for_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources",
        merge_fn=f"{_PKG}.merge.merge_for_thinker",
        next="thinker",
    )


def _thinker_stage(*, gpu: int, speech_enabled: bool, process: str) -> StageConfig:
    factory_args = {"thinker_max_seq_len": 8192}
    if speech_enabled:
        factory_args["speech_enabled"] = True
    return StageConfig(
        name="thinker",
        process=process,
        factory=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
        factory_args=factory_args,
        gpu=gpu,
        runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
        next="decode",
        stream_to=["talker_ar", "decode"] if speech_enabled else ["decode"],
        route_fn=(
            f"{_PKG}.request_builders.resolve_thinker_next_stages"
            if speech_enabled
            else None
        ),
        stream_done_to_fn=(
            f"{_PKG}.request_builders.resolve_thinker_stream_done_targets"
            if speech_enabled
            else None
        ),
        project_payload={
            "decode": f"{_PKG}.request_builders.project_thinker_to_decode",
        },
    )


def _decode_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name="decode",
        process=process,
        factory=f"{_PKG}.stages.create_decode_executor",
        terminal=True,
        can_accept_stream_before_payload=True,
    )


def _talker_stage(
    *,
    gpu: int,
    process: str,
    enable_partial_start: bool,
) -> StageConfig:
    return StageConfig(
        name="talker_ar",
        process=process,
        factory=f"{_PKG}.stages.create_talker_ar_executor_from_config",
        factory_args={
            # Note (Xuesong): must exceed talker_max_new_tokens (4096) +
            # prefill, else req_to_token_pool OOBs and crashes talker_ar.
            # Note (Chenyang): bumped 8192 → 32768 because the V1 talker
            # prefill replays the full thinker prompt as projected
            # embeddings, and a 30-frame video prompt is ~22K positions,
            # which overflows 8192 and triggers a FusedAddRMSNorm illegal
            # memory access in the talker forward.
            "talker_max_seq_len": 32768,
            "speech_enabled": True,
            "feedback_enabled": True,
            "enable_partial_start": enable_partial_start,
            "partial_start_min_chunks": 5,
        },
        gpu=gpu,
        runtime_arg_map={"max_seq_len": "talker_max_seq_len"},
        next="code2wav",
        stream_to=["code2wav"],
        project_payload={
            "code2wav": f"{_PKG}.request_builders.project_talker_to_code2wav",
        },
        can_accept_stream_before_payload=True,
    )


def _code2wav_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="code2wav",
        process=process,
        factory=f"{_PKG}.components.code2wav_scheduler.create_code2wav_scheduler",
        factory_args={"device": "cuda"},
        gpu=gpu,
        terminal=True,
        can_accept_stream_before_payload=True,
    )


def _text_stages() -> list[StageConfig]:
    return [
        _preprocessing_stage(process="pipeline"),
        _image_encoder_stage(gpu=0, process="pipeline"),
        _audio_encoder_stage(gpu=0, process="pipeline"),
        _aggregate_stage(process="pipeline", speech_enabled=False),
        _thinker_stage(gpu=0, speech_enabled=False, process="pipeline"),
        _decode_stage(process="pipeline"),
    ]


def _speech_stages(
    *,
    thinker_gpu: int,
    talker_gpu: int,
    process_by_stage: dict[str, str],
    enable_partial_start: bool,
) -> list[StageConfig]:
    return [
        _preprocessing_stage(process=process_by_stage["preprocessing"]),
        _image_encoder_stage(
            gpu=thinker_gpu,
            process=process_by_stage["image_encoder"],
        ),
        _audio_encoder_stage(
            gpu=thinker_gpu,
            process=process_by_stage["audio_encoder"],
        ),
        _aggregate_stage(
            process=process_by_stage["mm_aggregate"],
            speech_enabled=True,
        ),
        _thinker_stage(
            gpu=thinker_gpu,
            speech_enabled=True,
            process=process_by_stage["thinker"],
        ),
        _decode_stage(process=process_by_stage["decode"]),
        _talker_stage(
            gpu=talker_gpu,
            process=process_by_stage["talker_ar"],
            enable_partial_start=enable_partial_start,
        ),
        _code2wav_stage(gpu=talker_gpu, process=process_by_stage["code2wav"]),
    ]


_SPEECH_DEFAULT_PROCESSES = {
    "preprocessing": "preprocessing",
    "image_encoder": "image_encoder",
    "audio_encoder": "audio_encoder",
    "mm_aggregate": "mm_aggregate",
    "thinker": "thinker",
    "decode": "decode",
    "talker_ar": "talker_ar",
    "code2wav": "code2wav",
}


class Qwen3OmniPipelineConfig(PipelineConfig):
    """6-stage text-only pipeline."""

    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"
    env_defaults: dict[str, str] = Field(
        default_factory=lambda: dict(_DEEPGEMM_PRECOMPILE_ENV_DEFAULTS)
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker"}

    @classmethod
    def encoder_mem_reserve_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker"}

    model_path: str
    placement_policy: str | None = _PLACEMENT_POLICY
    stages: list[StageConfig] = Field(default_factory=_text_stages)


class Qwen3OmniSpeechPipelineConfig(PipelineConfig):
    """8-stage speech pipeline (text + audio output)."""

    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"
    env_defaults: dict[str, str] = Field(
        default_factory=lambda: dict(_DEEPGEMM_PRECOMPILE_ENV_DEFAULTS)
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker", "talker": "talker_ar"}

    @classmethod
    def encoder_mem_reserve_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker"}

    @classmethod
    def talker_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "talker_ar"}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "talker_ar"}

    @classmethod
    def code2wav_stage(cls) -> str | None:
        return "code2wav"

    model_path: str
    placement_policy: str | None = _PLACEMENT_POLICY
    terminal_stages_fn: str | None = f"{_PKG}.request_builders.resolve_terminal_stages"
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(
        default_factory=lambda: _speech_stages(
            thinker_gpu=0,
            talker_gpu=1,
            process_by_stage=_SPEECH_DEFAULT_PROCESSES,
            enable_partial_start=True,
        )
    )


class Qwen3OmniSpeechColocatedPipelineConfig(Qwen3OmniSpeechPipelineConfig):
    """8-stage speech pipeline for single-GPU stage colocation.

    The topology places image_encoder, audio_encoder, thinker, talker_ar, and
    code2wav on the same GPU while keeping preprocessing, aggregation, and
    decode as CPU stages. Runtime memory budgets are supplied by the selected
    config file so deployments can use hardware-appropriate stage fractions and
    SGLang AR cache fractions.
    """

    stages: list[StageConfig] = Field(
        default_factory=lambda: _speech_stages(
            thinker_gpu=0,
            talker_gpu=0,
            process_by_stage=_SPEECH_DEFAULT_PROCESSES,
            enable_partial_start=False,
        )
    )


EntryClass = Qwen3OmniSpeechPipelineConfig

Variants = {
    "text": Qwen3OmniPipelineConfig,
    "speech": Qwen3OmniSpeechPipelineConfig,
    "speech-colocated": Qwen3OmniSpeechColocatedPipelineConfig,
}
