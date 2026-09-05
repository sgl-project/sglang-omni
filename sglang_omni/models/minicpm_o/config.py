# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MiniCPM-o."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.minicpm_o"
THINKER_STAGE = "thinker"

# Note (ruoyu): every stage below used to share one "pipeline" process. The
# CPU-bound HF processor call in preprocessing and the eager vocoder in
# code2wav then competed with the thinker scheduler loop and the encoder
# dispatch for the same interpreter, and the profiler showed the audio
# encoder queue (not the thinker) owning ~65% of text-only latency at c10.
# Mirror qwen3_omni: one process per non-trivial stage; the thinker keeps
# mm_aggregate and decode (both trivial) so the merged prompt never crosses
# a process boundary again.
_PREPROCESSING_PROCESS = "preprocessing"
_IMAGE_ENCODER_PROCESS = "image_encoder"
_AUDIO_ENCODER_PROCESS = "audio_encoder"
_THINKER_PROCESS = "pipeline"
_TALKER_PROCESS = "talker"
_CODE2WAV_PROCESS = "code2wav"

# Each stage process would otherwise size its OpenMP pool to the whole host,
# and the tokenizer's Rayon pool only adds contention for one-prompt-per-call
# preprocessing (qwen3_omni colocated defaults).
_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "8",
    "TOKENIZERS_PARALLELISM": "false",
}

# Preprocessing is CPU work (whisper mel extraction, image slicing, chat
# template). SimpleScheduler runs N worker threads when max_concurrency > 1;
# the HF processor is stateless, so the calls are re-entrant.
PREPROCESSING_MAX_CONCURRENCY = 4
# The single-shot vocoder is launch-bound; two workers overlap one request's
# host-side work with another's GPU time on the shared stream.
CODE2WAV_MAX_CONCURRENCY = 2


def _preprocessing_stage(
    *, process: str, max_concurrency: int = PREPROCESSING_MAX_CONCURRENCY
) -> StageConfig:
    return StageConfig(
        name="preprocessing",
        process=process,
        factory_path=f"{_PKG}.stages.create_preprocessing_executor",
        factory=FactoryArgs(max_concurrency=max_concurrency),
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
        factory_path=f"{_PKG}.stages.create_image_encoder_executor",
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
        factory_path=f"{_PKG}.stages.create_audio_encoder_executor",
        gpu=gpu,
        disable_direct_cuda_ipc_payload=True,
        next="mm_aggregate",
        project_payload={
            "mm_aggregate": f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
        },
    )


def _aggregate_stage(*, process: str, gpu: int) -> StageConfig:
    return StageConfig(
        name="mm_aggregate",
        process=process,
        factory_path=f"{_PKG}.stages.create_aggregate_executor",
        gpu=gpu,
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        wait_for_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources",
        merge_fn=f"{_PKG}.merge.merge_for_thinker",
        next="thinker",
        disable_direct_cuda_ipc_payload=True,
    )


def _thinker_stage(
    *, gpu: int, process: str, speech_enabled: bool = False
) -> StageConfig:
    return EngineStageConfig(
        name="thinker",
        process=process,
        factory_path=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
        factory=FactoryArgs(max_seq_len=8192, enable_async_decode=True),
        gpu=gpu,
        next=["decode", "talker"] if speech_enabled else "decode",
        route_fn=(
            f"{_PKG}.request_builders.resolve_thinker_next_stages"
            if speech_enabled
            else None
        ),
        stream_to=["decode"],
        project_payload={
            "decode": f"{_PKG}.request_builders.project_thinker_to_decode",
            **(
                {"talker": f"{_PKG}.request_builders.project_thinker_to_talker"}
                if speech_enabled
                else {}
            ),
        },
    )


def _decode_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name="decode",
        process=process,
        factory_path=f"{_PKG}.stages.create_decode_executor",
        terminal=True,
        can_accept_stream_before_payload=True,
    )


def _talker_stage(*, gpu: int, process: str) -> StageConfig:
    return EngineStageConfig(
        name="talker",
        process=process,
        factory_path=f"{_PKG}.stages.create_sglang_talker_executor_from_config",
        factory=FactoryArgs(max_seq_len=4096),
        gpu=gpu,
        next="code2wav",
        project_payload={
            "code2wav": f"{_PKG}.request_builders.project_talker_to_code2wav",
        },
    )


def _code2wav_stage(
    *, gpu: int, process: str, max_concurrency: int = CODE2WAV_MAX_CONCURRENCY
) -> StageConfig:
    return StageConfig(
        name="code2wav",
        process=process,
        factory_path=f"{_PKG}.stages.create_code2wav_executor",
        factory=FactoryArgs(max_concurrency=max_concurrency),
        gpu=gpu,
        terminal=True,
    )


def _default_stages() -> list[StageConfig]:
    return [
        _preprocessing_stage(process=_PREPROCESSING_PROCESS),
        _image_encoder_stage(process=_IMAGE_ENCODER_PROCESS, gpu=0),
        _audio_encoder_stage(process=_AUDIO_ENCODER_PROCESS, gpu=0),
        _aggregate_stage(process=_THINKER_PROCESS, gpu=0),
        _thinker_stage(gpu=0, process=_THINKER_PROCESS),
        _decode_stage(process=_THINKER_PROCESS),
    ]


def _speech_stages() -> list[StageConfig]:
    return [
        _preprocessing_stage(process=_PREPROCESSING_PROCESS),
        _image_encoder_stage(process=_IMAGE_ENCODER_PROCESS, gpu=0),
        _audio_encoder_stage(process=_AUDIO_ENCODER_PROCESS, gpu=0),
        _aggregate_stage(process=_THINKER_PROCESS, gpu=0),
        _thinker_stage(gpu=0, process=_THINKER_PROCESS, speech_enabled=True),
        _decode_stage(process=_THINKER_PROCESS),
        # The sglang talker is a second engine; it cannot share the thinker's
        # process (one torch tensor-parallel group per process).
        _talker_stage(gpu=0, process=_TALKER_PROCESS),
        _code2wav_stage(gpu=0, process=_CODE2WAV_PROCESS),
    ]


class MiniCPMOPipelineConfig(PipelineConfig):
    """Thinker pipeline: preprocessing → [image/audio encoders] → mm_aggregate
    → thinker → decode."""

    architecture: ClassVar[str] = "MiniCPMO"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = Field(default_factory=_default_stages)
    env_defaults: dict[str, str] = Field(default_factory=lambda: dict(_ENV_DEFAULTS))

    # The encoders (and, in the speech variant, the talker engine and the
    # vocoder) run in their own processes on the thinker's GPU; skip per-stage
    # memory-fraction budgets like qwen3_omni does and let sglang's
    # mem_fraction_static govern each engine.
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )


class MiniCPMOSpeechPipelineConfig(MiniCPMOPipelineConfig):
    """Speech pipeline: text stages + talker (native sglang AR stage) +
    code2wav (stepaudio2 Token2wav). Audio output arrives non-streaming, one
    wav per request."""

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
        "talker": EngineStageConfig,
    }

    terminal_stages_fn: str | None = f"{_PKG}.request_builders.resolve_terminal_stages"
    stages: list[StageConfig] = Field(default_factory=_speech_stages)

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name in (THINKER_STAGE, "preprocessing"):
            return {"speech_enabled": True}
        return {}


EntryClass = MiniCPMOSpeechPipelineConfig

Variants = {
    "text": MiniCPMOPipelineConfig,
    "speech": MiniCPMOSpeechPipelineConfig,
}
