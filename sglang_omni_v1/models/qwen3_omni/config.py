# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-Omni."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni_v1.config import PipelineConfig, ProcessConfig, StageConfig

_PKG = "sglang_omni_v1.models.qwen3_omni"
_PLACEMENT_POLICY = f"{_PKG}.placement.Qwen3OmniPlacementPolicy"


def _preprocessing_stage() -> StageConfig:
    return StageConfig(
        name="preprocessing",
        factory=f"{_PKG}.stages.create_preprocessing_executor",
        factory_args={"thinker_max_seq_len": 8192},
        runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
        next=["image_encoder", "audio_encoder", "mm_aggregate"],
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


def _image_encoder_stage(*, gpu: int) -> StageConfig:
    return StageConfig(
        name="image_encoder",
        factory=f"{_PKG}.stages.create_image_encoder_executor",
        factory_args={"device": "cuda", "dtype": None},
        gpu=gpu,
        next="mm_aggregate",
        project_payload={
            "mm_aggregate": f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
        },
    )


def _audio_encoder_stage(*, gpu: int) -> StageConfig:
    return StageConfig(
        name="audio_encoder",
        factory=f"{_PKG}.stages.create_audio_encoder_executor",
        factory_args={"device": "cuda", "dtype": None},
        gpu=gpu,
        next="mm_aggregate",
        project_payload={
            "mm_aggregate": f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
        },
    )


def _aggregate_stage() -> StageConfig:
    return StageConfig(
        name="mm_aggregate",
        factory=f"{_PKG}.stages.create_aggregate_executor",
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        merge_fn=f"{_PKG}.merge.merge_for_thinker",
        next="thinker",
    )


def _thinker_stage(*, gpu: int, speech_enabled: bool) -> StageConfig:
    factory_args = {"thinker_max_seq_len": 8192}
    if speech_enabled:
        factory_args["speech_enabled"] = True
    return StageConfig(
        name="thinker",
        factory=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
        factory_args=factory_args,
        gpu=gpu,
        runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
        next=["decode", "talker_ar"] if speech_enabled else "decode",
        stream_to=["talker_ar"] if speech_enabled else [],
    )


def _decode_stage() -> StageConfig:
    return StageConfig(
        name="decode",
        factory=f"{_PKG}.stages.create_decode_executor",
        terminal=True,
    )


def _talker_stage(*, gpu: int) -> StageConfig:
    return StageConfig(
        name="talker_ar",
        factory=f"{_PKG}.stages.create_talker_ar_executor_from_config",
        factory_args={
            "talker_max_seq_len": 32768,
            "speech_enabled": True,
            "feedback_enabled": True,
        },
        gpu=gpu,
        runtime_arg_map={"max_seq_len": "talker_max_seq_len"},
        next="code2wav",
        stream_to=["code2wav"],
    )


def _code2wav_stage(*, gpu: int) -> StageConfig:
    return StageConfig(
        name="code2wav",
        factory=f"{_PKG}.components.code2wav_scheduler.create_code2wav_scheduler",
        factory_args={"device": "cuda"},
        gpu=gpu,
        terminal=True,
    )


def _text_stages() -> list[StageConfig]:
    return [
        _preprocessing_stage(),
        _image_encoder_stage(gpu=0),
        _audio_encoder_stage(gpu=0),
        _aggregate_stage(),
        _thinker_stage(gpu=0, speech_enabled=False),
        _decode_stage(),
    ]


def _speech_stages(*, thinker_gpu: int, talker_gpu: int) -> list[StageConfig]:
    return [
        _preprocessing_stage(),
        _image_encoder_stage(gpu=thinker_gpu),
        _audio_encoder_stage(gpu=thinker_gpu),
        _aggregate_stage(),
        _thinker_stage(gpu=thinker_gpu, speech_enabled=True),
        _decode_stage(),
        _talker_stage(gpu=talker_gpu),
        _code2wav_stage(gpu=talker_gpu),
    ]


class Qwen3OmniPipelineConfig(PipelineConfig):
    """6-stage text-only pipeline."""

    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker"}

    model_path: str
    placement_policy: str | None = _PLACEMENT_POLICY
    stages: list[StageConfig] = Field(default_factory=_text_stages)


class Qwen3OmniSpeechPipelineConfig(PipelineConfig):
    """8-stage speech pipeline (text + audio output)."""

    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker", "talker": "talker_ar"}

    model_path: str
    placement_policy: str | None = _PLACEMENT_POLICY
    stages: list[StageConfig] = Field(
        default_factory=lambda: _speech_stages(thinker_gpu=0, talker_gpu=1)
    )


class Qwen3OmniSpeechColocatedPipelineConfig(Qwen3OmniSpeechPipelineConfig):
    """8-stage speech pipeline with all GPU stages placed on one GPU.

    This class intentionally does not provide default memory fractions. A
    launch config must set measured ``total_gpu_memory_fraction`` values for
    every GPU stage and SGLang ``mem_fraction_static`` for both AR stages.
    """

    process: ProcessConfig = Field(default_factory=lambda: ProcessConfig(mode="multi"))
    stages: list[StageConfig] = Field(
        default_factory=lambda: _speech_stages(thinker_gpu=0, talker_gpu=0)
    )


EntryClass = Qwen3OmniSpeechPipelineConfig

Variants = {
    "text": Qwen3OmniPipelineConfig,
    "speech": Qwen3OmniSpeechPipelineConfig,
    "speech-colocated": Qwen3OmniSpeechColocatedPipelineConfig,
}
