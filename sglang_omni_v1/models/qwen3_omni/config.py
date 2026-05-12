# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-Omni."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni_v1.config import PipelineConfig, StageConfig

_PKG = "sglang_omni_v1.models.qwen3_omni"


class Qwen3OmniPipelineConfig(PipelineConfig):
    """6-stage text-only pipeline."""

    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker"}

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"thinker_max_seq_len": 8192},
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
        ),
        StageConfig(
            name="image_encoder",
            factory=f"{_PKG}.stages.create_image_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            next="mm_aggregate",
            project_payload={
                "mm_aggregate": (
                    f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
                )
            },
        ),
        StageConfig(
            name="audio_encoder",
            factory=f"{_PKG}.stages.create_audio_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            next="mm_aggregate",
            project_payload={
                "mm_aggregate": (
                    f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
                )
            },
        ),
        StageConfig(
            name="mm_aggregate",
            factory=f"{_PKG}.stages.create_aggregate_executor",
            wait_for=["preprocessing", "image_encoder", "audio_encoder"],
            merge_fn=f"{_PKG}.merge.merge_for_thinker",
            next="thinker",
        ),
        StageConfig(
            name="thinker",
            factory=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
            factory_args={"thinker_max_seq_len": 8192},
            gpu=0,
            next="decode",
        ),
        StageConfig(
            name="decode",
            factory=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
    ]


class Qwen3OmniSpeechPipelineConfig(PipelineConfig):
    """8-stage speech pipeline (text + audio output)."""

    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker", "talker": "talker_ar"}

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"thinker_max_seq_len": 8192},
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
        ),
        StageConfig(
            name="image_encoder",
            factory=f"{_PKG}.stages.create_image_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            next="mm_aggregate",
            project_payload={
                "mm_aggregate": (
                    f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
                )
            },
        ),
        StageConfig(
            name="audio_encoder",
            factory=f"{_PKG}.stages.create_audio_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            next="mm_aggregate",
            project_payload={
                "mm_aggregate": (
                    f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
                )
            },
        ),
        StageConfig(
            name="mm_aggregate",
            factory=f"{_PKG}.stages.create_aggregate_executor",
            wait_for=["preprocessing", "image_encoder", "audio_encoder"],
            merge_fn=f"{_PKG}.merge.merge_for_thinker",
            next="thinker",
        ),
        StageConfig(
            name="thinker",
            factory=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
            factory_args={"thinker_max_seq_len": 8192, "speech_enabled": True},
            gpu=0,
            next=["decode", "talker_ar"],
            stream_to=["talker_ar"],
        ),
        StageConfig(
            name="decode",
            factory=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
        StageConfig(
            name="talker_ar",
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
            },
            gpu=1,
            next="code2wav",
            stream_to=["code2wav"],
        ),
        StageConfig(
            name="code2wav",
            factory=f"{_PKG}.components.code2wav_scheduler.create_code2wav_scheduler",
            factory_args={"device": "cuda"},
            gpu=1,
            terminal=True,
        ),
    ]


EntryClass = Qwen3OmniSpeechPipelineConfig

Variants = {
    "text": Qwen3OmniPipelineConfig,
    "speech": Qwen3OmniSpeechPipelineConfig,
}
