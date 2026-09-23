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

PKG = "sglang_omni.models.minicpm_o"
THINKER_STAGE = "thinker"


def preprocessing_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name="preprocessing",
        process=process,
        factory_path=f"{PKG}.stages.create_preprocessing_executor",
        next=["image_encoder", "audio_encoder", "thinker"],
        route_fn=f"{PKG}.routing.resolve_preprocessing_next_stages",
        project_payload={
            "image_encoder": (f"{PKG}.routing.project_preprocessing_to_image_encoder"),
            "audio_encoder": (f"{PKG}.routing.project_preprocessing_to_audio_encoder"),
            "thinker": (f"{PKG}.routing.project_preprocessing_to_thinker"),
        },
    )


def image_encoder_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="image_encoder",
        process=process,
        factory_path=f"{PKG}.stages.create_image_encoder_executor",
        gpu=gpu,
        next="thinker",
        project_payload={"thinker": f"{PKG}.routing.project_encoder_to_thinker"},
    )


def audio_encoder_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="audio_encoder",
        process=process,
        factory_path=f"{PKG}.stages.create_audio_encoder_executor",
        gpu=gpu,
        disable_direct_cuda_ipc_payload=True,
        next="thinker",
        project_payload={"thinker": f"{PKG}.routing.project_encoder_to_thinker"},
    )


def thinker_stage(
    *, gpu: int, process: str, speech_enabled: bool = False
) -> StageConfig:
    return EngineStageConfig(
        name="thinker",
        process=process,
        factory_path=f"{PKG}.stages.create_sglang_thinker_executor_from_config",
        factory=FactoryArgs(max_seq_len=8192, enable_async_decode=True),
        gpu=gpu,
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        wait_for_fn=f"{PKG}.routing.resolve_thinker_wait_sources",
        merge_fn=f"{PKG}.merge.merge_for_thinker",
        next=["decode", "talker"] if speech_enabled else "decode",
        route_fn=(
            f"{PKG}.routing.resolve_thinker_next_stages" if speech_enabled else None
        ),
        stream_to=["decode"],
        project_payload={
            "decode": f"{PKG}.routing.project_thinker_to_decode",
            **(
                {"talker": f"{PKG}.routing.project_thinker_to_talker"}
                if speech_enabled
                else {}
            ),
        },
    )


def decode_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name="decode",
        process=process,
        factory_path=f"{PKG}.stages.create_decode_executor",
        terminal=True,
        can_accept_stream_before_payload=True,
    )


def talker_stage(*, gpu: int, process: str) -> StageConfig:
    return EngineStageConfig(
        name="talker",
        process=process,
        factory_path=f"{PKG}.stages.create_sglang_talker_executor_from_config",
        factory=FactoryArgs(max_seq_len=4096),
        gpu=gpu,
        next="code2wav",
        project_payload={
            "code2wav": f"{PKG}.routing.project_talker_to_code2wav",
        },
    )


def code2wav_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="code2wav",
        process=process,
        factory_path=f"{PKG}.stages.create_code2wav_executor",
        factory=FactoryArgs(
            max_batch_size=8,
            max_batch_wait_ms=0.0,
            batch_wait_when_idle=False,
            enable_flow_variable_length=True,
        ),
        # Note (Chenyang): As a general comment and my usual understanding
        # of SGLang Omni, SGLang Omni has a poor runtime which leads to a
        # underutilized GPU/SMs. To address this, we recommend users to set
        # batchs for your compute but never wait for grouping the batchs.
        # As SGLang Omni Runtime moves better, we shall probably wait several
        # ms for grouping the batchs, but right now, set it to 0.0.
        gpu=gpu,
        terminal=True,
    )


def text_stages() -> list[StageConfig]:
    return [
        preprocessing_stage(process="pipeline"),
        # note (MayDomine): the thinker initializes the TP group reused by encoders.
        thinker_stage(gpu=0, process="pipeline"),
        image_encoder_stage(process="pipeline", gpu=0),
        audio_encoder_stage(process="pipeline", gpu=0),
        decode_stage(process="pipeline"),
    ]


def speech_stages() -> list[StageConfig]:
    return [
        preprocessing_stage(process="pipeline"),
        # note (MayDomine): the thinker initializes the TP group reused by encoders.
        thinker_stage(gpu=0, process="pipeline", speech_enabled=True),
        image_encoder_stage(process="pipeline", gpu=0),
        audio_encoder_stage(process="pipeline", gpu=0),
        decode_stage(process="pipeline"),
        # note (MayDomine): each engine requires a separate process-global TP group.
        talker_stage(gpu=0, process="talker"),
        # note (MayDomine): vocoding must not block the thinker's event loop.
        code2wav_stage(gpu=0, process="code2wav"),
    ]


class MiniCPMOPipelineConfig(PipelineConfig):
    """Text-output pipeline with image and audio encoder fan-in."""

    architecture: ClassVar[str] = "MiniCPMO"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = Field(default_factory=text_stages)


class MiniCPMOSpeechPipelineConfig(MiniCPMOPipelineConfig):
    """Text and speech pipeline producing one waveform per request."""

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
        "talker": EngineStageConfig,
    }

    # note (MayDomine): each engine manages its own static memory fraction.
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )

    terminal_stages_fn: str | None = f"{PKG}.routing.resolve_terminal_stages"
    stages: list[StageConfig] = Field(default_factory=speech_stages)

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name in (THINKER_STAGE, "preprocessing"):
            return {"speech_enabled": True}
        return {}


EntryClass = MiniCPMOSpeechPipelineConfig

Variants = {
    "text": MiniCPMOPipelineConfig,
    "speech": MiniCPMOSpeechPipelineConfig,
}
