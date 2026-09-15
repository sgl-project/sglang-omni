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


def _preprocessing_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name="preprocessing",
        process=process,
        factory_path=f"{_PKG}.stages.create_preprocessing_executor",
        next=["image_encoder", "audio_encoder", "thinker"],
        route_fn=f"{_PKG}.request_builders.resolve_preprocessing_next_stages",
        project_payload={
            "image_encoder": (
                f"{_PKG}.request_builders.project_preprocessing_to_image_encoder"
            ),
            "audio_encoder": (
                f"{_PKG}.request_builders.project_preprocessing_to_audio_encoder"
            ),
            "thinker": (f"{_PKG}.request_builders.project_preprocessing_to_thinker"),
        },
    )


def _image_encoder_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="image_encoder",
        process=process,
        factory_path=f"{_PKG}.stages.create_image_encoder_executor",
        gpu=gpu,
        next="thinker",
        project_payload={
            "thinker": f"{_PKG}.request_builders.project_encoder_to_thinker"
        },
    )


def _audio_encoder_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="audio_encoder",
        process=process,
        factory_path=f"{_PKG}.stages.create_audio_encoder_executor",
        gpu=gpu,
        disable_direct_cuda_ipc_payload=True,
        next="thinker",
        project_payload={
            "thinker": f"{_PKG}.request_builders.project_encoder_to_thinker"
        },
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
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        wait_for_fn=f"{_PKG}.request_builders.resolve_thinker_wait_sources",
        merge_fn=f"{_PKG}.merge.merge_for_thinker",
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


def _code2wav_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="code2wav",
        process=process,
        factory_path=f"{_PKG}.stages.create_code2wav_executor",
        gpu=gpu,
        terminal=True,
    )


def _default_stages() -> list[StageConfig]:
    return [
        _preprocessing_stage(process="pipeline"),
        # Config order is construction order inside one process. The thinker's
        # sglang engine must initialize the tensor-parallel group first; the
        # encoders' srt vision layers then reuse it (initialize_model_parallel
        # asserts the group does not already exist).
        _thinker_stage(gpu=0, process="pipeline"),
        _image_encoder_stage(process="pipeline", gpu=0),
        _audio_encoder_stage(process="pipeline", gpu=0),
        _decode_stage(process="pipeline"),
    ]


def _speech_stages() -> list[StageConfig]:
    return [
        _preprocessing_stage(process="pipeline"),
        # Thinker before the encoders: see _default_stages.
        _thinker_stage(gpu=0, process="pipeline", speech_enabled=True),
        _image_encoder_stage(process="pipeline", gpu=0),
        _audio_encoder_stage(process="pipeline", gpu=0),
        _decode_stage(process="pipeline"),
        # The sglang talker is a second engine; it cannot share the thinker's
        # process (one torch tensor-parallel group per process).
        _talker_stage(gpu=0, process="talker"),
        # The CFM flow's 10-step iterative vocode would contend with the
        # thinker/encoders if run on the pipeline process's event loop.
        _code2wav_stage(gpu=0, process="code2wav"),
    ]


class MiniCPMOPipelineConfig(PipelineConfig):
    """Thinker pipeline: preprocessing → [image/audio encoders] → thinker
    → decode. Encoder outputs fan in directly on the thinker stage."""

    architecture: ClassVar[str] = "MiniCPMO"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = Field(default_factory=_default_stages)


class MiniCPMOSpeechPipelineConfig(MiniCPMOPipelineConfig):
    """Speech pipeline: text stages + talker (native sglang AR stage) +
    code2wav (stepaudio2 Token2wav). Audio output arrives non-streaming, one
    wav per request."""

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
        "talker": EngineStageConfig,
    }

    # The talker engine runs in a separate process on the thinker's GPU; skip
    # per-stage memory-fraction budgets like qwen3_omni does and let sglang's
    # mem_fraction_static govern each engine.
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )

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
