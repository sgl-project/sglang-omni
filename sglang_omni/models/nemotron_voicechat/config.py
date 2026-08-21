# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for NVIDIA NemotronLabs VoiceChat 11B."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    RealtimeAudioConfig,
    StageConfig,
)

from .payload_types import FRAME_SAMPLES, INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE

_PKG = "sglang_omni.models.nemotron_voicechat"

PERCEPTION_STAGE = "perception"
THINKER_STAGE = "thinker"
TALKER_STAGE = "talker"
CODE2WAV_STAGE = "code2wav"


class NemotronVoiceChatPipelineConfig(PipelineConfig):
    """Frame-locked full-duplex speech-to-speech pipeline.

    Perception and codec use the checkpoint's NeMo modules. The two
    autoregressive stages use SGLang and retain a streaming KV session keyed by
    the realtime session id.
    """

    architecture: ClassVar[str] = "NemotronVoiceChatForConditionalGeneration"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
        TALKER_STAGE: EngineStageConfig,
    }
    realtime_audio: ClassVar[RealtimeAudioConfig] = RealtimeAudioConfig(
        mode="frame",
        input_sample_rate=INPUT_SAMPLE_RATE,
        output_sample_rate=OUTPUT_SAMPLE_RATE,
        frame_samples=FRAME_SAMPLES,
        max_pending_frames=256,
        max_inflight_frames=4,
        warmup_frames=2,
    )

    model_path: str
    entry_stage: str = PERCEPTION_STAGE
    stages: list[StageConfig] = [  # noqa: RUF012 - Pydantic copies model defaults
        StageConfig(
            name=PERCEPTION_STAGE,
            process=PERCEPTION_STAGE,
            factory_path=f"{_PKG}.stages.create_perception_executor",
            factory=FactoryArgs(dtype="float32"),
            gpu=0,
            gpu_memory_fraction=0.12,
            next=THINKER_STAGE,
        ),
        EngineStageConfig(
            name=THINKER_STAGE,
            process=THINKER_STAGE,
            factory_path=f"{_PKG}.stages.create_thinker_executor",
            factory=FactoryArgs(dtype="bfloat16", context_length=8192),
            engine=EngineArgs(mem_fraction_static=0.45),
            gpu=0,
            gpu_memory_fraction=0.52,
            next=TALKER_STAGE,
        ),
        EngineStageConfig(
            name=TALKER_STAGE,
            process=TALKER_STAGE,
            factory_path=f"{_PKG}.stages.create_talker_executor",
            factory=FactoryArgs(dtype="float32", context_length=8192),
            engine=EngineArgs(mem_fraction_static=0.20),
            gpu=0,
            gpu_memory_fraction=0.22,
            next=CODE2WAV_STAGE,
        ),
        StageConfig(
            name=CODE2WAV_STAGE,
            process=CODE2WAV_STAGE,
            factory_path=f"{_PKG}.stages.create_code2wav_executor",
            factory=FactoryArgs(dtype="float32"),
            gpu=0,
            gpu_memory_fraction=0.08,
            terminal=True,
        ),
    ]

    @classmethod
    def code2wav_stage(cls) -> str | None:
        return CODE2WAV_STAGE

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": THINKER_STAGE, "talker": TALKER_STAGE}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": THINKER_STAGE}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": TALKER_STAGE}

    def model_post_init(self, __context: Any = None) -> None:  # noqa: PYI063
        super().model_post_init(__context)
        expected = {
            PERCEPTION_STAGE: THINKER_STAGE,
            THINKER_STAGE: TALKER_STAGE,
            TALKER_STAGE: CODE2WAV_STAGE,
            CODE2WAV_STAGE: None,
        }
        stages = {stage.name: stage for stage in self.stages}
        if set(stages) != set(expected):
            raise ValueError(
                "Nemotron VoiceChat requires exactly perception, thinker, "
                f"talker, and code2wav stages; got {sorted(stages)}"
            )
        if self.resolved_entry_stage != PERCEPTION_STAGE:
            raise ValueError("Nemotron VoiceChat entry_stage must be 'perception'")
        for name, next_stage in expected.items():
            stage = stages[name]
            if stage.next != next_stage or stage.terminal != (next_stage is None):
                raise ValueError(
                    f"Invalid Nemotron VoiceChat topology at {name!r}: "
                    f"expected next={next_stage!r}, terminal={next_stage is None}"
                )


EntryClass = NemotronVoiceChatPipelineConfig
