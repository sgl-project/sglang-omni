# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Ming-Omni."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    ExecutorConfig,
    InputHandlerConfig,
    PipelineConfig,
    RelayConfig,
    StageConfig,
)
from sglang_omni.models.ming_omni.pipeline.next_stage import (
    AGGREGATE_STAGE,
    AUDIO_STAGE,
    DECODE_STAGE,
    PREPROCESSING_STAGE,
    TALKER_STAGE,
    THINKER_STAGE,
)


class MingOmniPipelineConfig(PipelineConfig):
    """6-stage text-only pipeline for Ming-Omni.

    preprocessing → audio_encoder → mm_aggregate → thinker → decode
    """

    architecture: ClassVar[str] = "BailingMM2NativeForConditionalGeneration"

    model_path: str
    entry_stage: str = "preprocessing"
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_preprocessing_executor",
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=AUDIO_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_audio_encoder_executor",
                args={
                    "device": "cuda",
                    "dtype": None,
                },
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=AGGREGATE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_aggregate_executor",
                args={},
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.aggregate_next",
            input_handler=InputHandlerConfig(
                type="aggregated",
                sources=[PREPROCESSING_STAGE, AUDIO_STAGE],
                merge_fn="sglang_omni.models.ming_omni.pipeline.merge.merge_for_thinker",
            ),
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=THINKER_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_sglang_thinker_executor_from_config",
                args={
                    "thinker_max_seq_len": 8192,
                },
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.thinker_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=DECODE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_decode_executor",
                args={},
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.decode_next",
            relay=RelayConfig(device="cpu"),
        ),
    ]


class MingOmniSpeechPipelineConfig(PipelineConfig):
    """7-stage pipeline for Ming-Omni with text + speech output.

    Adds a talker stage that generates audio from thinker's decoded text.
    The talker is a self-contained BailingTalker2 (own LLM + CFM + AudioVAE).
    """

    architecture: ClassVar[str] = "BailingMM2NativeForConditionalGeneration"

    model_path: str
    entry_stage: str = "preprocessing"
    terminal_stages: list[str] = [DECODE_STAGE, TALKER_STAGE]
    gpu_placement: dict[str, int] = {
        "thinker": 0,
        "talker": 1,
    }

    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_preprocessing_executor",
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=AUDIO_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_audio_encoder_executor",
                args={"device": "cuda", "dtype": None},
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=AGGREGATE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_aggregate_executor",
                args={},
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.aggregate_next",
            input_handler=InputHandlerConfig(
                type="aggregated",
                sources=[PREPROCESSING_STAGE, AUDIO_STAGE],
                merge_fn="sglang_omni.models.ming_omni.pipeline.merge.merge_for_thinker",
            ),
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=THINKER_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_sglang_thinker_executor_from_config",
                args={"thinker_max_seq_len": 8192},
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.thinker_next_speech",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=DECODE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_decode_executor",
                args={},
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.decode_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=TALKER_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_talker_executor",
                args={
                    "device": "cuda",
                    "voice": "DB30",
                },
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.talker_next",
            relay=RelayConfig(device="cuda"),
        ),
    ]


EntryClass = MingOmniPipelineConfig
