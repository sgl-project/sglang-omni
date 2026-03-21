# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration helpers for Qwen3-Omni."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    ExecutorConfig,
    InputHandlerConfig,
    PipelineConfig,
    RelayConfig,
    StageConfig,
)
from sglang_omni.config.schema import StreamTargetConfig
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    AGGREGATE_STAGE,
    AUDIO_STAGE,
    CODE2WAV_STAGE,
    CODE_PREDICTOR_STAGE,
    DECODE_STAGE,
    IMAGE_STAGE,
    PREPROCESSING_STAGE,
    TALKER_AR_STAGE,
    THINKER_STAGE,
)


class Qwen3OmniPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"

    model_path: str
    entry_stage: str = "preprocessing"
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_preprocessing_executor",
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=IMAGE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_image_encoder_executor",
                args={
                    "device": "cuda",
                    "dtype": None,
                },
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=AUDIO_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_audio_encoder_executor",
                args={
                    "device": "cuda",
                    "dtype": None,
                },
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=AGGREGATE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_aggregate_executor",
                args={},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.aggregate_next",
            input_handler=InputHandlerConfig(
                type="aggregated",
                sources=[PREPROCESSING_STAGE, IMAGE_STAGE, AUDIO_STAGE],
                merge_fn="sglang_omni.models.qwen3_omni.pipeline.merge.merge_for_thinker",
            ),
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=THINKER_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_sglang_thinker_executor_from_config",
                args={
                    "thinker_max_seq_len": 8192,
                },
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.thinker_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=DECODE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_decode_executor",
                args={},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.decode_next",
            relay=RelayConfig(device="cpu"),
        ),
    ]


class Qwen3OmniSpeechPipelineConfig(PipelineConfig):
    """9-stage pipeline config for Qwen3 Omni with text + speech output."""

    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"

    model_path: str
    entry_stage: str = "preprocessing"
    terminal_stages: list[str] = [DECODE_STAGE, CODE2WAV_STAGE]
    gpu_placement: dict[str, int] = {
        "thinker": 0,
        "talker_ar": 1,
        "code_predictor": 1,
        "code2wav": 1,
    }

    stages: list[StageConfig] = [
        # Stages 1-4: same as text-only
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_preprocessing_executor",
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=IMAGE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_image_encoder_executor",
                args={"device": "cuda", "dtype": None},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=AUDIO_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_audio_encoder_executor",
                args={"device": "cuda", "dtype": None},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=AGGREGATE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_aggregate_executor",
                args={},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.aggregate_next",
            input_handler=InputHandlerConfig(
                type="aggregated",
                sources=[PREPROCESSING_STAGE, IMAGE_STAGE, AUDIO_STAGE],
                merge_fn="sglang_omni.models.qwen3_omni.pipeline.merge.merge_for_thinker",
            ),
            relay=RelayConfig(device="cpu"),
        ),
        # Stage 5: Thinker (speech_enabled, fan-out)
        StageConfig(
            name=THINKER_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_sglang_thinker_executor_from_config",
                args={"thinker_max_seq_len": 8192, "speech_enabled": True},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.thinker_next_speech",
            relay=RelayConfig(device="cuda"),
            stream_to=[StreamTargetConfig(to_stage=TALKER_AR_STAGE)],
        ),
        # Stage 6: Decode (terminal)
        StageConfig(
            name=DECODE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_decode_executor",
                args={},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.decode_next",
            relay=RelayConfig(device="cpu"),
        ),
        # Stage 7: Talker AR
        StageConfig(
            name=TALKER_AR_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_talker_ar_executor_from_config",
                args={
                    "talker_max_seq_len": 4096,
                    "speech_enabled": True,
                    "feedback_enabled": True,
                },
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.talker_ar_next",
            relay=RelayConfig(device="cuda"),
            stream_to=[StreamTargetConfig(to_stage=CODE_PREDICTOR_STAGE)],
        ),
        # Stage 8: Code Predictor (streaming: consumes chunks from Talker, sends chunks to Code2Wav)
        StageConfig(
            name=CODE_PREDICTOR_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.components.code_predictor_executor.create_code_predictor_executor_from_config",
                args={"code_predictor_max_seq_len": 256},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.code_predictor_next",
            relay=RelayConfig(device="cuda"),
            stream_to=[
                StreamTargetConfig(to_stage=CODE2WAV_STAGE),
                StreamTargetConfig(to_stage=TALKER_AR_STAGE, bootstrap=False),
            ],
        ),
        # Stage 9: Code2Wav (terminal)
        StageConfig(
            name=CODE2WAV_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.components.code2wav_executor.create_code2wav_executor",
                args={"device": "cuda"},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.code2wav_next",
            relay=RelayConfig(device="cuda"),
        ),
    ]


EntryClass = Qwen3OmniPipelineConfig
