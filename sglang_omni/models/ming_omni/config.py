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
    IMAGE_STAGE,
    PREPROCESSING_STAGE,
    TALKER_STAGE,
    THINKER_STAGE,
)


class MingOmniPipelineConfig(PipelineConfig):
    """6-stage text/vision pipeline for Ming-Omni.

    preprocessing → audio_encoder + image_encoder → mm_aggregate → thinker → decode
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
            name=IMAGE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_image_encoder_executor",
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
                sources=[PREPROCESSING_STAGE, AUDIO_STAGE, IMAGE_STAGE],
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

    def __init__(self, **kwargs):
        # Extract server_args_overrides and inject into thinker stage
        server_args_overrides = kwargs.pop("server_args_overrides", None)
        super().__init__(**kwargs)
        if server_args_overrides:
            for stage in self.stages:
                if stage.name == THINKER_STAGE:
                    if stage.executor.args is None:
                        stage.executor.args = {}
                    existing = stage.executor.args.setdefault(
                        "server_args_overrides", {}
                    )
                    existing.update(server_args_overrides)


class MingOmniSpeechPipelineConfig(PipelineConfig):
    """7-stage pipeline for Ming-Omni with text + speech output.

    Adds a talker stage that generates audio from thinker's decoded text.
    The talker is a self-contained MingOmniTalker (own LLM + CFM + AudioVAE).
    """

    architecture: ClassVar[str] = "BailingMM2NativeForConditionalGeneration"

    model_path: str
    entry_stage: str = "preprocessing"
    terminal_stages: list[str] = [DECODE_STAGE, TALKER_STAGE]
    gpu_placement: dict[str, int] = {
        "thinker": 0,
        "talker": 1,
    }

    def __init__(self, **kwargs):
        server_args_overrides = kwargs.pop("server_args_overrides", None)
        super().__init__(**kwargs)
        if server_args_overrides:
            for stage in self.stages:
                if stage.name == THINKER_STAGE:
                    if stage.executor.args is None:
                        stage.executor.args = {}
                    existing = stage.executor.args.setdefault(
                        "server_args_overrides", {}
                    )
                    existing.update(server_args_overrides)

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
            name=IMAGE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_image_encoder_executor",
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
                sources=[PREPROCESSING_STAGE, AUDIO_STAGE, IMAGE_STAGE],
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
