# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration helpers for Qwen3-Omni."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import (
    ExecutorConfig,
    InputHandlerConfig,
    PipelineConfig,
    RelayConfig,
    StageConfig,
)
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    AGGREGATE_STAGE,
    AUDIO_STAGE,
    DECODE_STAGE,
    IMAGE_STAGE,
    PREPROCESSING_STAGE,
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
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_thinker_executor",
                args={
                    "device": "cuda",
                    "dtype": None,
                    "max_seq_len": 8192,
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

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)

        # TODO: we need to refactor this factory pattern to avoid
        for stage in self.stages:
            if stage.name != AGGREGATE_STAGE:
                stage.executor.args["model_id"] = self.model_path


EntryClass = Qwen3OmniPipelineConfig
