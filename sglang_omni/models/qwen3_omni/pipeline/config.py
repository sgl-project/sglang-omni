# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration helpers for Qwen3-Omni."""

from __future__ import annotations

from sglang_omni.config import PipelineConfig
from sglang_omni.models.omni_base.pipeline_config import (
    StageSpec,
    create_omni_pipeline_config,
)
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    AGGREGATE_STAGE,
    AUDIO_STAGE,
    DECODE_STAGE,
    IMAGE_STAGE,
    PREPROCESSING_STAGE,
    THINKER_STAGE,
)


def create_text_first_pipeline_config(
    *,
    model_id: str,
    name: str = "qwen3_omni_text_first",
    preprocessing_device: str = "cpu",
    image_device: str = "cuda:0",
    audio_device: str = "cuda:0",
    thinker_device: str = "cuda:0",
    thinker_max_seq_len: int = 8192,
    dtype: str | None = None,
    relay_type: str = "shm",
    fused_stages: list[list[str]] | None = None,
) -> PipelineConfig:
    stage_specs: list[StageSpec] = [
        {
            "name": PREPROCESSING_STAGE,
            "executor_factory": (
                "sglang_omni.models.qwen3_omni.pipeline.stages.create_preprocessing_executor"
            ),
            "executor_args": {"model_id": model_id},
            "get_next": "sglang_omni.models.qwen3_omni.pipeline.next_stage.preprocessing_next",
            "relay_device": preprocessing_device,
        },
        {
            "name": IMAGE_STAGE,
            "executor_factory": (
                "sglang_omni.models.qwen3_omni.pipeline.stages.create_image_encoder_executor"
            ),
            "executor_args": {
                "model_id": model_id,
                "device": image_device,
                "dtype": dtype,
            },
            "get_next": "sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
            "relay_device": image_device,
        },
        {
            "name": AUDIO_STAGE,
            "executor_factory": (
                "sglang_omni.models.qwen3_omni.pipeline.stages.create_audio_encoder_executor"
            ),
            "executor_args": {
                "model_id": model_id,
                "device": audio_device,
                "dtype": dtype,
            },
            "get_next": "sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
            "relay_device": audio_device,
        },
        {
            "name": AGGREGATE_STAGE,
            "executor_factory": (
                "sglang_omni.models.qwen3_omni.pipeline.stages.create_aggregate_executor"
            ),
            "executor_args": {},
            "get_next": "sglang_omni.models.qwen3_omni.pipeline.next_stage.aggregate_next",
            "input_handler": {
                "type": "aggregated",
                "sources": [PREPROCESSING_STAGE, IMAGE_STAGE, AUDIO_STAGE],
                "merge_fn": "sglang_omni.models.qwen3_omni.pipeline.merge.merge_for_thinker",
            },
            "relay_device": "cpu",
        },
        {
            "name": THINKER_STAGE,
            "executor_factory": (
                "sglang_omni.models.qwen3_omni.pipeline.stages.create_thinker_executor"
            ),
            "executor_args": {
                "model_id": model_id,
                "device": thinker_device,
                "dtype": dtype,
                "max_seq_len": thinker_max_seq_len,
            },
            "get_next": "sglang_omni.models.qwen3_omni.pipeline.next_stage.thinker_next",
            "relay_device": thinker_device,
        },
        {
            "name": DECODE_STAGE,
            "executor_factory": "sglang_omni.models.qwen3_omni.pipeline.stages.create_decode_executor",
            "executor_args": {"model_id": model_id},
            "get_next": "sglang_omni.models.qwen3_omni.pipeline.next_stage.decode_next",
            "relay_device": "cpu",
        },
    ]

    return create_omni_pipeline_config(
        name=name,
        entry_stage=PREPROCESSING_STAGE,
        stages=stage_specs,
        relay_type=relay_type,
        fused_stages=fused_stages,
    )
