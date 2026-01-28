# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration helpers for Qwen3-Omni."""

from __future__ import annotations

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
    FRONTEND_STAGE,
    IMAGE_STAGE,
    THINKER_STAGE,
)


def create_text_first_pipeline_config(
    *,
    model_id: str,
    name: str = "qwen3_omni_text_first",
    frontend_device: str = "cpu",
    image_device: str = "cuda:3",
    audio_device: str = "cuda:3",
    thinker_device: str = "cuda:3",
    thinker_max_seq_len: int = 8192,
    dtype: str | None = None,
    relay_type: str = "shm",
    fused_stages: list[list[str]] | None = None,
) -> PipelineConfig:
    def _relay(device: str) -> RelayConfig:
        return RelayConfig(type=relay_type, device=device)

    return PipelineConfig(
        name=name,
        entry_stage=FRONTEND_STAGE,
        fused_stages=fused_stages or [],
        stages=[
            StageConfig(
                name=FRONTEND_STAGE,
                executor=ExecutorConfig(
                    factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_frontend_executor",
                    args={"model_id": model_id},
                ),
                get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.frontend_next",
                relay=_relay(frontend_device),
            ),
            StageConfig(
                name=IMAGE_STAGE,
                executor=ExecutorConfig(
                    factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_image_encoder_executor",
                    args={
                        "model_id": model_id,
                        "device": image_device,
                        "dtype": dtype,
                    },
                ),
                get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
                relay=_relay(image_device),
            ),
            StageConfig(
                name=AUDIO_STAGE,
                executor=ExecutorConfig(
                    factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_audio_encoder_executor",
                    args={
                        "model_id": model_id,
                        "device": audio_device,
                        "dtype": dtype,
                    },
                ),
                get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
                relay=_relay(audio_device),
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
                    sources=[FRONTEND_STAGE, IMAGE_STAGE, AUDIO_STAGE],
                    merge_fn="sglang_omni.models.qwen3_omni.pipeline.merge.merge_for_thinker",
                ),
                relay=_relay("cpu"),
            ),
            StageConfig(
                name=THINKER_STAGE,
                executor=ExecutorConfig(
                    factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_thinker_executor",
                    args={
                        "model_id": model_id,
                        "device": thinker_device,
                        "dtype": dtype,
                        "max_seq_len": thinker_max_seq_len,
                    },
                ),
                get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.thinker_next",
                relay=_relay(thinker_device),
            ),
            StageConfig(
                name=DECODE_STAGE,
                executor=ExecutorConfig(
                    factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_decode_executor",
                    args={"model_id": model_id},
                ),
                get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.decode_next",
                relay=_relay("cpu"),
            ),
        ],
    )
