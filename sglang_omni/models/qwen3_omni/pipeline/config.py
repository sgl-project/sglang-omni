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
    enable_intra_stage_overlap: bool = False,
    max_pending: int = 4,
) -> PipelineConfig:
    def _relay(device: str) -> RelayConfig:
        return RelayConfig(type=relay_type, device=device)

    # Build image encoder stage - can be overlap or regular
    if enable_intra_stage_overlap:
        image_encoder_stage = StageConfig(
            name=IMAGE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.executors.intra_stage_overlap_executor.create_intra_stage_overlap_executor",
                args={
                    "cpu_executor": {
                        "factory": "sglang_omni.models.qwen3_omni.pipeline.stages.create_frontend_executor",
                        "args": {
                            "model_id": model_id,
                            "use_thread_pool": True,
                            "max_workers": 4,
                        },
                    },
                    "gpu_executor": {
                        "factory": "sglang_omni.models.qwen3_omni.pipeline.stages.create_image_encoder_executor",
                        "args": {
                            "model_id": model_id,
                            "device": image_device,
                            "dtype": dtype,
                        },
                    },
                    "max_pending": max_pending,
                },
            ),
            # In overlap mode, IMAGE_STAGE contains frontend, so it must route to
            # both AUDIO_STAGE and AGGREGATE_STAGE (not just AGGREGATE_STAGE)
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.image_encoder_with_frontend_next",
            relay=_relay(image_device),
        )
    else:
        image_encoder_stage = StageConfig(
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
        )

    # When using intra-stage overlap, frontend is merged into image_encoder
    # so we skip the separate frontend stage
    stages = []
    if not enable_intra_stage_overlap:
        stages.append(
            StageConfig(
                name=FRONTEND_STAGE,
                executor=ExecutorConfig(
                    factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_frontend_executor",
                    args={"model_id": model_id},
                ),
                get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.frontend_next",
                relay=_relay(frontend_device),
            )
        )

    # Aggregate stage sources depend on whether we have separate frontend
    aggregate_sources = (
        [IMAGE_STAGE, AUDIO_STAGE]
        if enable_intra_stage_overlap
        else [FRONTEND_STAGE, IMAGE_STAGE, AUDIO_STAGE]
    )

    return PipelineConfig(
        name=name,
        entry_stage=IMAGE_STAGE if enable_intra_stage_overlap else FRONTEND_STAGE,
        fused_stages=fused_stages or [],
        stages=stages
        + [
            image_encoder_stage,
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
                    sources=aggregate_sources,
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
