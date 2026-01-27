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
from sglang_omni.models.adapter_registry import register_adapter
from sglang_omni.models.qwen3_omni.adapter import Qwen3OmniAdapter


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
) -> PipelineConfig:
    adapter = register_adapter(Qwen3OmniAdapter(model_id=model_id))
    adapter_name = adapter.name

    def _relay(device: str) -> RelayConfig:
        return RelayConfig(type=relay_type, device=device)

    return PipelineConfig(
        name=name,
        entry_stage="frontend",
        stages=[
            StageConfig(
                name="frontend",
                executor=ExecutorConfig(
                    factory="sglang_omni.models.qwen3_omni.frontend.create_frontend_executor",
                    args={"model_id": model_id, "adapter_name": adapter_name},
                ),
                get_next="sglang_omni.models.omni_generic.frontend_next",
                relay=_relay(frontend_device),
            ),
            StageConfig(
                name="image_encoder",
                executor=ExecutorConfig(
                    factory="sglang_omni.models.qwen3_omni.image_encoder.create_image_encoder_executor",
                    args={
                        "model_id": model_id,
                        "adapter_name": adapter_name,
                        "device": image_device,
                        "dtype": dtype,
                    },
                ),
                get_next="sglang_omni.models.omni_generic.encoder_next",
                relay=_relay(image_device),
            ),
            StageConfig(
                name="audio_encoder",
                executor=ExecutorConfig(
                    factory="sglang_omni.models.qwen3_omni.audio_encoder.create_audio_encoder_executor",
                    args={
                        "model_id": model_id,
                        "adapter_name": adapter_name,
                        "device": audio_device,
                        "dtype": dtype,
                    },
                ),
                get_next="sglang_omni.models.omni_generic.encoder_next",
                relay=_relay(audio_device),
            ),
            StageConfig(
                name="mm_aggregate",
                executor=ExecutorConfig(
                    factory="sglang_omni.models.qwen3_omni.adapter.create_aggregate_executor",
                    args={"adapter_name": adapter_name},
                ),
                get_next="sglang_omni.models.omni_generic.aggregate_next",
                input_handler=InputHandlerConfig(
                    type="aggregated",
                    sources=["frontend", "image_encoder", "audio_encoder"],
                    merge_fn="sglang_omni.models.omni_generic.merge_for_adapter",
                ),
                relay=_relay("cpu"),
            ),
            StageConfig(
                name="thinker",
                executor=ExecutorConfig(
                    factory="sglang_omni.models.qwen3_omni.thinker.create_thinker_executor",
                    args={
                        "model_id": model_id,
                        "adapter_name": adapter_name,
                        "device": thinker_device,
                        "dtype": dtype,
                        "max_seq_len": thinker_max_seq_len,
                    },
                ),
                get_next="sglang_omni.models.omni_generic.thinker_next",
                relay=_relay(thinker_device),
            ),
            StageConfig(
                name="decode",
                executor=ExecutorConfig(
                    factory="sglang_omni.models.qwen3_omni.adapter.create_decode_executor",
                    args={"model_id": model_id, "adapter_name": adapter_name},
                ),
                get_next="sglang_omni.models.omni_generic.decode_next",
                relay=_relay("cpu"),
            ),
        ],
    )
