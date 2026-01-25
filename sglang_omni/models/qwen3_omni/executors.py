# SPDX-License-Identifier: Apache-2.0
"""Executor factories for the Qwen3-Omni split pipeline."""

from __future__ import annotations

from transformers import AutoTokenizer

from sglang_omni.executors import EngineExecutor, FrontendExecutor
from sglang_omni.models.omni_generic import (
    create_adapter_aggregate_executor,
    create_adapter_decode_executor,
    create_adapter_encoder_executor,
    create_adapter_frontend_executor,
    create_adapter_thinker_executor,
)
from sglang_omni.models.qwen3_omni.adapter import AUDIO_STAGE, IMAGE_STAGE
from sglang_omni.models.qwen3_omni.hf_split import (
    Qwen3OmniAudioEncoder,
    Qwen3OmniImageEncoder,
    Qwen3OmniSplitThinker,
)


def create_frontend_executor(model_id: str, *, adapter_name: str) -> FrontendExecutor:
    del model_id
    return create_adapter_frontend_executor(adapter_name)


def create_image_encoder_executor(
    model_id: str,
    *,
    adapter_name: str,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3OmniImageEncoder(model_id=model_id, device=device, dtype=dtype)
    return create_adapter_encoder_executor(adapter_name, stage_name=IMAGE_STAGE, model=model)


def create_audio_encoder_executor(
    model_id: str,
    *,
    adapter_name: str,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    model = Qwen3OmniAudioEncoder(model_id=model_id, device=device, dtype=dtype)
    return create_adapter_encoder_executor(adapter_name, stage_name=AUDIO_STAGE, model=model)


def create_aggregate_executor(*, adapter_name: str) -> FrontendExecutor:
    return create_adapter_aggregate_executor(adapter_name)


def create_thinker_executor(
    model_id: str,
    *,
    adapter_name: str,
    device: str = "cuda",
    dtype: str | None = None,
    max_seq_len: int = 8192,
) -> EngineExecutor:
    model = Qwen3OmniSplitThinker(model_id=model_id, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return create_adapter_thinker_executor(
        adapter_name,
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        device=device,
    )


def create_decode_executor(model_id: str, *, adapter_name: str) -> FrontendExecutor:
    del model_id
    return create_adapter_decode_executor(adapter_name)
