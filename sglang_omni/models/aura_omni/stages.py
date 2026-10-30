# SPDX-License-Identifier: Apache-2.0
"""Stage factories owned by the AURA Omni pipeline.

The ASR and TTS stages reuse the Qwen3-ASR and Qwen3-TTS factories as they are.
"""

from __future__ import annotations

from collections.abc import Mapping

from transformers import AutoTokenizer

from sglang_omni.models.aura_omni.engine_builder import (
    AuraEngineBuilder,
    AuraRequestData,
)
from sglang_omni.models.aura_omni.request_builders import (
    DECODE_STAGE,
    build_decode_result,
    preprocess_aura_payload,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.streaming_detokenizer import StreamingDetokenizeScheduler
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint


def create_preprocessing_executor(
    *, max_concurrency: int
) -> ThreadedSimpleScheduler[StagePayload, StagePayload]:
    # Remote media is fetched here, so requests must not wait on each other.
    return ThreadedSimpleScheduler(
        preprocess_aura_payload, max_concurrency=max_concurrency
    )


def create_aura_executor(
    model_path: str,
    *,
    max_new_tokens: int,
    request_build_max_workers: int,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: Mapping[str, object] | None = None,
    total_gpu_memory_fraction: float | None = None,
) -> OmniScheduler[AuraRequestData]:
    return AuraEngineBuilder(
        max_new_tokens=max_new_tokens,
        request_build_max_workers=request_build_max_workers,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )


def create_decode_executor(model_path: str) -> StreamingDetokenizeScheduler:
    tokenizer = AutoTokenizer.from_pretrained(resolve_checkpoint(model_path))
    return StreamingDetokenizeScheduler(
        tokenizer,
        tokenizer.eos_token_id,
        build_result=build_decode_result,
        stage_name=DECODE_STAGE,
    )
