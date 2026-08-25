# SPDX-License-Identifier: Apache-2.0
"""Stage factory for SGLang-backed Whisper ASR inference."""

from __future__ import annotations

from typing import Any


def create_sglang_whisper_asr_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "float16",
    max_running_requests: int = 64,
    max_new_tokens: int = 256,
    mem_fraction_static: float = 0.85,
    enable_encoder_cuda_graph: bool = False,
    encoder_graph_batch_buckets: list[int] | None = None,
    request_build_max_workers: int = 8,
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 2,
    request_build_max_pending: int | None = 16,
    prefill_coalesce_requests: int = 2,
    prefill_coalesce_wait_ms: float = 6.0,
    prefill_coalesce_when_idle: bool = True,
    prefill_coalesce_requires_pending_builds: bool = True,
    prefill_coalesce_after_builds_during_decode: bool = False,
    enable_pre_lm_encoder: bool = True,
    pre_lm_cache_max_entries: int = 1024,
    pre_lm_cache_size_bytes: int | None = None,
    pre_lm_max_batch_size: int = 8,
    pre_lm_max_batch_wait_ms: int = 0,
    pre_lm_cache_pin_host_memory: bool = True,
    server_args_overrides: dict[str, Any] | None = None,
):
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    return WhisperASREngineBuilder(
        max_running_requests=max_running_requests,
        mem_fraction_static=mem_fraction_static,
        max_new_tokens=max_new_tokens,
        enable_encoder_cuda_graph=enable_encoder_cuda_graph,
        encoder_graph_batch_buckets=encoder_graph_batch_buckets,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        request_build_max_workers=request_build_max_workers,
        request_build_max_pending=request_build_max_pending,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
        prefill_coalesce_when_idle=prefill_coalesce_when_idle,
        prefill_coalesce_requires_pending_builds=(
            prefill_coalesce_requires_pending_builds
        ),
        prefill_coalesce_after_builds_during_decode=(
            prefill_coalesce_after_builds_during_decode
        ),
        enable_pre_lm_encoder=enable_pre_lm_encoder,
        pre_lm_cache_max_entries=pre_lm_cache_max_entries,
        pre_lm_cache_size_bytes=pre_lm_cache_size_bytes,
        pre_lm_max_batch_size=pre_lm_max_batch_size,
        pre_lm_max_batch_wait_ms=pre_lm_max_batch_wait_ms,
        pre_lm_cache_pin_host_memory=pre_lm_cache_pin_host_memory,
    ).build(
        model_path,
        device=device,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_whisper_asr_executor(*args, **kwargs):
    return create_sglang_whisper_asr_executor(*args, **kwargs)


__all__ = ["create_sglang_whisper_asr_executor", "create_whisper_asr_executor"]
