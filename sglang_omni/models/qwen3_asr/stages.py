# SPDX-License-Identifier: Apache-2.0
"""Stage factory for SGLang-backed Qwen3-ASR inference."""

from __future__ import annotations

from typing import Any


def create_sglang_qwen3_asr_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "auto",
    max_running_requests: int = 64,
    max_new_tokens: int = 256,
    mem_fraction_static: float | None = None,
    mm_embedding_cache_size_bytes: int = 0,
    enable_torch_compile: bool = False,
    torch_compile_max_bs: int = 1,
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 1,
    mm_attention_backend: str | None = None,
    request_build_max_workers: int = 8,
    request_build_max_pending: int | None = 32,
    prefill_coalesce_requests: int = 16,
    prefill_coalesce_wait_ms: float = 40.0,
    prefill_coalesce_when_idle: bool = True,
    prefill_coalesce_requires_pending_builds: bool = True,
    prefill_coalesce_after_builds_during_decode: bool = True,
    enable_pre_lm_encoder: bool = True,
    pre_lm_cache_max_entries: int = 4096,
    pre_lm_cache_size_bytes: int = 2 * 1024**3,
    pre_lm_max_batch_size: int = 8,
    pre_lm_max_batch_wait_ms: int = 0,
    server_args_overrides: dict[str, Any] | None = None,
):
    from sglang_omni.models.qwen3_asr.engine_builder import Qwen3ASREngineBuilder

    return Qwen3ASREngineBuilder(
        max_running_requests=max_running_requests,
        max_new_tokens=max_new_tokens,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        mem_fraction_static=mem_fraction_static,
        mm_embedding_cache_size_bytes=mm_embedding_cache_size_bytes,
        enable_torch_compile=enable_torch_compile,
        torch_compile_max_bs=torch_compile_max_bs,
        mm_attention_backend=mm_attention_backend,
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
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_qwen3_asr_executor(*args, **kwargs):
    return create_sglang_qwen3_asr_executor(*args, **kwargs)


__all__ = ["create_sglang_qwen3_asr_executor", "create_qwen3_asr_executor"]
