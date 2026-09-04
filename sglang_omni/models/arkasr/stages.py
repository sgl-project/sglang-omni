# SPDX-License-Identifier: Apache-2.0
"""Stage factory for SGLang-backed ARK-ASR-3B inference."""

from __future__ import annotations

from typing import Any


def create_sglang_arkasr_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_running_requests: int = 32,
    encoder_max_batch_size: int = 8,
    max_new_tokens: int = 256,
    mem_fraction_static: float | None = None,
    mm_embedding_cache_size_bytes: int = 0,
    enable_torch_compile: bool = False,
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 2,
    mm_attention_backend: str | None = None,
    request_build_max_workers: int = 2,
    request_build_max_pending: int | None = 16,
    prefill_coalesce_requests: int = 16,
    prefill_coalesce_wait_ms: float = 32.0,
    prefill_coalesce_when_idle: bool = True,
    prefill_coalesce_requires_pending_builds: bool = True,
    enable_pre_lm_encoder: bool = True,
    pre_lm_cache_max_entries: int = 4096,
    pre_lm_cache_size_bytes: int = 2 * 1024**3,
    pre_lm_max_batch_size: int = 8,
    pre_lm_max_batch_wait_ms: int = 0,
    pre_lm_max_pending: int = 32,
    enable_encoder_cuda_graph: bool = False,
    stream_emit_interval_s: float = 0.05,
    server_args_overrides: dict[str, Any] | None = None,
):
    from sglang_omni.models.arkasr.engine_builder import ArkasrEngineBuilder

    return ArkasrEngineBuilder(
        max_running_requests=max_running_requests,
        encoder_max_batch_size=encoder_max_batch_size,
        max_new_tokens=max_new_tokens,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        mem_fraction_static=mem_fraction_static,
        mm_embedding_cache_size_bytes=mm_embedding_cache_size_bytes,
        enable_torch_compile=enable_torch_compile,
        mm_attention_backend=mm_attention_backend,
        request_build_max_workers=request_build_max_workers,
        request_build_max_pending=request_build_max_pending,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
        prefill_coalesce_when_idle=prefill_coalesce_when_idle,
        prefill_coalesce_requires_pending_builds=(
            prefill_coalesce_requires_pending_builds
        ),
        enable_pre_lm_encoder=enable_pre_lm_encoder,
        pre_lm_cache_max_entries=pre_lm_cache_max_entries,
        pre_lm_cache_size_bytes=pre_lm_cache_size_bytes,
        pre_lm_max_batch_size=pre_lm_max_batch_size,
        pre_lm_max_batch_wait_ms=pre_lm_max_batch_wait_ms,
        pre_lm_max_pending=pre_lm_max_pending,
        enable_encoder_cuda_graph=enable_encoder_cuda_graph,
        stream_emit_interval_s=stream_emit_interval_s,
    ).build(
        model_path,
        device=device,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_arkasr_executor(*args, **kwargs):
    return create_sglang_arkasr_executor(*args, **kwargs)


__all__ = ["create_sglang_arkasr_executor", "create_arkasr_executor"]
