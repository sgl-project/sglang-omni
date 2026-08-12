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
    max_new_tokens: int = 256,
    mem_fraction_static: float | None = None,
    mm_embedding_cache_size_bytes: int = 0,
    enable_torch_compile: bool = False,
    mm_attention_backend: str | None = None,
    request_build_max_workers: int = 2,
    request_build_max_pending: int | None = 16,
    server_args_overrides: dict[str, Any] | None = None,
):
    from sglang_omni.models.arkasr.engine_builder import ArkasrEngineBuilder

    return ArkasrEngineBuilder(
        max_running_requests=max_running_requests,
        max_new_tokens=max_new_tokens,
        mem_fraction_static=mem_fraction_static,
        mm_embedding_cache_size_bytes=mm_embedding_cache_size_bytes,
        enable_torch_compile=enable_torch_compile,
        mm_attention_backend=mm_attention_backend,
        request_build_max_workers=request_build_max_workers,
        request_build_max_pending=request_build_max_pending,
    ).build(
        model_path,
        device=device,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_arkasr_executor(*args, **kwargs):
    return create_sglang_arkasr_executor(*args, **kwargs)


__all__ = ["create_sglang_arkasr_executor", "create_arkasr_executor"]
