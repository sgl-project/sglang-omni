# SPDX-License-Identifier: Apache-2.0
"""Stage factory for SGLang-backed Whisper ASR inference."""

from __future__ import annotations

from typing import Any


def create_sglang_whisper_asr_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "float16",
    max_running_requests: int = 16,
    max_new_tokens: int = 256,
    mem_fraction_static: float = 0.85,
    enable_encoder_cuda_graph: bool = False,
    encoder_graph_batch_buckets: list[int] | None = None,
    server_args_overrides: dict[str, Any] | None = None,
):
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    return WhisperASREngineBuilder(
        max_running_requests=max_running_requests,
        mem_fraction_static=mem_fraction_static,
        max_new_tokens=max_new_tokens,
        enable_encoder_cuda_graph=enable_encoder_cuda_graph,
        encoder_graph_batch_buckets=encoder_graph_batch_buckets,
    ).build(
        model_path,
        device=device,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_whisper_asr_executor(*args, **kwargs):
    return create_sglang_whisper_asr_executor(*args, **kwargs)


__all__ = ["create_sglang_whisper_asr_executor", "create_whisper_asr_executor"]
