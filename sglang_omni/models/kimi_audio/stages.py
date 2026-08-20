# SPDX-License-Identifier: Apache-2.0
"""Kimi-Audio stage factory."""

from __future__ import annotations

from typing import Any


def create_kimi_audio_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_running_requests: int = 8,
    max_new_tokens: int = 512,
    mem_fraction_static: float | None = None,
    enable_torch_compile: bool = False,
    request_build_max_workers: int = 2,
    request_build_max_pending: int | None = 8,
    audio_tokenizer_path: str = "THUDM/glm-4-voice-tokenizer",
    server_args_overrides: dict[str, Any] | None = None,
):
    from .engine_builder import KimiAudioEngineBuilder

    return KimiAudioEngineBuilder(
        max_running_requests=max_running_requests,
        max_new_tokens=max_new_tokens,
        mem_fraction_static=mem_fraction_static,
        enable_torch_compile=enable_torch_compile,
        request_build_max_workers=request_build_max_workers,
        request_build_max_pending=request_build_max_pending,
        audio_tokenizer_path=audio_tokenizer_path,
    ).build(
        model_path,
        device=device,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


__all__ = ["create_kimi_audio_executor"]
