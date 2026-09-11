# SPDX-License-Identifier: Apache-2.0
"""Stage factory for MOSS-Transcribe-Diarize Prefill/Decode inference."""

from __future__ import annotations

from typing import Any, Literal


def _builder_defaults() -> dict[str, Any]:
    return {
        "max_running_requests": 16,
        "max_new_tokens": None,
        "context_length": None,
        "mem_fraction_static": 0.80,
        "mm_embedding_cache_size_bytes": 0,
        "encoder_cache_size_bytes": 0,
        "enable_torch_compile": False,
        "torch_compile_max_bs": 4,
        "enable_async_decode": True,
        "async_decode_min_batch_size": 1,
        "prefill_coalesce_requests": 4,
        "prefill_coalesce_wait_ms": 12.0,
        "prefill_coalesce_when_idle": True,
        "prefill_coalesce_requires_pending_builds": True,
        "prefill_coalesce_after_builds_during_decode": True,
        "encoder_chunk_buckets": list(range(1, 9)),
        "encoder_torch_compile": False,
        "encoder_max_batch_size": 2,
        "request_build_max_workers": 8,
        "request_build_max_pending": 16,
        "stream_emit_interval_s": 0.05,
    }


def _create_pd_executor(
    model_path: str,
    *,
    pd_role: Literal["prefill", "decode"],
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
    **builder_overrides: Any,
):
    from sglang_omni.models.moss_transcribe_diarize.pd.engine_builder import (
        MossTranscribeDiarizePDEngineBuilder,
    )

    builder_kwargs = _builder_defaults()
    builder_kwargs.update(builder_overrides)
    return MossTranscribeDiarizePDEngineBuilder(
        pd_role=pd_role,
        **builder_kwargs,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_sglang_moss_transcribe_diarize_prefill_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
    **builder_overrides: Any,
):
    return _create_pd_executor(
        model_path,
        pd_role="prefill",
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
        **builder_overrides,
    )


def create_sglang_moss_transcribe_diarize_decode_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
    **builder_overrides: Any,
):
    return _create_pd_executor(
        model_path,
        pd_role="decode",
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
        **builder_overrides,
    )


__all__ = [
    "create_sglang_moss_transcribe_diarize_decode_executor",
    "create_sglang_moss_transcribe_diarize_prefill_executor",
]
