# SPDX-License-Identifier: Apache-2.0
"""Stage factory for Nemotron Transcribe."""

from __future__ import annotations

from collections.abc import Mapping

from sglang_omni.models.nemotron_transcribe.config import EncoderCompileMode
from sglang_omni.models.nemotron_transcribe.engine_builder import (
    NemotronTranscribeEngineBuilder,
)
from sglang_omni.models.nemotron_transcribe.request_builders import (
    NemotronTranscribeRequestData,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def create_nemotron_transcribe_executor(
    model_path: str,
    *,
    dtype: str,
    max_new_tokens: int,
    context_length: int,
    max_audio_seconds: float | None,
    mm_embedding_cache_size_bytes: int,
    chunked_prefill_size: int,
    encoder_max_batch_size: int,
    enable_pre_lm_encoder: bool,
    pre_lm_cache_max_entries: int,
    pre_lm_cache_size_bytes: int,
    pre_lm_max_batch_wait_ms: int,
    request_build_max_workers: int,
    request_build_max_pending: int | None,
    stream_emit_interval_s: float,
    compile_perception: bool,
    perception_cuda_graphs: bool,
    perception_compile_mode: EncoderCompileMode,
    device: str | None = None,
    gpu_id: int | None = None,
    mem_fraction_static: float | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler[NemotronTranscribeRequestData]:
    return NemotronTranscribeEngineBuilder(
        max_new_tokens=max_new_tokens,
        context_length=context_length,
        max_audio_seconds=max_audio_seconds,
        mem_fraction_static=mem_fraction_static,
        mm_embedding_cache_size_bytes=mm_embedding_cache_size_bytes,
        chunked_prefill_size=chunked_prefill_size,
        encoder_max_batch_size=encoder_max_batch_size,
        enable_pre_lm_encoder=enable_pre_lm_encoder,
        pre_lm_cache_max_entries=pre_lm_cache_max_entries,
        pre_lm_cache_size_bytes=pre_lm_cache_size_bytes,
        pre_lm_max_batch_wait_ms=pre_lm_max_batch_wait_ms,
        request_build_max_workers=request_build_max_workers,
        request_build_max_pending=request_build_max_pending,
        stream_emit_interval_s=stream_emit_interval_s,
        compile_perception=compile_perception,
        perception_cuda_graphs=perception_cuda_graphs,
        perception_compile_mode=perception_compile_mode,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


__all__ = ["create_nemotron_transcribe_executor"]
