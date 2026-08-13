# SPDX-License-Identifier: Apache-2.0
"""Stage factory for SGLang-backed MOSS-Transcribe-Diarize inference."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from transformers import AutoConfig, GenerationConfig

from sglang_omni.models.moss_transcribe_diarize import (  # noqa: F401
    hf_config as _hf_config,
)

# Note (yijiang): Dense buckets avoid CUDA-graph capture at padded sizes
# we don't hit; pad-to-power-of-2 would tax a compute-bound encoder with
# no cross-request batching yet.
# Cap tuned to p99 audio duration ([1,8] covers up to ~4min)
_DEFAULT_ENCODER_CHUNK_BUCKETS = list(range(1, 9))


@contextmanager
def _missing_additional_chat_templates_compat() -> Iterator[None]:
    """Treat a missing optional chat-template directory as no extra templates."""
    import transformers.processing_utils as processing_utils
    import transformers.utils.hub as hub_utils
    from huggingface_hub.errors import RepositoryNotFoundError

    patched: list[tuple[Any, Any]] = []

    def patch_list_repo_templates(module: Any) -> None:
        original = getattr(module, "list_repo_templates", None)
        if original is None:
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return original(*args, **kwargs)
            except RepositoryNotFoundError as exc:
                if "additional_chat_templates" in str(exc):
                    return []
                raise

        setattr(module, "list_repo_templates", wrapped)
        patched.append((module, original))

    try:
        patch_list_repo_templates(processing_utils)
        patch_list_repo_templates(hub_utils)
        yield
    finally:
        for module, original in reversed(patched):
            setattr(module, "list_repo_templates", original)


def _default_context_length(model_path: str) -> int:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(config, "text_config", None)
    return int(getattr(text_config, "max_position_embeddings", 131072))


def _default_max_new_tokens(model_path: str) -> int:
    try:
        generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception:
        return 5120
    return int(getattr(generation_config, "max_new_tokens", None) or 5120)


def create_sglang_moss_transcribe_diarize_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_running_requests: int = 16,
    max_new_tokens: int | None = None,
    context_length: int | None = None,
    mem_fraction_static: float | None = 0.80,
    mm_embedding_cache_size_bytes: int = 0,
    encoder_cache_size_bytes: int = 0,
    enable_torch_compile: bool = False,
    # note (yichi): async on by default for MOSS-TD; --decode-mode sync to opt out.
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 2,
    prefill_coalesce_requests: int = 4,
    prefill_coalesce_wait_ms: float = 12.0,
    prefill_coalesce_when_idle: bool = True,
    prefill_coalesce_requires_pending_builds: bool = True,
    prefill_coalesce_after_builds_during_decode: bool = True,
    encoder_chunk_buckets: list[int] | None = None,
    encoder_torch_compile: bool = False,
    encoder_max_batch_size: int = 2,
    # note (yichi): 8 parallel mel extractions measured optimal; fewer starve
    # the encoder feed, more oversubscribe the CPU.
    request_build_max_workers: int = 8,
    request_build_max_pending: int | None = 16,
    stream_emit_interval_s: float = 0.05,
    server_args_overrides: dict[str, Any] | None = None,
):
    from sglang_omni.models.moss_transcribe_diarize.engine_builder import (
        MossTranscribeDiarizeEngineBuilder,
    )

    buckets = (
        encoder_chunk_buckets
        if encoder_chunk_buckets is not None
        else _DEFAULT_ENCODER_CHUNK_BUCKETS
    )
    return MossTranscribeDiarizeEngineBuilder(
        max_running_requests=max_running_requests,
        max_new_tokens=max_new_tokens,
        context_length=context_length,
        mem_fraction_static=mem_fraction_static,
        mm_embedding_cache_size_bytes=mm_embedding_cache_size_bytes,
        encoder_cache_size_bytes=encoder_cache_size_bytes,
        enable_torch_compile=enable_torch_compile,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        encoder_chunk_buckets=buckets,
        encoder_torch_compile=encoder_torch_compile,
        encoder_max_batch_size=encoder_max_batch_size,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
        prefill_coalesce_when_idle=prefill_coalesce_when_idle,
        prefill_coalesce_requires_pending_builds=(
            prefill_coalesce_requires_pending_builds
        ),
        prefill_coalesce_after_builds_during_decode=(
            prefill_coalesce_after_builds_during_decode
        ),
        request_build_max_workers=request_build_max_workers,
        request_build_max_pending=request_build_max_pending,
        stream_emit_interval_s=stream_emit_interval_s,
    ).build(
        model_path,
        device=device,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_moss_transcribe_diarize_executor(*args, **kwargs):
    return create_sglang_moss_transcribe_diarize_executor(*args, **kwargs)


__all__ = [
    "create_sglang_moss_transcribe_diarize_executor",
    "create_moss_transcribe_diarize_executor",
]
