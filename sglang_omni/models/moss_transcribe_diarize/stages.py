# SPDX-License-Identifier: Apache-2.0
"""Stage factory for SGLang-backed MOSS-Transcribe-Diarize inference."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from transformers import AutoConfig, AutoProcessor, GenerationConfig

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_transcribe_diarize import (  # noqa: F401
    hf_config as _hf_config,
)
from sglang_omni.models.moss_transcribe_diarize.request_builders import (
    make_moss_transcribe_diarize_scheduler_adapters,
    make_moss_transcribe_diarize_stream_output_builder,
)
from sglang_omni.scheduling.bootstrap import (
    create_sglang_infrastructure_defer_cuda_graph,
)
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import (
    SGLangOutputProcessor,
    build_sglang_server_args,
)

# Note (yijiang): Dense buckets avoid CUDA-graph capture at padded sizes
# we don't hit; pad-to-power-of-2 would tax a compute-bound encoder with
# no cross-request batching yet.
# Cap tuned to p99 audio duration ([1,8] covers up to ~4min)
_DEFAULT_ENCODER_GRAPH_CHUNK_BUCKETS = list(range(1, 9))


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
    encoder_graph_chunk_buckets: list[int] | None = None,
    request_build_max_workers: int = 2,
    request_build_max_pending: int | None = 16,
    stream_emit_interval_s: float = 0.05,
    server_args_overrides: dict[str, Any] | None = None,
):
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    with _missing_additional_chat_templates_compat():
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    resolved_max_new_tokens = (
        int(max_new_tokens)
        if max_new_tokens is not None
        else _default_max_new_tokens(model_path)
    )
    resolved_context_length = (
        int(context_length)
        if context_length is not None
        else _default_context_length(model_path)
    )

    overrides = build_generation_batch_overrides(
        max_running_requests=max_running_requests,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=False,
        disable_overlap_schedule=True,
        enable_torch_compile=enable_torch_compile,
        mem_fraction_static=mem_fraction_static,
        max_prefill_tokens=4096,
        chunked_prefill_size=4096,
        sampling_backend="pytorch",
        dtype=dtype,
    )

    server_args = build_sglang_server_args(
        model_path,
        context_length=resolved_context_length,
        **overrides,
    )
    validate_generation_batch_policy(
        model_name="MOSS-Transcribe-Diarize",
        server_args=server_args,
    )

    want_cuda_graph, (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        gpu_id,
        model_arch_override="MossTranscribeDiarizeForConditionalGeneration",
    )

    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()
        buckets = (
            encoder_graph_chunk_buckets
            if encoder_graph_chunk_buckets is not None
            else _DEFAULT_ENCODER_GRAPH_CHUNK_BUCKETS
        )
        input_feature_len = int(processor.feature_extractor.nb_max_frames)
        model_worker.model_runner.model.init_encoder_graphs(buckets, input_feature_len)

    init_mm_embedding_cache(mm_embedding_cache_size_bytes)
    model_worker.model_runner.model.init_encoder_cache(encoder_cache_size_bytes)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    request_builder, result_adapter = make_moss_transcribe_diarize_scheduler_adapters(
        processor=processor,
        tokenizer=tokenizer,
        max_new_tokens=resolved_max_new_tokens,
    )
    stream_output_builder = make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=tokenizer,
        min_emit_interval_s=stream_emit_interval_s,
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=ModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_output_builder=stream_output_builder,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        request_build_max_workers=request_build_max_workers,
        request_build_max_pending=request_build_max_pending,
    )


def create_moss_transcribe_diarize_executor(*args, **kwargs):
    return create_sglang_moss_transcribe_diarize_executor(*args, **kwargs)


__all__ = [
    "create_sglang_moss_transcribe_diarize_executor",
    "create_moss_transcribe_diarize_executor",
]
