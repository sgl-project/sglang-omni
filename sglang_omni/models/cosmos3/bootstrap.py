# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 text scheduler construction."""

from __future__ import annotations

from typing import Any


def create_text_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_async_decode: bool = False,
    async_decode_min_batch_size: int = 2,
):
    """Create the SGLang-backed Cosmos3 text AR scheduler."""

    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.model_runner.base import ModelRunner
    from sglang_omni.models.cosmos3.request_builders import (
        make_text_scheduler_adapters,
        make_text_stream_output_builder,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_manager,
        decode_manager,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override="Cosmos3TextForCausalLM",
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )

    output_processor = SGLangOutputProcessor(capture_hidden=False)
    model_runner = ModelRunner(model_worker, output_processor)
    tokenizer = get_tokenizer(
        model_config.model_path,
        trust_remote_code=True,
    )
    request_builder, result_adapter = make_text_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=model_config.vocab_size,
        generation_config=model_config.hf_generation_config,
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_manager,
        decode_manager=decode_manager,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_output_builder=make_text_stream_output_builder(),
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
    )


__all__ = ["create_text_scheduler"]
