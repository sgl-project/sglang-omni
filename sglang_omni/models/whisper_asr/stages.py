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
    server_args_overrides: dict[str, Any] | None = None,
):
    from transformers import AutoProcessor, GenerationConfig

    from sglang_omni.model_runner.base import ModelRunner
    from sglang_omni.models.whisper_asr.request_builders import (
        make_whisper_scheduler_adapters,
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

    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    generation_config = GenerationConfig.from_pretrained(model_path)
    encoder_token_count = int(processor.feature_extractor.nb_max_frames // 2)

    overrides = build_generation_batch_overrides(
        max_running_requests=max_running_requests,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=False,
        disable_overlap_schedule=True,
        enable_torch_compile=True,
        mem_fraction_static=mem_fraction_static,
        max_prefill_tokens=4096,
        chunked_prefill_size=4096,
        sampling_backend="pytorch",
        dtype=dtype,
    )

    server_args = build_sglang_server_args(
        model_path,
        context_length=encoder_token_count + int(max_new_tokens) + 8,
        **overrides,
    )
    validate_generation_batch_policy(
        model_name="Whisper ASR",
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
        model_arch_override="WhisperForConditionalGeneration",
    )

    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    request_builder, result_adapter = make_whisper_scheduler_adapters(
        processor=processor,
        tokenizer=tokenizer,
        generation_config=generation_config,
        encoder_token_count=encoder_token_count,
        max_new_tokens=max_new_tokens,
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
    )


def create_whisper_asr_executor(*args, **kwargs):
    return create_sglang_whisper_asr_executor(*args, **kwargs)


__all__ = ["create_sglang_whisper_asr_executor", "create_whisper_asr_executor"]
