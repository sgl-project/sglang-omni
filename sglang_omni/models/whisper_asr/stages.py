# SPDX-License-Identifier: Apache-2.0
"""Stage factory for SGLang-backed Whisper ASR inference."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version
from sglang_omni.utils.gpu_memory import format_bytes_gib, get_process_gpu_memory_bytes

logger = logging.getLogger(__name__)


def _log_memory_checkpoint(checkpoint: str, gpu_id: int) -> None:
    logger.info(
        "Whisper ASR memory checkpoint=%s gpu=%d process_gpu_memory=%s",
        checkpoint,
        gpu_id,
        format_bytes_gib(get_process_gpu_memory_bytes(gpu_id)),
    )


def create_sglang_whisper_asr_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "float16",
    max_running_requests: int = 16,
    max_new_tokens: int = 256,
    mem_fraction_static: float = 0.85,
    enable_torch_compile: bool = True,
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
        get_decode_cuda_graph_bs,
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
    sm_version = get_visible_gpu_sm_version(gpu_id)

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
        context_length=encoder_token_count + int(max_new_tokens) + 8,
        **overrides,
    )
    validate_generation_batch_policy(
        model_name="Whisper ASR",
        server_args=server_args,
    )
    logger.info(
        "Whisper ASR runtime profile: sm=%s dtype=%s "
        "attention_backend=%s encoder_attention_backend=torch_sdpa "
        "cuda_graph=%s cuda_graph_bs=%s torch_compile=%s "
        "max_running_requests=%s mem_fraction_static=%s",
        sm_version,
        server_args.dtype,
        server_args.attention_backend,
        not server_args.disable_cuda_graph,
        get_decode_cuda_graph_bs(server_args),
        server_args.enable_torch_compile,
        server_args.max_running_requests,
        server_args.mem_fraction_static,
    )
    _log_memory_checkpoint("pre_model_load", gpu_id)

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
    _log_memory_checkpoint("post_static_allocation", gpu_id)

    if want_cuda_graph:
        model_worker.model_runner.init_cuda_graphs()
        _log_memory_checkpoint("post_cuda_graph_capture", gpu_id)

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
