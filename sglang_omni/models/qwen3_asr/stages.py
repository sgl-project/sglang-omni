# SPDX-License-Identifier: Apache-2.0
"""Stage factory for SGLang-backed Qwen3-ASR inference."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from transformers import AutoFeatureExtractor, AutoTokenizer

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.qwen3_asr.request_builders import (
    make_qwen3_asr_scheduler_adapters,
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
from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version
from sglang_omni.utils.gpu_memory import (
    format_bytes_gib,
    get_process_gpu_memory_bytes,
)

logger = logging.getLogger(__name__)


def _log_memory_checkpoint(checkpoint: str, gpu_id: int) -> None:
    logger.info(
        "Qwen3-ASR memory checkpoint=%s gpu=%d process_gpu_memory=%s",
        checkpoint,
        gpu_id,
        format_bytes_gib(get_process_gpu_memory_bytes(gpu_id)),
    )


def create_sglang_qwen3_asr_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "float16",
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

    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_path, trust_remote_code=True
    )

    encoder_token_count = int(feature_extractor.nb_max_frames // 2)

    defaults: dict[str, Any] = {
        "disable_cuda_graph": False,
        "disable_overlap_schedule": True,
        "enable_torch_compile": enable_torch_compile,
        "mem_fraction_static": mem_fraction_static,
        "max_prefill_tokens": 4096,
        "chunked_prefill_size": 4096,
        "sampling_backend": "pytorch",
        "dtype": dtype,
    }
    sm_version = get_visible_gpu_sm_version(gpu_id)
    mm_backend_reason = "explicit override"
    if mm_attention_backend is not None:
        defaults["mm_attention_backend"] = mm_attention_backend
    else:
        mm_backend_reason = "SGLang automatic selection"
        if sm_version == 89 or (sm_version is not None and sm_version >= 100):
            defaults["mm_attention_backend"] = "triton_attn"
            mm_backend_reason = f"validated capability policy for SM{sm_version}"
    overrides = build_generation_batch_overrides(
        max_running_requests=max_running_requests,
        server_args_overrides=server_args_overrides,
        **defaults,
    )

    server_args = build_sglang_server_args(
        model_path,
        context_length=encoder_token_count + int(max_new_tokens) + 8,
        **overrides,
    )
    validate_generation_batch_policy(
        model_name="Qwen3-ASR",
        server_args=server_args,
    )
    logger.info(
        "Qwen3-ASR runtime profile: sm=%s dtype=%s attention_backend=%s "
        "mm_attention_backend=%s mm_backend_reason=%s cuda_graph=%s "
        "cuda_graph_bs=%s torch_compile=%s max_running_requests=%s "
        "mem_fraction_static=%s",
        sm_version,
        server_args.dtype,
        server_args.attention_backend,
        server_args.mm_attention_backend,
        mm_backend_reason,
        not server_args.disable_cuda_graph,
        server_args.cuda_graph_bs,
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
        model_arch_override="Qwen3ASRForConditionalGeneration",
    )
    _log_memory_checkpoint("post_static_allocation", gpu_id)

    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()
    _log_memory_checkpoint("post_cuda_graph_capture", gpu_id)

    init_mm_embedding_cache(mm_embedding_cache_size_bytes)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    request_builder, result_adapter = make_qwen3_asr_scheduler_adapters(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
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
        request_build_max_workers=request_build_max_workers,
        request_build_max_pending=request_build_max_pending,
    )


def create_qwen3_asr_executor(*args, **kwargs):
    return create_sglang_qwen3_asr_executor(*args, **kwargs)


__all__ = ["create_sglang_qwen3_asr_executor", "create_qwen3_asr_executor"]
