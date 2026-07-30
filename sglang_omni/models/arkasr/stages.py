# SPDX-License-Identifier: Apache-2.0
"""Stage factory for SGLang-backed ARK-ASR-3B inference."""

from __future__ import annotations

from typing import Any

from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from transformers import AutoConfig, AutoTokenizer, WhisperFeatureExtractor

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.arkasr.request_builders import make_arkasr_scheduler_adapters
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


def create_sglang_arkasr_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
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
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    merge_factor = int(getattr(hf_config, "merge_factor", 4))
    audio_token_id = int(getattr(hf_config, "audio_token_id", 151663))

    encoder_token_count = int(getattr(feature_extractor, "nb_max_frames", 3000) // 2)

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
    if mm_attention_backend is not None:
        defaults["mm_attention_backend"] = mm_attention_backend
    else:
        sm_version = get_visible_gpu_sm_version(gpu_id)
        if sm_version is not None and sm_version >= 100:
            defaults["mm_attention_backend"] = "triton_attn"
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
    validate_generation_batch_policy(model_name="ARK-ASR", server_args=server_args)

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
        model_arch_override="ArkasrForConditionalGeneration",
    )

    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    init_mm_embedding_cache(mm_embedding_cache_size_bytes)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    request_builder, result_adapter = make_arkasr_scheduler_adapters(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        max_new_tokens=max_new_tokens,
        merge_factor=merge_factor,
        audio_token_id=audio_token_id,
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


def create_arkasr_executor(*args, **kwargs):
    return create_sglang_arkasr_executor(*args, **kwargs)


__all__ = ["create_sglang_arkasr_executor", "create_arkasr_executor"]
