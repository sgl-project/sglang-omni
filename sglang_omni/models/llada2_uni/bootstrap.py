# SPDX-License-Identifier: Apache-2.0
"""Bootstrap helpers for LLaDA2-Uni scheduler."""

from __future__ import annotations

from typing import Any

from sglang_omni.vendor.sglang.server_args import override_server_args


def register_llada2_uni_cfg() -> None:
    """Call before build_server_args for the omni LowConfidenceCFG variant.

    The text-only LowConfidence variant does not need this registration.
    """
    from sglang.srt.dllm.algorithm import algo_name_to_cls

    from sglang_omni.models.llada2_uni.cfg_attention_backend import (
        register_llada2_cfg_flashinfer_backend,
    )
    from sglang_omni.models.llada2_uni.low_confidence_cfg import LowConfidenceCFG

    algo_name_to_cls["LowConfidenceCFG"] = LowConfidenceCFG
    register_llada2_cfg_flashinfer_backend()


def validate_cfg(server_args: Any) -> None:
    from sglang.srt.arg_groups.model_override_base import (
        attention_backends_of,
        resolved_view,
    )

    cfg = resolved_view(server_args)
    if cfg.dllm_algorithm != "LowConfidenceCFG":
        return
    if not cfg.disable_cuda_graph:
        raise ValueError("LowConfidenceCFG does not support CUDA graphs")

    register_llada2_uni_cfg()
    from sglang_omni.models.llada2_uni.cfg_attention_backend import (
        CFG_ATTENTION_BACKEND,
    )

    if any(backend != CFG_ATTENTION_BACKEND for backend in attention_backends_of(cfg)):
        raise ValueError(
            "LowConfidenceCFG requires llada2_uni_cfg_flashinfer (DLLM pad masking), "
            "not the upstream llada2_cfg_flashinfer text-condition mask backend"
        )
    if cfg.dllm_fdfo:
        raise ValueError("LowConfidenceCFG requires synchronous DLLM, not FDFO")


def create_dllm_thinker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
):
    """Create an DllmScheduler for the LLaDA2-Uni thinker.

    Returns a ``DllmScheduler`` with ``dllm_config`` set, ready to be
    driven by a ``Stage``.
    """
    from sglang.srt.dllm.config import DllmConfig
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.models.llada2_uni.request_builders import (
        make_dllm_thinker_scheduler_adapters,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.dllm_scheduler import DllmScheduler

    validate_cfg(server_args)
    dllm_config = DllmConfig.from_server_args(server_args)

    # sglang supports radix cache with dLLM, but Omni's dLLM staging
    # path has only been validated without it; keep it disabled deliberately.
    override_server_args(
        server_args,
        "sglang_omni.llada2_uni.disable_radix_cache",
        disable_radix_cache=True,
    )

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override="LLaDA2MoeModelLM",
    )

    tokenizer = get_tokenizer(model_config.model_path, trust_remote_code=True)

    request_builder, result_adapter = make_dllm_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=model_config.vocab_size,
        dllm_config=dllm_config,
    )

    return DllmScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        dllm_config=dllm_config,
        request_builder=request_builder,
        result_adapter=result_adapter,
    )
