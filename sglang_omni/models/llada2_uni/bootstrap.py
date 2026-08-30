# SPDX-License-Identifier: Apache-2.0
"""Bootstrap helpers for LLaDA2-Uni scheduler."""

from __future__ import annotations

from typing import Any

from sglang_omni.vendor.sglang.server_args import override_server_args


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
