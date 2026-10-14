# SPDX-License-Identifier: Apache-2.0
"""Construct the target and draft before sizing their shared KV allocation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    refresh_deep_gemm_layout_memory_budget,
)
from sglang.srt.runtime_context import get_context, get_schedule
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

from sglang_omni.model_runner.model_worker import ModelWorker, ModelWorkerConfig
from sglang_omni.model_runner.speculative_target_worker import SpeculativeTargetWorker
from sglang_omni.scheduling.bootstrap import init_sglang_cuda_graphs
from sglang_omni.scheduling.sglang_backend.cache import create_tree_cache
from sglang_omni.scheduling.stage_kv_budget import consume_stage_kv_cache_bytes

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DFlashInfrastructure:
    model_worker: ModelWorker
    target_adapter: SpeculativeTargetWorker
    draft_worker: DFlashWorkerV2
    tree_cache: BasePrefixCache
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    draft_context_limit_tokens: int
    target_position_limit_tokens: int


def create_dflash_infrastructure(
    server_args: ServerArgs,
    gpu_id: int,
    *,
    tp_rank: int,
    nccl_port: int | None,
    total_gpu_memory_fraction: float | None,
    enable_prefill_input_embeds: bool,
) -> DFlashInfrastructure:
    """Initialize the opt-in Qwen thinker DFlash lifecycle."""
    configuration = resolved_view(server_args)
    if get_context().is_config_namespace_published("model"):
        raise RuntimeError("The thinker must own an unpublished runtime process")
    elif configuration.tp_size != 1 or configuration.pp_size != 1:
        raise ValueError("Thinker DFlash currently requires a TP1/PP1 worker")
    elif configuration.enable_streaming_session:
        raise ValueError("Thinker DFlash does not support streaming sessions")
    elif (
        configuration.speculative_draft_window_size is not None
        or configuration.enable_hierarchical_cache
        or configuration.enable_lmcache
        or configuration.enable_unified_cache_external_linker
        or configuration.disaggregation_mode != "null"
        or configuration.enable_pd_role_switch
        or configuration.enable_two_batch_overlap
        or configuration.enable_lora
    ):
        raise ValueError(
            "Thinker DFlash requires ordinary full-attention caching without "
            "draft windowing, external caches, disaggregation, LoRA, or batch overlap"
        )
    elif envs.SGLANG_SIMULATE_ACC_LEN.get() > 0:
        raise ValueError("Thinker DFlash requires real draft acceptance")
    elif (
        configuration.speculative_accept_threshold_single != 1.0
        or configuration.speculative_accept_threshold_acc != 1.0
    ):
        raise ValueError("Thinker DFlash requires exact acceptance thresholds")
    else:
        pass
    worker_config = ModelWorkerConfig(
        model_arch_override="Qwen3OmniThinkerForCausalLM",
        nccl_port=nccl_port,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        kv_cache_bytes=consume_stage_kv_cache_bytes(),
        enable_prefill_input_embeds=enable_prefill_input_embeds,
    )
    model_worker = ModelWorker(
        config=worker_config,
        server_args=server_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
    )
    target_adapter = SpeculativeTargetWorker(model_worker)
    draft_worker = DFlashWorkerV2(
        server_args=server_args,
        gpu_id=gpu_id,
        nccl_port=model_worker.nccl_port,
        target_worker=target_adapter,
    )
    target_runner = model_worker.model_runner
    target_runner.account_preloaded_weights(
        target_runner.preloaded_weights_bytes + draft_worker.preloaded_weights_bytes
    )
    target_runner.alloc_memory_pool()
    request_pool, token_allocator = model_worker.get_memory_pool()
    draft_worker.alloc_memory_pool(
        memory_pool_config=target_runner.memory_pool_config,
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=token_allocator,
    )
    draft_worker.init_hicache_draft_plan()
    target_runner.init_attention_backends()
    draft_worker.init_attention_backends()
    init_sglang_cuda_graphs(model_worker)
    draft_worker.init_cuda_graphs()
    draft_worker.prewarm_sampling()
    if target_runner.token_to_kv_pool.post_capture_active:
        target_runner.post_capture_resize_kv_pool(
            draft_runners=(draft_worker.draft_model_runner,)
        )
    else:
        pass
    refresh_deep_gemm_layout_memory_budget(target_runner, only_if_initialized=True)
    draft_runner = draft_worker.draft_model_runner
    if (
        target_runner.req_to_token_pool is not request_pool
        or draft_runner.req_to_token_pool is not request_pool
        or target_runner.token_to_kv_pool_allocator is not token_allocator
        or draft_runner.token_to_kv_pool_allocator is not token_allocator
    ):
        raise RuntimeError("Thinker DFlash requires shared request and KV allocation")
    else:
        pass
    context_capacities_tokens = [
        target_runner.model_config.context_len,
        draft_runner.model_config.context_len,
        request_pool.max_context_len,
        request_pool.req_to_token.shape[1],
        token_allocator.size,
        target_runner.token_to_kv_pool.size,
        draft_runner.token_to_kv_pool.size,
    ]
    target_rotary_capacities_tokens = []
    for layers in (target_runner.model.model.layers, draft_worker.draft_model.layers):
        if len(layers) == 0:
            raise RuntimeError("Thinker DFlash requires initialized attention layers")
        else:
            pass
        for layer in layers:
            rotary_embedding = layer.self_attn.rotary_emb
            if rotary_embedding.position_start != 0:
                raise ValueError("Thinker DFlash requires zero-based rotary positions")
            else:
                pass
            rotary_capacity_tokens = rotary_embedding.cos_sin_cache.shape[0]
            context_capacities_tokens.append(rotary_capacity_tokens)
            if layers is target_runner.model.model.layers:
                target_rotary_capacities_tokens.append(rotary_capacity_tokens)
            else:
                pass
    if any(
        type(capacity) is not int or capacity <= 0
        for capacity in context_capacities_tokens
    ):
        raise RuntimeError(
            "Thinker DFlash requires positive initialized context capacities"
        )
    else:
        pass
    # note (Codex): Runtime capacity does not certify draft training length or acceptance.
    draft_context_limit_tokens = min(context_capacities_tokens)
    target_position_limit_tokens = min(target_rotary_capacities_tokens)
    logger.info(
        f"Thinker DFlash initialized context budget: {draft_context_limit_tokens} tokens; "
        f"target={context_capacities_tokens[0]}, draft={context_capacities_tokens[1]}, "
        f"request_mapping={context_capacities_tokens[3]}, KV={context_capacities_tokens[4:7]}, "
        f"minimum_rotary_cache={min(context_capacities_tokens[7:])}, "
        f"target_position_limit={target_position_limit_tokens}"
    )
    tree_cache = create_tree_cache(
        request_pool,
        token_allocator,
        get_schedule().page_size,
        target_runner.sliding_window_size,
    )
    return DFlashInfrastructure(
        model_worker=model_worker,
        target_adapter=target_adapter,
        draft_worker=draft_worker,
        tree_cache=tree_cache,
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=token_allocator,
        draft_context_limit_tokens=draft_context_limit_tokens,
        target_position_limit_tokens=target_position_limit_tokens,
    )
