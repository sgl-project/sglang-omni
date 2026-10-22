# SPDX-License-Identifier: Apache-2.0
"""Generic SGLang bootstrap utilities for model-specific schedulers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypedDict

from typing_extensions import Unpack

if TYPE_CHECKING:
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

    from sglang_omni.model_runner.model_worker import ModelWorker
    from sglang_omni.vendor.sglang.core import ServerArgs
else:
    pass

from sglang_omni.utils.gpu_compat import (
    get_visible_gpu_sm_version,
    gpu_architecture_for_sm,
)

logger = logging.getLogger(__name__)


class SGLangServerArgsForDiagnostics(Protocol):
    attention_backend: str | None
    prefill_attention_backend: str | None
    decode_attention_backend: str | None
    sampling_backend: str | None


class InfrastructureOptions(TypedDict, total=False):
    tp_rank: int
    nccl_port: int | None
    model_arch_override: str | None
    weight_prefix: str | None
    total_gpu_memory_fraction: float | None
    enable_prefill_input_embeds: bool
    before_memory_pool: Callable[["ModelWorker"], None] | None
    supports_speculative_decoding: bool
    speculative_worker_class: type[BaseSpecWorker] | None


def describe_sglang_runtime_configuration(
    server_args: SGLangServerArgsForDiagnostics,
    gpu_id: int,
) -> str:
    from sglang.srt.arg_groups.model_override_base import (
        attention_backends_of,
        resolved_view,
    )

    sm_version = get_visible_gpu_sm_version(gpu_id)
    cfg = resolved_view(server_args)
    prefill_attention_backend, decode_attention_backend = attention_backends_of(cfg)
    return (
        f"SGLang runtime configuration: gpu_id={gpu_id}, sm={sm_version}, "
        f"architecture={gpu_architecture_for_sm(sm_version)}, "
        f"attention_backend={cfg.attention_backend}, "
        f"decode_attention_backend={decode_attention_backend}, "
        f"prefill_attention_backend={prefill_attention_backend}, "
        f"sampling_backend={cfg.sampling_backend}"
    )


def init_sglang_cuda_graphs(
    model_worker: "ModelWorker",
) -> None:
    """Initialize SGLang graphs with Omni's prefill-embedding capture view."""
    if not model_worker.enable_prefill_input_embeds:
        # Required even when graphs are disabled: SGLang installs its eager
        # phase runner from init_cuda_graphs().
        model_worker.model_runner.init_cuda_graphs()
    else:
        model_config = model_worker.model_config
        original_is_multimodal = model_config.is_multimodal
        # Attention backends have already captured the model's real modality.
        # Only the prefill graph runner needs this temporary view so it allocates
        # the upstream input_embeds replay buffer.
        model_config.is_multimodal = True
        try:
            model_worker.model_runner.init_cuda_graphs()
        finally:
            model_config.is_multimodal = original_is_multimodal
    if model_worker.speculative_worker is not None:
        model_worker.speculative_worker.init_cuda_graphs()
        model_worker.speculative_worker.prewarm_sampling()
    else:
        pass


def create_sglang_infrastructure(
    server_args: "ServerArgs",
    gpu_id: int,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    model_arch_override: str | None = None,
    weight_prefix: str | None = None,
    total_gpu_memory_fraction: float | None = None,
    defer_cuda_graph_capture: bool = False,
    enable_prefill_input_embeds: bool = False,
    before_memory_pool: Callable[["ModelWorker"], None] | None = None,
    supports_speculative_decoding: bool = False,
    speculative_worker_class: type[BaseSpecWorker] | None = None,
):
    """Create SGLang worker, memory pools, and tree cache.

    before_memory_pool runs with the model worker after the weights are loaded
    and before the KV pool is sized, for resources the stage keeps for the life
    of the process. When the server args name a speculative algorithm, a stage
    that sets supports_speculative_decoding gets the draft worker
    (speculative_worker_class, else the algorithm's own) attached to its model
    worker, sharing its request and KV allocators; any other stage is refused.
    """
    # ModelWorker publishes server_args as the process-wide runtime context
    # once its pre-publish declarations are made; publishing again would
    # silently reconfigure whatever already runs here, so an engine is only
    # built where the context is unpublished. A construction that failed after
    # publishing is therefore not retried here.
    from sglang.srt.runtime_context import get_context, get_schedule, get_spec
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    from sglang_omni.model_runner.model_worker import ModelWorker, ModelWorkerConfig
    from sglang_omni.model_runner.speculative_target_worker import (
        SpeculativeTargetWorker,
    )
    from sglang_omni.scheduling.sglang_backend import create_tree_cache
    from sglang_omni.scheduling.stage_kv_budget import consume_stage_kv_cache_bytes

    if get_context().is_config_namespace_published("model"):
        raise RuntimeError(
            "this process already holds a published SGLang runtime context; "
            "an SGLang AR engine must own its OS process. Place SGLang AR "
            "stages in separate processes."
        )
    else:
        pass

    logger.info(describe_sglang_runtime_configuration(server_args, gpu_id))

    kv_cache_bytes = consume_stage_kv_cache_bytes()
    worker_config = ModelWorkerConfig(
        model_arch_override=model_arch_override,
        weight_prefix=weight_prefix,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        kv_cache_bytes=kv_cache_bytes,
        enable_prefill_input_embeds=enable_prefill_input_embeds,
    )
    model_worker = ModelWorker(
        config=worker_config,
        server_args=server_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
    )

    if before_memory_pool is not None:
        # note(ratish): sglang sizes the pool from free memory at this point, so
        # whatever the stage keeps resident has to exist before the reading.
        before_memory_pool(model_worker)
    else:
        pass

    # Phase order follows upstream Scheduler.init_model_worker().
    model_runner = model_worker.model_runner
    speculative_algorithm = SpeculativeAlgorithm.from_string(
        get_spec().speculative_algorithm
    )
    if speculative_algorithm.is_none():
        model_runner.alloc_memory_pool()
        model_runner.init_attention_backends()
    elif not supports_speculative_decoding:
        # note (wilsonzheng0327): speculative tokens skip the runner's decode hooks,
        # so only a stage that hands them on through on_speculative_tokens opts in.
        raise ValueError(
            f"speculative_algorithm={speculative_algorithm.name} is set, but this "
            "stage does not support speculative decoding"
        )
    else:
        # note (wilsonzheng0327): the draft loads before the pool is sized, so its
        # weights are counted, and its KV pool shares the target's slot ids.
        target_worker = SpeculativeTargetWorker(model_worker)
        if speculative_worker_class is None:
            speculative_worker_class = speculative_algorithm.create_worker(server_args)
        else:
            pass
        speculative_worker = speculative_worker_class(
            server_args=server_args,
            gpu_id=gpu_id,
            nccl_port=model_worker.nccl_port,
            target_worker=target_worker,
        )
        model_runner.alloc_memory_pool()
        if model_runner.token_to_kv_pool.post_capture_active:
            raise ValueError(
                "speculative decoding does not support post-capture KV sizing"
            )
        else:
            pass
        req_to_token_pool, token_to_kv_pool_allocator = model_worker.get_memory_pool()
        speculative_worker.alloc_memory_pool(
            memory_pool_config=model_runner.memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        speculative_worker.init_hicache_draft_plan()
        model_runner.init_attention_backends()
        speculative_worker.init_attention_backends()
        model_worker.speculative_target_worker = target_worker
        model_worker.speculative_worker = speculative_worker

    if not defer_cuda_graph_capture:
        init_sglang_cuda_graphs(model_worker)
    else:
        pass

    req_to_token_pool, token_to_kv_pool_allocator = model_worker.get_memory_pool()

    tree_cache = create_tree_cache(
        req_to_token_pool,
        token_to_kv_pool_allocator,
        get_schedule().page_size,
        model_runner.sliding_window_size,
    )

    return (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        model_worker.model_config,
    )


# note (luojiaxuan): Some Omni generation stages cannot capture CUDA graphs
# immediately during infrastructure construction. At that point the shared
# request pools exist, but stage-owned decode state may not: speech tokenizers
# may still need to be attached, sampler or feedback buffers may not be
# allocated, stage-local decode helpers may not be compiled, and the
# model-specific buffer capacity may not yet have been checked against the
# serving batch policy. Capturing before that work would freeze replay around an
# incomplete decode path and can make later steady-state requests either miss the
# intended graph buckets or overrun model-side per-request buffers. The capture
# priority should be the hot path users repeatedly pay for under concurrency:
# decode batches admitted by max_running_requests, capped by cuda_graph_max_bs
# and request-token slots, with all per-request model buffers already allocated.
# One-time bootstrap work such as processor loading, cache construction, audio
# decoder/vocoder setup, and other host-side staging should stay outside CUDA
# graph coverage because graph replay will not amortize it.
def create_sglang_infrastructure_defer_cuda_graph(
    server_args: "ServerArgs",
    gpu_id: int,
    **kwargs: Unpack[InfrastructureOptions],
):
    """Build shared SGLang infrastructure while deferring CUDA graph capture.

    The caller finishes stage-specific decode setup, then runs
    init_cuda_graphs() only when this returns that CUDA graphs were requested.
    """
    from sglang.srt.arg_groups.model_override_base import resolved_view

    cfg = resolved_view(server_args)
    want_cuda_graph = not bool(cfg.disable_cuda_graph)
    infrastructure = create_sglang_infrastructure(
        server_args,
        gpu_id,
        defer_cuda_graph_capture=want_cuda_graph,
        **kwargs,
    )
    return want_cuda_graph, infrastructure
