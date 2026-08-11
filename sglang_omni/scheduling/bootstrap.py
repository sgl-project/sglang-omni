# SPDX-License-Identifier: Apache-2.0
"""Generic SGLang bootstrap utilities for model-specific schedulers."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from sglang_omni.utils.gpu_compat import (
    get_visible_gpu_sm_version,
    gpu_architecture_for_sm,
)
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)


class _SGLangServerArgsForDiagnostics(Protocol):
    attention_backend: str | None
    sampling_backend: str | None

    def get_attention_backends(self) -> tuple[str | None, str | None]: ...


def _describe_sglang_runtime_configuration(
    server_args: _SGLangServerArgsForDiagnostics,
    gpu_id: int,
) -> str:
    sm_version = get_visible_gpu_sm_version(gpu_id)
    prefill_attention_backend, decode_attention_backend = (
        server_args.get_attention_backends()
    )
    return (
        f"SGLang runtime configuration: gpu_id={gpu_id}, sm={sm_version}, "
        f"architecture={gpu_architecture_for_sm(sm_version)}, "
        f"attention_backend={server_args.attention_backend}, "
        f"decode_attention_backend={decode_attention_backend}, "
        f"prefill_attention_backend={prefill_attention_backend}, "
        f"sampling_backend={server_args.sampling_backend}"
    )


def init_sglang_cuda_graphs(model_worker: Any) -> None:
    """Initialize SGLang graphs with Omni's prefill-embedding capture view."""
    if not model_worker.enable_prefill_input_embeds:
        # Required even when graphs are disabled: SGLang installs its eager
        # phase runner from init_cuda_graphs().
        model_worker.model_runner.init_cuda_graphs()
        return

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


def create_sglang_infrastructure(
    server_args: Any,
    gpu_id: int,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    model_arch_override: str | None = None,
    weight_prefix: str | None = None,
    capture_hidden_layers: list[int] | None = None,
    total_gpu_memory_fraction: float | None = None,
    defer_cuda_graph_capture: bool = False,
    enable_prefill_input_embeds: bool = False,
):
    """Create SGLang worker, memory pools, tree cache, and prefill/decode managers."""
    from sglang_omni.model_runner.model_worker import ModelWorker, ModelWorkerConfig
    from sglang_omni.scheduling.sglang_backend import (
        DecodeManager,
        PrefillManager,
        create_tree_cache,
    )

    logger.info(_describe_sglang_runtime_configuration(server_args, gpu_id))

    model_worker = ModelWorker(
        config=ModelWorkerConfig(
            model_arch_override=model_arch_override,
            weight_prefix=weight_prefix,
            nccl_port=nccl_port,
            total_gpu_memory_fraction=total_gpu_memory_fraction,
            enable_prefill_input_embeds=enable_prefill_input_embeds,
        ),
        server_args=server_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
    )

    if capture_hidden_layers:
        from sglang_omni.model_runner._hidden_capture import (
            install_hidden_capture_hooks,
        )

        model = model_worker.model_runner.model
        install_hidden_capture_hooks(model, capture_hidden_layers)

    # SGLang 0.5.15 split model loading, KV-pool allocation, attention-backend
    # (order re-verified against 0.5.16 Scheduler.init_model_worker)
    # initialization, and CUDA-graph initialization into explicit phases. Keep
    # the same order as upstream's Scheduler.init_model_worker(), while
    # preserving Omni's pre-backend hidden-capture hook installation above.
    model_runner = model_worker.model_runner
    model_runner.alloc_memory_pool()
    model_runner.init_attention_backends()

    if not defer_cuda_graph_capture:
        init_sglang_cuda_graphs(model_worker)

    req_to_token_pool, token_to_kv_pool_allocator = model_worker.get_memory_pool()

    tree_cache = create_tree_cache(
        server_args,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        server_args.page_size,
    )

    enable_overlap = not server_args.disable_overlap_schedule

    prefill_mgr = PrefillManager(
        page_size=server_args.page_size,
        chunked_prefill_size=server_args.chunked_prefill_size,
        max_prefill_tokens=server_args.max_prefill_tokens,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_worker.model_config,
        enable_overlap=enable_overlap,
    )

    decode_mgr = DecodeManager(
        server_args=server_args,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        on_retract=lambda req: prefill_mgr.add_one_request(req),
    )

    return (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_worker.model_config,
    )


# note (luojiaxuan): Some Omni generation stages cannot let the generic SGLang
# worker capture CUDA graphs immediately during infrastructure construction. At
# that point the shared request pools exist, but stage-owned decode state may not:
# speech tokenizers may still need to be attached, sampler or feedback buffers
# may not be allocated, stage-local decode helpers may not be compiled, and the
# model-specific buffer capacity may not yet have been checked against the
# serving batch policy. Capturing before that work would freeze replay around an
# incomplete decode path and can make later steady-state requests either miss the
# intended graph buckets or overrun model-side per-request buffers. The capture
# priority should be the hot path users repeatedly pay for under concurrency:
# decode batches admitted by max_running_requests, capped by cuda_graph_max_bs
# and request-token slots, with all per-request model buffers already allocated.
# One-time bootstrap work such as processor loading, cache construction, audio
# decoder/vocoder setup, and other host-side staging should stay outside CUDA
# graph coverage because graph replay will not amortize it. This helper therefore
# disables worker-time capture only long enough to build the shared SGLang
# infrastructure, restores the user's CUDA-graph setting, and tells the caller
# whether it should call init_cuda_graphs() after its stage-specific setup.
def create_sglang_infrastructure_defer_cuda_graph(
    server_args: Any,
    gpu_id: int,
    **kwargs: Any,
):
    """Build shared SGLang infrastructure while deferring CUDA graph capture.

    The caller finishes stage-specific decode setup, then runs
    init_cuda_graphs() only when this returns that CUDA graphs were requested.
    """
    want_cuda_graph = not bool(server_args.disable_cuda_graph)
    if want_cuda_graph:
        override_server_args(
            server_args,
            "sglang_omni.defer_cuda_graph_capture",
            disable_cuda_graph=True,
        )
    try:
        infrastructure = create_sglang_infrastructure(
            server_args,
            gpu_id,
            defer_cuda_graph_capture=want_cuda_graph,
            **kwargs,
        )
    finally:
        if want_cuda_graph:
            override_server_args(
                server_args,
                "sglang_omni.restore_cuda_graph_capture",
                disable_cuda_graph=False,
            )
    return want_cuda_graph, infrastructure
