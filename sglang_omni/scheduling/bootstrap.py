# SPDX-License-Identifier: Apache-2.0
"""Generic SGLang bootstrap utilities for model-specific schedulers."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from sglang_omni.utils.gpu_compat import (
    get_visible_gpu_sm_version,
    gpu_architecture_for_sm,
)

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
    from sglang.srt.utils.tensor_bridge import use_mlx

    if use_mlx():
        # note (yexiaodong): The MLX stub has no Torch graph lifecycle because
        # native MLX lazy evaluation owns graph execution.
        return
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


def _hidden_capture_max_tokens(server_args: Any) -> int:
    """Largest token-row count a single thinker forward can produce.

    Covers chunked prefill, non-chunked prefill, decode batches, and every
    configured CUDA graph bucket maximum, so the capture buffers are large
    enough for both eager forwards and graph replay.
    """
    chunked_prefill_size = server_args.chunked_prefill_size
    candidates: list[Any] = []
    if chunked_prefill_size is not None and chunked_prefill_size > 0:
        candidates.append(chunked_prefill_size)
    else:
        candidates.append(server_args.max_prefill_tokens)
        # Note(wenyao): Without chunking, SGLang always admits the first prefill request even
        # when it exceeds the batch token budget, up to the model context bound.
        candidates.append(server_args.context_length)
    candidates.append(server_args.max_running_requests)
    candidates.append(server_args.cuda_graph_config.decode.max_bs)
    candidates.append(server_args.cuda_graph_config.prefill.max_bs)

    positive = [int(value) for value in candidates if value is not None and value > 0]
    if not positive:
        raise ValueError(
            "Cannot derive hidden capture capacity: none of chunked_prefill_size, "
            "max_prefill_tokens, context_length, max_running_requests, or the "
            "CUDA graph batch maxima is positive"
        )
    return max(positive)


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
    """Create SGLang worker, memory pools, and tree cache."""
    # ModelRunner.__init__ publishes server_args as the process-wide runtime
    # context; publishing again would silently reconfigure whatever already runs
    # here, so an engine is only built where the context is unpublished. A
    # construction that failed after publishing is therefore not retried here.
    from sglang.srt.runtime_context import get_context

    from sglang_omni.model_runner.model_worker import ModelWorker, ModelWorkerConfig
    from sglang_omni.scheduling.sglang_backend import create_tree_cache

    if get_context().is_config_namespace_published("model"):
        raise RuntimeError(
            "this process already holds a published SGLang runtime context; "
            "an SGLang AR engine must own its OS process. Place SGLang AR "
            "stages in separate processes."
        )

    logger.info(_describe_sglang_runtime_configuration(server_args, gpu_id))

    worker_config = ModelWorkerConfig(
        model_arch_override=model_arch_override,
        weight_prefix=weight_prefix,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        enable_prefill_input_embeds=enable_prefill_input_embeds,
    )
    from sglang.srt.utils.tensor_bridge import use_mlx

    if use_mlx():
        from sglang_omni.model_runner.mlx_model_worker import create_mlx_model_worker

        model_worker = create_mlx_model_worker(
            config=worker_config,
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
        )
    else:
        model_worker = ModelWorker(
            config=worker_config,
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
        )

    if capture_hidden_layers:
        from sglang_omni.model_runner._hidden_capture import (
            install_hidden_capture_hooks,
        )

        model = model_worker.model_runner.model
        install_hidden_capture_hooks(
            model,
            capture_hidden_layers,
            max_tokens=_hidden_capture_max_tokens(server_args),
        )

    # Phase order follows upstream Scheduler.init_model_worker().
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
    server_args: Any,
    gpu_id: int,
    **kwargs: Any,
):
    """Build shared SGLang infrastructure while deferring CUDA graph capture.

    The caller finishes stage-specific decode setup, then runs
    init_cuda_graphs() only when this returns that CUDA graphs were requested.
    """
    want_cuda_graph = not bool(server_args.disable_cuda_graph)
    infrastructure = create_sglang_infrastructure(
        server_args,
        gpu_id,
        defer_cuda_graph_capture=want_cuda_graph,
        **kwargs,
    )
    return want_cuda_graph, infrastructure
