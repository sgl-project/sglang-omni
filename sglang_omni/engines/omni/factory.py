# SPDX-License-Identifier: Apache-2.0
"""Factory functions for creating OmniEngine instances."""

from __future__ import annotations

import logging
from typing import Any

import torch

from .engine import OmniEngine
from .model_runner import ModelRunner
from .runtime.ar import (
    ARBatchPlanner,
    ARInputPreparer,
    AROutputProcessor,
    ARResourceManager,
)
from .runtime.cache import SimpleCacheManager
from .runtime.common import (
    EosIterationController,
    SimpleResourceManager,
    SinglePassIterationController,
)
from .runtime.encoder import (
    EncoderBatchPlanner,
    EncoderInputPreparer,
    EncoderOutputProcessor,
)
from .scheduler import Scheduler

logger = logging.getLogger(__name__)


def create_encoder_engine(
    model: torch.nn.Module,
    tokenizer: Any = None,
    max_batch_size: int = 32,
    pooling: str = "last",
    device: str = "cuda",
    use_cache: bool = False,
    cache_size: int | None = None,
) -> OmniEngine:
    """Create an encoder engine.

    Args:
        model: The encoder model (e.g., BERT, RoBERTa)
        tokenizer: Optional tokenizer (used to get pad_token_id)
        max_batch_size: Maximum batch size for scheduling
        pooling: Pooling strategy - "last", "mean", or "cls"
        device: Device to run on
        use_cache: Enable encoder output cache
        cache_size: Max cache entries (None for unbounded)

    Returns:
        OmniEngine configured for encoder models

    Example:
        from transformers import BertModel, BertTokenizer

        model = BertModel.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        engine = create_encoder_engine(model, tokenizer)
        await engine.start()

        # Create request data
        input_ids = tokenizer.encode("Hello world", return_tensors="pt")
        data = EncoderRequestData(input_ids=input_ids[0])

        await engine.add_request("req-1", data)
        result = await engine.get_result("req-1")  # Returns embeddings tensor
    """
    # Get pad_token_id from tokenizer if available
    pad_token_id = 0
    if tokenizer is not None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0

    scheduler = Scheduler(
        batch_planner=EncoderBatchPlanner(max_batch_size=max_batch_size),
        resource_manager=SimpleResourceManager(max_count=max_batch_size),
        iteration_controller=SinglePassIterationController(),
    )

    # Create model runner (stateless)
    model_runner = ModelRunner(
        model=model,
        input_preparer=EncoderInputPreparer(pad_token_id=pad_token_id),
        output_processor=EncoderOutputProcessor(pooling=pooling),
        device=device,
    )

    # Create cache manager (if needed)
    cache_manager = None
    if use_cache:
        cache_manager = SimpleCacheManager(max_size=cache_size)

    return OmniEngine(
        scheduler=scheduler,
        model_runner=model_runner,
        cache_manager=cache_manager,
        enable_overlap=False,
    )


# Alias: generic name for non-AR single-pass engines
create_single_pass_engine = create_encoder_engine


def create_ar_engine(
    model: torch.nn.Module,
    tokenizer: Any = None,
    max_seq_len: int = 2048,
    device: str = "cuda",
    enable_overlap: bool = False,
) -> OmniEngine:
    """Create an AR engine (single request, HF KV cache).

    Args:
        model: The causal LM model (e.g., LLaMA, GPT-2)
        tokenizer: Tokenizer (used to get eos_token_id)
        max_seq_len: Maximum sequence length
        device: Device to run on
        enable_overlap: Enable overlap scheduling (GPU/CPU pipelining)

    Returns:
        OmniEngine configured for AR models

    Example:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        engine = create_ar_engine(model, tokenizer, enable_overlap=True)
        await engine.start()

        # Create request data
        input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
        data = ARRequestData(
            input_ids=input_ids[0],
            max_new_tokens=256,
            temperature=0.7,
        )

        await engine.add_request("req-1", data)
        result = await engine.get_result("req-1")  # ARRequestData with output_ids

        generated_text = tokenizer.decode(result.output_ids)
    """
    # Get eos_token_id from tokenizer
    eos_token_id = 2
    if tokenizer is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None) or 2

    def _stream_adapter(request, output):
        token = output.data
        if isinstance(token, tuple):
            token = token[0]
        if token is None:
            return None
        return int(token)

    scheduler = Scheduler(
        batch_planner=ARBatchPlanner(),
        resource_manager=ARResourceManager(max_count=1),
        iteration_controller=EosIterationController(
            eos_token_id=eos_token_id,
            max_length=max_seq_len,
        ),
        stream_adapter=_stream_adapter,
    )

    # Create model runner
    model_runner = ModelRunner(
        model=model,
        input_preparer=ARInputPreparer(),
        output_processor=AROutputProcessor(),
        device=device,
    )

    return OmniEngine(
        scheduler=scheduler,
        model_runner=model_runner,
        enable_overlap=enable_overlap,
    )


def create_sglang_ar_engine(
    server_args: Any,
    gpu_id: int = 0,
    enable_overlap: bool | None = None,
    stream_adapter=None,
    capture_hidden: bool = False,
    capture_hidden_layers: list[int] | None = None,
    model_arch_override: str | None = None,
    weight_prefix: str | None = None,
    feedback_enabled: bool = False,
    feedback_mailbox: Any | None = None,
) -> OmniEngine:
    """Create an AR engine backed by SGLang's ModelWorker and KV cache.

    Uses SGLang's PrefillManager, DecodeManager, and paged KV cache for
    continuous batching with chunked prefill support.

    Args:
        server_args: SGLang ServerArgs configuration
        gpu_id: GPU device ID
        enable_overlap: Enable overlap scheduling. If None, uses the value
                       from server_args (not server_args.disable_overlap_schedule).

    Returns:
        OmniEngine configured with SGLang backend
    """
    from sglang_omni.engines.ar.sglang_backend.model_worker import (
        ModelWorker,
        ModelWorkerConfig,
    )
    from sglang_omni.engines.ar.sglang_backend.scheduler.cache import create_tree_cache
    from sglang_omni.engines.ar.sglang_backend.scheduler.decode import DecodeManager
    from sglang_omni.engines.ar.sglang_backend.scheduler.prefill import PrefillManager

    from .runtime.sglang_ar import (
        SGLangBatchPlanner,
        SGLangIterationController,
        SGLangModelRunner,
        SGLangOutputProcessor,
        SGLangResourceManager,
    )

    # Determine overlap setting
    if enable_overlap is None:
        enable_overlap = not getattr(server_args, "disable_overlap_schedule", False)

    # Feedback-gated engines (e.g. Talker AR) require synchronous step processing
    # because overlap scheduling can execute one step ahead of the feedback check.
    if feedback_enabled and enable_overlap:
        logger.debug("Disabling overlap for feedback-enabled engine")
        enable_overlap = False

    # Initialize model worker
    model_worker = ModelWorker(
        config=ModelWorkerConfig(
            model_arch_override=model_arch_override,
            weight_prefix=weight_prefix,
        ),
        server_args=server_args,
        gpu_id=gpu_id,
    )

    # Configure multi-layer hidden state capture (e.g., [0, 24] for embed + layer-24)
    # Uses forward hooks to capture intermediate hidden states from transformer layers.
    # The VL wrapper doesn't support layers_to_capture directly, so we hook into the
    # text model's layers and store captured states on the model instance.
    if capture_hidden_layers:
        from .runtime._hidden_capture import install_hidden_capture_hooks

        model = model_worker.model_runner.model
        install_hidden_capture_hooks(model, capture_hidden_layers)

    # Get memory pools
    req_to_token_pool, token_to_kv_pool_allocator = model_worker.get_memory_pool()

    # Create tree cache
    tree_cache = create_tree_cache(
        server_args,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        server_args.page_size,
    )

    # Create prefill and decode managers
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

    # Assemble SGLang-specific components
    batch_planner = SGLangBatchPlanner(prefill_mgr, decode_mgr, server_args)
    resource_mgr = SGLangResourceManager(
        token_to_kv_pool_allocator, req_to_token_pool, tree_cache
    )
    iteration_ctrl = SGLangIterationController(
        tree_cache, feedback_enabled=feedback_enabled
    )
    output_proc = SGLangOutputProcessor(
        capture_hidden=capture_hidden,
        capture_hidden_layers=capture_hidden_layers,
        model=model_worker.model_runner.model if capture_hidden_layers else None,
    )

    if stream_adapter is None:

        def stream_adapter(request, output):
            if request.data.req.is_chunked > 0:
                return None
            token = output.data
            return int(token) if token is not None else None

    scheduler = Scheduler(
        batch_planner=batch_planner,
        resource_manager=resource_mgr,
        iteration_controller=iteration_ctrl,
        stream_adapter=stream_adapter,
    )
    # Wire abort callback so feedback-timeout aborts go through the
    # scheduler's proper cleanup path (free resources, notify waiters).
    batch_planner._abort_callback = scheduler.abort_request
    sglang_model_runner = SGLangModelRunner(
        model_worker, output_proc, batch_planner=batch_planner
    )

    return OmniEngine(
        scheduler=scheduler,
        model_runner=sglang_model_runner,
        enable_overlap=enable_overlap,
        feedback_mailbox=feedback_mailbox,
    )
