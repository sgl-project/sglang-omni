# SPDX-License-Identifier: Apache-2.0
"""Factory functions for creating MiniCPM-V SGLang engines.

This module provides config patching and engine creation for MiniCPM-V with
SGLang paged attention backend.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.engines.omni.engine import OmniEngine
from sglang_omni.engines.omni.scheduler import Scheduler

logger = logging.getLogger(__name__)


def _patch_minicpm_config_for_sglang(model_path: str) -> None:
    """Patch MiniCPM-V config to add standard HF attribute aliases for SGLang.

    SGLang expects certain standard HF config attributes (num_attention_heads,
    hidden_size, image_token_id, etc.) that MiniCPM's config may structure differently.
    """
    from transformers import AutoConfig

    from sglang_omni.models.minicpm_v.components.common import get_image_token_id

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load config for patching: {e}")
        return

    # Get the image token ID for embedding injection
    try:
        image_token_id = get_image_token_id(model_path)
    except Exception:
        image_token_id = 151858  # MiniCPM-V default

    # Get the config class
    config_class = type(config)
    if hasattr(config_class, "_sglang_patched"):
        return

    # Get the LLM config (may be nested)
    llm_config = getattr(config, "llm_config", config)
    llm_config_class = type(llm_config)

    # Check if patching is needed
    if hasattr(llm_config_class, "_sglang_patched"):
        return

    original_init = llm_config_class.__init__

    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Ensure standard HF attribute aliases exist
        if not hasattr(self, "torch_dtype"):
            self.torch_dtype = torch.bfloat16
        # Add token IDs for SGLang model runner multimodal injection
        if not hasattr(self, "image_token_id"):
            self.image_token_id = image_token_id
        if not hasattr(self, "video_token_id"):
            self.video_token_id = -1  # Not supported in MiniCPM-V
        if not hasattr(self, "audio_token_id"):
            self.audio_token_id = -1  # Not supported in MiniCPM-V 2.6
        # Ensure architectures points to our SGLang model
        if not hasattr(self, "architectures") or self.architectures is None:
            self.architectures = ["MiniCPMVSGLangLLM"]
        elif "MiniCPMVSGLangLLM" not in self.architectures:
            # Keep original architecture for reference but add our SGLang model
            pass  # SGLang will be told to use our model via model_override_args

    llm_config_class.__init__ = _patched_init
    llm_config_class._sglang_patched = True

    # Mark main config as patched too if different
    if config_class is not llm_config_class:
        config_class._sglang_patched = True


def _truncate_rope_to_bf16(model: torch.nn.Module) -> None:
    """Truncate RoPE cos/sin cache to bf16 for numerical stability."""
    for module in model.modules():
        if hasattr(module, "cos_sin_cache"):
            module.cos_sin_cache.data = module.cos_sin_cache.data.to(torch.bfloat16).to(
                torch.float32
            )


def create_minicpm_sglang_engine(
    server_args: Any,
    tokenizer: Any,
    *,
    gpu_id: int = 0,
    max_new_tokens: int = 2048,
) -> OmniEngine:
    """Create a MiniCPM-V SGLang engine with paged attention.

    This creates a SGLang-backed LLM engine for MiniCPM-V with:
    - Continuous batching support
    - Paged attention KV cache
    - CUDA graph capture for decode

    Args:
        server_args: SGLang ServerArgs with model path and config
        tokenizer: HuggingFace tokenizer for the model
        gpu_id: GPU device ID to use
        max_new_tokens: Maximum tokens to generate per request

    Returns:
        OmniEngine configured for MiniCPM-V
    """
    from sglang_omni.engines.ar.sglang_backend.model_worker import (
        ModelWorker,
        ModelWorkerConfig,
    )
    from sglang_omni.engines.ar.sglang_backend.scheduler.cache import create_tree_cache
    from sglang_omni.engines.ar.sglang_backend.scheduler.decode import DecodeManager
    from sglang_omni.engines.ar.sglang_backend.scheduler.prefill import PrefillManager
    from sglang_omni.engines.omni.runtime.sglang_ar import (
        SGLangBatchPlanner,
        SGLangIterationController,
        SGLangModelRunner,
        SGLangResourceManager,
    )

    _patch_minicpm_config_for_sglang(server_args.model_path)

    if server_args.attention_backend is None:
        server_args.attention_backend = "flashinfer"

    # Model worker initialization
    model_worker = ModelWorker(
        config=ModelWorkerConfig(),
        server_args=server_args,
        gpu_id=gpu_id,
    )

    _truncate_rope_to_bf16(model_worker.model_runner.model)

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
        enable_overlap=False,
    )
    decode_mgr = DecodeManager(
        server_args=server_args,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        on_retract=lambda req: prefill_mgr.add_one_request(req),
    )

    # Create components for OmniEngine
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    batch_planner = SGLangBatchPlanner(prefill_mgr, decode_mgr, server_args)
    resource_mgr = SGLangResourceManager(
        token_to_kv_pool_allocator, req_to_token_pool, tree_cache
    )
    iteration_ctrl = SGLangIterationController(
        tree_cache=tree_cache,
        eos_token_ids=[eos_token_id] if eos_token_id else [],
        max_new_tokens=max_new_tokens,
    )

    scheduler = Scheduler(
        batch_planner=batch_planner,
        resource_manager=resource_mgr,
        iteration_controller=iteration_ctrl,
    )
    model_runner = SGLangModelRunner(
        model_worker=model_worker,
        batch_planner=batch_planner,
        embed_tokens=model_worker.model_runner.model.get_embed_tokens(),
    )

    return OmniEngine(scheduler=scheduler, model_runner=model_runner)


def create_minicpm_sglang_engine_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    context_length: int = 8192,
    max_new_tokens: int = 2048,
    server_args_overrides: dict[str, Any] | None = None,
) -> OmniEngine:
    """Create a MiniCPM-V SGLang engine from config args.

    This is the high-level entry point that builds ServerArgs internally.

    Args:
        model_path: Path to the MiniCPM-V model
        gpu_id: GPU device ID
        context_length: Maximum context length
        max_new_tokens: Maximum tokens to generate
        server_args_overrides: Additional ServerArgs overrides

    Returns:
        OmniEngine configured for MiniCPM-V
    """
    from transformers import AutoTokenizer

    from sglang_omni.engines.ar.sglang_backend.server_args_builder import (
        build_sglang_server_args,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    server_args = build_sglang_server_args(
        model_path,
        context_length=context_length,
        **(server_args_overrides or {}),
    )

    return create_minicpm_sglang_engine(
        server_args=server_args,
        tokenizer=tokenizer,
        gpu_id=gpu_id,
        max_new_tokens=max_new_tokens,
    )
