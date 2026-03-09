# SPDX-License-Identifier: Apache-2.0
"""Factory function for creating S2-Pro (FishQwen3OmniForCausalLM) engines."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.engines.omni.engine import OmniEngine
from sglang_omni.engines.omni.scheduler import Scheduler

from .runtime.s2pro_sglang_ar import (
    S2ProSGLangIterationController,
    S2ProSGLangModelRunner,
    S2ProSGLangOutputProcessor,
    S2ProSGLangResourceManager,
)
from .tokenizer import S2ProTokenizerAdapter


def _patch_fish_config_for_sglang(model_path: str) -> None:
    """Patch FishQwen3Config to add standard HF attribute aliases for SGLang."""
    import sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.modeling  # registers AutoConfig
    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.modeling import (
        FishQwen3Config,
        FishQwen3OmniConfig,
    )

    if hasattr(FishQwen3Config, "_sglang_patched"):
        return

    # Patch text config: add standard HF attribute aliases
    original_init = FishQwen3Config.__init__

    def _patched_text_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.num_attention_heads = self.n_head
        self.hidden_size = self.dim
        self.num_hidden_layers = self.n_layer
        self.num_key_value_heads = self.n_local_heads
        # S2-pro is trained in bf16
        self.torch_dtype = torch.bfloat16
        if self.architectures is None:
            self.architectures = ["S2ProSGLangTextModel"]

    FishQwen3Config.__init__ = _patched_text_init
    FishQwen3Config._sglang_patched = True

    # Patch top-level config: add architectures for ModelConfig
    original_omni_init = FishQwen3OmniConfig.__init__

    def _patched_omni_init(self, *args, **kwargs):
        original_omni_init(self, *args, **kwargs)
        if self.architectures is None:
            self.architectures = ["S2ProSGLangTextModel"]

    FishQwen3OmniConfig.__init__ = _patched_omni_init


def _truncate_rope_to_bf16(model: torch.nn.Module) -> None:
    # Match fish_speech's bf16 RoPE training precision to avoid logit divergence
    for module in model.modules():
        if hasattr(module, "cos_sin_cache"):
            module.cos_sin_cache.data = module.cos_sin_cache.data.to(torch.bfloat16).to(
                torch.float32
            )


def create_s2pro_sglang_engine(
    server_args: Any,
    audio_decoder: torch.nn.Module,
    tokenizer: Any = None,
    *,
    gpu_id: int = 0,
    num_codebooks: int = 10,
    codebook_size: int = 4096,
    max_new_tokens: int = 2048,
    top_k: int = 30,
    ras_window: int = 16,
    ras_temperature: float = 1.5,
    ras_top_p: float = 0.95,
    use_torch_compile: bool = True,
) -> OmniEngine:
    """Create a paged-attention S2-Pro engine using SGLang backend."""
    from sglang_omni.engines.ar.sglang_backend.model_worker import (
        ModelWorker,
        ModelWorkerConfig,
    )
    from sglang_omni.engines.ar.sglang_backend.scheduler.cache import create_tree_cache
    from sglang_omni.engines.ar.sglang_backend.scheduler.decode import DecodeManager
    from sglang_omni.engines.ar.sglang_backend.scheduler.prefill import PrefillManager
    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangBatchPlanner

    _patch_fish_config_for_sglang(server_args.model_path)

    # Use FlashAttention backend (fa3) to match fish_speech's flash_attn_with_kvcache. Refer to day0 support blog
    if server_args.attention_backend is None:
        server_args.attention_backend = "fa3"

    adapter = S2ProTokenizerAdapter(tokenizer)
    im_end_id = adapter.eos_token_ids[0]
    semantic_begin_id = adapter.semantic_begin_id
    semantic_end_id = adapter.semantic_end_id

    # Initialize SGLang model worker (loads text model with RadixAttention)
    model_worker = ModelWorker(
        config=ModelWorkerConfig(),
        server_args=server_args,
        gpu_id=gpu_id,
    )

    # Match fish_speech's bf16 RoPE precision. Refer to day0 support blog
    _truncate_rope_to_bf16(model_worker.model_runner.model)

    # Get memory pools
    req_to_token_pool, token_to_kv_pool_allocator = model_worker.get_memory_pool()

    # Create tree cache for prefix caching
    tree_cache = create_tree_cache(
        server_args,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        server_args.page_size,
    )

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

    # Assemble S2-Pro SGLang components
    batch_planner = SGLangBatchPlanner(prefill_mgr, decode_mgr, server_args)
    resource_mgr = S2ProSGLangResourceManager(
        token_to_kv_pool_allocator, req_to_token_pool, tree_cache
    )
    output_processor = S2ProSGLangOutputProcessor(
        audio_decoder=audio_decoder,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        semantic_begin_id=semantic_begin_id,
        semantic_end_id=semantic_end_id,
        im_end_id=im_end_id,
        top_k=top_k,
        ras_window=ras_window,
        ras_temperature=ras_temperature,
        ras_top_p=ras_top_p,
        use_torch_compile=use_torch_compile,
    )
    iteration_ctrl = S2ProSGLangIterationController(
        tree_cache=tree_cache,
        im_end_token_id=im_end_id,
        max_new_tokens=max_new_tokens,
    )

    def _stream_adapter(request, output):
        step_out = output.data
        if step_out is None:
            return None
        return step_out.codes

    scheduler = Scheduler(
        batch_planner=batch_planner,
        resource_manager=resource_mgr,
        iteration_controller=iteration_ctrl,
        stream_adapter=_stream_adapter,
    )
    model_runner = S2ProSGLangModelRunner(model_worker, output_processor, batch_planner)

    return OmniEngine(scheduler=scheduler, model_runner=model_runner)
