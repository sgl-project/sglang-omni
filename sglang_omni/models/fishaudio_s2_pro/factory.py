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

    original_init = FishQwen3Config.__init__

    def _patched_text_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.num_attention_heads = self.n_head
        self.hidden_size = self.dim
        self.num_hidden_layers = self.n_layer
        self.num_key_value_heads = self.n_local_heads
        self.torch_dtype = torch.bfloat16
        if self.architectures is None:
            self.architectures = ["S2ProSGLangTextModel"]

    FishQwen3Config.__init__ = _patched_text_init
    FishQwen3Config._sglang_patched = True

    original_omni_init = FishQwen3OmniConfig.__init__

    def _patched_omni_init(self, *args, **kwargs):
        original_omni_init(self, *args, **kwargs)
        if self.architectures is None:
            self.architectures = ["S2ProSGLangTextModel"]

    FishQwen3OmniConfig.__init__ = _patched_omni_init


def _truncate_rope_to_bf16(model: torch.nn.Module) -> None:
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
) -> OmniEngine:
    """Create a unified S2-Pro engine (slow+fast head in one CUDA graph)."""
    from sglang_omni.engines.ar.sglang_backend.model_worker import (
        ModelWorker,
        ModelWorkerConfig,
    )
    from sglang_omni.engines.ar.sglang_backend.scheduler.cache import create_tree_cache
    from sglang_omni.engines.ar.sglang_backend.scheduler.decode import DecodeManager
    from sglang_omni.engines.ar.sglang_backend.scheduler.prefill import PrefillManager
    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangBatchPlanner

    _patch_fish_config_for_sglang(server_args.model_path)

    if server_args.attention_backend is None:
        server_args.attention_backend = "fa3"

    # Enable hidden state capture for unified decode
    want_cuda_graph = not server_args.disable_cuda_graph
    if want_cuda_graph:
        server_args.enable_return_hidden_states = True

    adapter = S2ProTokenizerAdapter(tokenizer)
    im_end_id = adapter.eos_token_ids[0]
    semantic_begin_id = adapter.semantic_begin_id
    semantic_end_id = adapter.semantic_end_id

    # Defer CUDA graph capture: ModelWorker.__init__ captures graphs, but
    # setup_vq_decode (which attaches the audio decoder / codebook loop)
    # must run first so the graph includes _decode_codebooks.
    server_args.disable_cuda_graph = True
    model_worker = ModelWorker(
        config=ModelWorkerConfig(),
        server_args=server_args,
        gpu_id=gpu_id,
    )
    server_args.disable_cuda_graph = not want_cuda_graph

    _truncate_rope_to_bf16(model_worker.model_runner.model)

    # Set up unified decode: slow head + fast head in one forward
    text_model = model_worker.model_runner.model
    max_bs = server_args.max_running_requests
    audio_decoder.setup_caches(max_batch_size=max_bs, dtype=torch.bfloat16)
    text_model.setup_vq_decode(
        audio_decoder,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        semantic_begin_id=semantic_begin_id,
        semantic_end_id=semantic_end_id,
        im_end_id=im_end_id,
        max_batch_size=max_bs,
    )

    # Now capture CUDA graphs with _decode_codebooks in the graph
    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    req_to_token_pool, token_to_kv_pool_allocator = model_worker.get_memory_pool()

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

    batch_planner = SGLangBatchPlanner(prefill_mgr, decode_mgr, server_args)
    resource_mgr = S2ProSGLangResourceManager(
        token_to_kv_pool_allocator, req_to_token_pool, tree_cache
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
    model_runner = S2ProSGLangModelRunner(
        model_worker,
        batch_planner,
        semantic_begin_id,
        semantic_end_id,
    )

    return OmniEngine(scheduler=scheduler, model_runner=model_runner)
