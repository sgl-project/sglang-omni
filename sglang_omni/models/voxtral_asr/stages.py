# SPDX-License-Identifier: Apache-2.0
"""Stage factory for SGLang-backed Voxtral realtime ASR inference."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from sglang.srt.managers.mm_utils import init_mm_embedding_cache

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.voxtral_asr.model_config import VoxtralRealtimeConfig
from sglang_omni.models.voxtral_asr.request_builders import (
    make_voxtral_asr_scheduler_adapters,
)
from sglang_omni.scheduling.bootstrap import (
    create_sglang_infrastructure_defer_cuda_graph,
    init_sglang_cuda_graphs,
)
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import (
    SGLangOutputProcessor,
    build_sglang_server_args,
)


def _import_mistral_tokenizer(model_path: str) -> Any:
    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Voxtral ASR requires the `mistral_common` package. "
            "Install it with: pip install 'mistral_common[audio]>=1.11.0'"
        ) from exc
    tekken_path = os.path.join(model_path, "tekken.json")
    return MistralTokenizer.from_file(tekken_path)


def _write_voxtral_asr_config(checkpoint_dir: str) -> str:
    """Write a temporary HF-style config.json so SGLang can bootstrap.

    Voxtral realtime checkpoints ship with ``params.json`` instead of a HF
    ``config.json``.  We expose the text backbone dimensions to SGLang and
    the model class itself re-parses ``params.json`` for the audio encoder.
    """
    cfg = VoxtralRealtimeConfig.from_model_path(checkpoint_dir).text_config
    path = os.path.join(
        tempfile.gettempdir(),
        f"voxtral_asr_sglang_config_{abs(hash(checkpoint_dir))}.json",
    )
    data = {
        "model_type": "llama",
        "architectures": ["VoxtralRealtimeForConditionalGeneration"],
        "hidden_size": cfg.dim,
        "intermediate_size": cfg.hidden_dim,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_seq_len,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.norm_eps,
        "tie_word_embeddings": cfg.tied_embeddings,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def create_sglang_voxtral_asr_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_running_requests: int = 32,
    max_new_tokens: int = 4096,
    mem_fraction_static: float | None = None,
    mm_embedding_cache_size_bytes: int = 0,
    enable_torch_compile: bool = False,
    mm_attention_backend: str | None = None,
    request_build_max_workers: int = 2,
    request_build_max_pending: int | None = 16,
    server_args_overrides: dict[str, Any] | None = None,
):
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    if os.path.isdir(model_path):
        checkpoint_dir = model_path
    else:
        from huggingface_hub import snapshot_download

        checkpoint_dir = snapshot_download(model_path)
    tokenizer = _import_mistral_tokenizer(checkpoint_dir)
    decrypted_config_file = _write_voxtral_asr_config(checkpoint_dir)
    voxtral_config = VoxtralRealtimeConfig.from_model_path(checkpoint_dir)

    # Audio encoder turns ~30s of audio into at most ~750 pooled tokens;
    # with block_pool_size=4 that is ~3000 frames after conv.  We size the
    # context generously for long-form offline transcription.
    max_source_positions = voxtral_config.audio_config.max_source_positions or 1500
    audio_encoder_token_count = (
        max_source_positions * voxtral_config.audio_config.block_pool_size
    )

    defaults: dict[str, Any] = {
        "disable_cuda_graph": False,
        "disable_overlap_schedule": True,
        "enable_torch_compile": enable_torch_compile,
        "mem_fraction_static": mem_fraction_static,
        "max_prefill_tokens": 4096,
        "chunked_prefill_size": 4096,
        "sampling_backend": "pytorch",
        "dtype": dtype,
    }
    if mm_attention_backend is not None:
        defaults["mm_attention_backend"] = mm_attention_backend

    overrides = build_generation_batch_overrides(
        max_running_requests=max_running_requests,
        server_args_overrides=server_args_overrides,
        **defaults,
    )
    overrides["decrypted_config_file"] = decrypted_config_file

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=audio_encoder_token_count + int(max_new_tokens) + 8,
        **overrides,
    )
    validate_generation_batch_policy(
        model_name="Voxtral-ASR",
        server_args=server_args,
    )

    want_cuda_graph, (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        model_config,
    ) = create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        gpu_id,
        model_arch_override="VoxtralRealtimeForConditionalGeneration",
    )

    if want_cuda_graph:
        init_sglang_cuda_graphs(model_worker)

    init_mm_embedding_cache(mm_embedding_cache_size_bytes)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    request_builder, result_adapter = make_voxtral_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        model_runner=ModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
        request_build_max_workers=request_build_max_workers,
        request_build_max_pending=request_build_max_pending,
    )


def create_voxtral_asr_executor(*args, **kwargs):
    return create_sglang_voxtral_asr_executor(*args, **kwargs)


__all__ = ["create_sglang_voxtral_asr_executor", "create_voxtral_asr_executor"]
