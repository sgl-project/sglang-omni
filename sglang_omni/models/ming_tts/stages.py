# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Ming-Omni-TTS 16.8B pipeline."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.models.ming_tts.audio_config import resolve_ming_tts_audio_vae_config
from sglang_omni.models.ming_tts.hf_config import (
    MING_TTS_AUDIO_VAE_ATTN_IMPLEMENTATION,
    register_ming_tts_hf_config,
)
from sglang_omni.models.ming_tts.request_builders import preprocess_ming_tts_payload
from sglang_omni.models.ming_tts.tokenizer import load_ming_tts_tokenizer
from sglang_omni.models.ming_tts.weight_loading import load_ming_tts_audio_vae_weights
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint

logger = logging.getLogger(__name__)


def _check_ming_tts_tp_backbone_config(config: Any, tp_size: int) -> None:
    tp_size = int(tp_size)
    if tp_size <= 1:
        return

    llm_config = getattr(config, "llm_config", None)
    if llm_config is None:
        raise ValueError("Ming-Omni-TTS TP requires llm_config")

    def require_int_config_field(field: str) -> int:
        value = getattr(llm_config, field, None)
        if value is None:
            raise ValueError(f"Ming-Omni-TTS llm_config is missing {field}")
        return int(value)

    hidden_size = require_int_config_field("hidden_size")
    head_dim = require_int_config_field("head_dim")
    num_heads = require_int_config_field("num_attention_heads")
    num_kv_heads = require_int_config_field("num_key_value_heads")
    if min(hidden_size, head_dim, num_heads, num_kv_heads) <= 0:
        raise ValueError(
            "Ming-Omni-TTS TP requires positive hidden/head dimensions: "
            f"hidden_size={hidden_size}, head_dim={head_dim}, "
            f"num_attention_heads={num_heads}, "
            f"num_key_value_heads={num_kv_heads}"
        )
    if head_dim * num_heads != hidden_size:
        raise ValueError(
            "Ming-Omni-TTS TP requires head_dim * num_attention_heads "
            f"to equal hidden_size ({head_dim} * {num_heads} != {hidden_size})"
        )
    if hidden_size % tp_size != 0:
        raise ValueError(
            "Ming-Omni-TTS TP requires hidden_size divisible by tp_size: "
            f"hidden_size={hidden_size}, tp_size={tp_size}"
        )
    if num_heads % tp_size != 0:
        raise ValueError(
            "Ming-Omni-TTS TP requires attention heads divisible by tp_size: "
            f"num_attention_heads={num_heads}, tp_size={tp_size}"
        )
    if num_kv_heads >= tp_size and num_kv_heads % tp_size != 0:
        raise ValueError(
            "Ming-Omni-TTS TP requires KV heads divisible by tp_size: "
            f"num_key_value_heads={num_kv_heads}, tp_size={tp_size}"
        )
    if num_kv_heads < tp_size and tp_size % num_kv_heads != 0:
        raise ValueError(
            "Ming-Omni-TTS TP requires KV heads to divide or be divisible "
            f"by tp_size: num_key_value_heads={num_kv_heads}, "
            f"tp_size={tp_size}"
        )


def create_preprocessing_executor(
    model_path: str,
    *,
    context_length: int | None = None,
    max_decode_steps_cap: int | None = None,
    max_concurrency: int = 1,
) -> SimpleScheduler:
    checkpoint_dir = _resolve_checkpoint(model_path)
    config = _load_ming_tts_config(checkpoint_dir)
    context_length = int(context_length or _resolve_context_length(config))
    tokenizer = load_ming_tts_tokenizer(
        checkpoint_dir,
        llm_config=config.llm_config,
    )

    def _preprocess(payload):
        return preprocess_ming_tts_payload(
            payload,
            tokenizer=tokenizer,
            context_length=context_length,
            max_decode_steps_cap=max_decode_steps_cap,
        )

    return SimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    context_length: int | None = None,
    server_args_overrides: dict[str, Any] | None = None,
    total_gpu_memory_fraction: float | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
) -> Any:
    from sglang_omni.models.ming_tts.engine_builder import MingTtsEngineBuilder

    user_overrides = dict(server_args_overrides or {})
    if "tp_size" in user_overrides and int(user_overrides["tp_size"]) != int(tp_size):
        raise ValueError(
            "Ming-Omni-TTS tts_engine tp_size conflicts with "
            f"server_args_overrides.tp_size={user_overrides['tp_size']!r}"
        )
    context_length = int(user_overrides.pop("context_length", context_length or 0) or 0)

    return MingTtsEngineBuilder(
        context_length=context_length or None,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        tp_rank=tp_rank,
        tp_size=tp_size,
        nccl_port=nccl_port,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=user_overrides,
    )


def create_tts_engine_executor(*args, **kwargs) -> Any:
    return create_sglang_tts_engine_executor(*args, **kwargs)


def create_reference_encode_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    context_length: int | None = None,
    max_concurrency: int = 1,
    ref_audio_cache: bool = True,
    ref_audio_cache_max_items: int = 256,
    ref_audio_cache_max_bytes: int = 64 * 1024 * 1024,
) -> SimpleScheduler:
    from sglang_omni.models.ming_tts.reference_encode import MingTTSReferenceEncoder

    checkpoint_dir = _resolve_checkpoint(model_path)
    config = _load_ming_tts_config(checkpoint_dir)
    context_length = int(context_length or _resolve_context_length(config))
    tokenizer = load_ming_tts_tokenizer(
        checkpoint_dir,
        llm_config=config.llm_config,
    )
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"

    audio_config = resolve_ming_tts_audio_vae_config(
        config.audio_tokenizer_config,
        attn_implementation=MING_TTS_AUDIO_VAE_ATTN_IMPLEMENTATION,
    )
    encoder = MingTTSReferenceEncoder.from_config(
        audio_config,
        checkpoint_dir=checkpoint_dir,
        device=device,
        dtype=dtype,
        patch_size=int(config.ditar_config["patch_size"]),
        ref_audio_cache=ref_audio_cache,
        ref_audio_cache_max_items=ref_audio_cache_max_items,
        ref_audio_cache_max_bytes=ref_audio_cache_max_bytes,
    )
    report = load_ming_tts_audio_vae_weights(checkpoint_dir, encoder.audio_vae)
    logger.info("%s", report.summary())

    def _encode(payload):
        return encoder.encode_payload(
            payload,
            tokenizer=tokenizer,
            context_length=context_length,
        )

    return SimpleScheduler(_encode, max_concurrency=max_concurrency)


def create_audio_decode_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    decode_mode: str = "chunked",
    keep_latents: bool = False,
    max_batch_size: int = 1,
    max_batch_wait_ms: int = 0,
) -> SimpleScheduler:
    if decode_mode != "chunked":
        raise ValueError("Ming-Omni-TTS currently supports only decode_mode='chunked'")

    from sglang_omni.models.ming_tts.audio_decode import (
        MingAudioDecoder,
        MingTTSBatchVocoder,
    )

    checkpoint_dir = _resolve_checkpoint(model_path)
    config = _load_ming_tts_config(checkpoint_dir)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"

    audio_config = resolve_ming_tts_audio_vae_config(
        config.audio_tokenizer_config,
        attn_implementation=MING_TTS_AUDIO_VAE_ATTN_IMPLEMENTATION,
    )
    decoder = MingAudioDecoder.from_config(
        audio_config,
        device=device,
        dtype=dtype,
    )
    report = load_ming_tts_audio_vae_weights(checkpoint_dir, decoder.audio_vae)
    logger.info("%s", report.summary())

    vocoder = MingTTSBatchVocoder(
        decoder,
        decode_mode=decode_mode,
        keep_latents=keep_latents,
    )
    return vocoder.build_scheduler(
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


def _load_ming_tts_config(model_path: str) -> Any:
    register_ming_tts_hf_config()
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path, trust_remote_code=False)


def _resolve_context_length(config: Any) -> int:
    llm_config = config.llm_config
    value = getattr(llm_config, "max_position_embeddings", None)
    if value is None:
        raise ValueError("Ming-Omni-TTS llm_config is missing max_position_embeddings")
    return int(value)


__all__ = [
    "create_audio_decode_executor",
    "create_preprocessing_executor",
    "create_reference_encode_executor",
    "create_sglang_tts_engine_executor",
    "create_tts_engine_executor",
]
