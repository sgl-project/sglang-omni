# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Ming-Omni-TTS 16.8B pipeline."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.models.ming_tts.audio_config import resolve_ming_tts_audio_vae_config
from sglang_omni.models.ming_tts.config import (
    MING_TTS_AUDIO_DECODE_MAX_BATCH_SIZE,
    MING_TTS_AUDIO_DECODE_MAX_BATCH_WAIT_MS,
    MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES,
    MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES,
    validate_ming_tts_audio_decode_batch_config,
    validate_ming_tts_audio_decode_cadence_config,
)
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
    keep_latents: bool = False,
    initial_chunk_patches: int = MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES,
    steady_chunk_patches: int = MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES,
    max_batch_size: int = MING_TTS_AUDIO_DECODE_MAX_BATCH_SIZE,
    max_batch_wait_ms: int = MING_TTS_AUDIO_DECODE_MAX_BATCH_WAIT_MS,
) -> Any:
    validate_ming_tts_audio_decode_cadence_config(
        initial_chunk_patches=initial_chunk_patches,
        steady_chunk_patches=steady_chunk_patches,
    )
    max_batch_size, max_batch_wait_ms = validate_ming_tts_audio_decode_batch_config(
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )

    from sglang_omni.models.ming_tts.audio_decode import MingAudioDecoder
    from sglang_omni.models.ming_tts.streaming_vocoder import (
        MingTTSStreamingVocoderScheduler,
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

    logger.info(
        "Ming-Omni-TTS AudioVAE streaming cadence: "
        "initial_patches=%d steady_patches=%d",
        initial_chunk_patches,
        steady_chunk_patches,
    )
    return MingTTSStreamingVocoderScheduler(
        decoder,
        patch_size=int(config.audio_patch_size),
        latent_dim=int(config.latent_dim),
        initial_chunk_patches=initial_chunk_patches,
        steady_chunk_patches=steady_chunk_patches,
        keep_latents=keep_latents,
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
