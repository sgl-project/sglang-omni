# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Ming-Omni-TTS 16.8B pipeline."""

from __future__ import annotations

import logging
import math
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sglang_omni.models.ming_tts.audio_config import resolve_ming_tts_audio_vae_config
from sglang_omni.models.ming_tts.config import (
    MING_TTS_AUDIO_DECODE_MAX_BATCH_SIZE,
    MING_TTS_AUDIO_DECODE_MAX_BATCH_WAIT_MS,
    MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES,
    MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES,
    MING_TTS_DEFAULT_STREAM_SLOTS,
    MING_TTS_DEFAULT_STREAMING_CUDA_GRAPH,
    validate_ming_tts_audio_decode_batch_config,
    validate_ming_tts_audio_decode_cadence_config,
    validate_ming_tts_audio_decode_stream_slots,
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
from sglang_omni.utils.gpu_memory import (
    format_bytes_gib,
    get_gpu_device_info,
    get_process_gpu_memory_bytes,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch

    from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import (
        AudioVAE,
    )
    from sglang_omni.models.ming_tts.audio_config import AudioVAEconfig


def _resolve_audio_vae_dtype(dtype: str | torch.dtype) -> torch.dtype:
    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        return torch.bfloat16
    if isinstance(dtype, str):
        value = dtype.removeprefix("torch.")
        torch_dtype = getattr(torch, value, None)
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        raise ValueError(f"Unsupported Ming-Omni-TTS AudioVAE dtype: {dtype!r}")
    raise TypeError(f"Unsupported Ming-Omni-TTS AudioVAE dtype: {dtype!r}")


def _load_ming_tts_audio_vae(
    checkpoint_dir: str,
    audio_config: AudioVAEconfig,
    *,
    device: str | torch.device,
    dtype: str | torch.dtype,
) -> AudioVAE:
    import torch

    from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import (
        AudioVAE,
    )

    if getattr(audio_config, "semantic_module_kwargs", None) is not None:
        raise ValueError(
            "Ming-Omni-TTS serving currently uses the talker AudioVAE "
            "encode/decode path and does not support semantic_module_kwargs"
        )

    audio_vae = AudioVAE(audio_config).eval()
    audio_vae.to(
        device=torch.device(device),
        dtype=_resolve_audio_vae_dtype(dtype),
    )
    report = load_ming_tts_audio_vae_weights(checkpoint_dir, audio_vae)
    logger.info("%s", report.summary())
    return audio_vae


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
    from sglang_omni.models.ming_tts.reference_encode import (
        MingSpeakerEmbeddingExtractor,
        MingTTSReferenceEncoder,
    )

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
    audio_vae = _load_ming_tts_audio_vae(
        checkpoint_dir,
        audio_config,
        device=device,
        dtype=dtype,
    )

    encoder = MingTTSReferenceEncoder(
        audio_vae,
        MingSpeakerEmbeddingExtractor(str(Path(checkpoint_dir) / "campplus.onnx")),
        patch_size=int(config.ditar_config["patch_size"]),
        cache_model_identity=str(checkpoint_dir) if ref_audio_cache else None,
        cache_max_items=ref_audio_cache_max_items,
        cache_max_bytes=ref_audio_cache_max_bytes,
    )

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
    streaming_cuda_graph: bool = MING_TTS_DEFAULT_STREAMING_CUDA_GRAPH,
    stream_slots: int = MING_TTS_DEFAULT_STREAM_SLOTS,
    max_batch_size: int = MING_TTS_AUDIO_DECODE_MAX_BATCH_SIZE,
    max_batch_wait_ms: int = MING_TTS_AUDIO_DECODE_MAX_BATCH_WAIT_MS,
    total_gpu_memory_fraction: float | None = None,
    process_total_gpu_memory_fraction: float | None = None,
) -> Any:
    validate_ming_tts_audio_decode_cadence_config(
        initial_chunk_patches=initial_chunk_patches,
        steady_chunk_patches=steady_chunk_patches,
    )
    validate_ming_tts_audio_decode_batch_config(
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
    validate_ming_tts_audio_decode_stream_slots(stream_slots)
    if not isinstance(streaming_cuda_graph, bool):
        raise ValueError("Ming-Omni-TTS streaming_cuda_graph must be a boolean")

    import torch

    from sglang_omni.models.ming_tts.audio_decode import MingAudioDecoder
    from sglang_omni.models.ming_tts.streaming_vocoder import (
        MingTTSStreamingVocoderScheduler,
    )

    component_fraction = total_gpu_memory_fraction
    process_fraction = process_total_gpu_memory_fraction
    for name, value in (
        ("total_gpu_memory_fraction", component_fraction),
        ("process_total_gpu_memory_fraction", process_fraction),
    ):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Ming-Omni-TTS {name} must be a real number")
        value = float(value)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"Ming-Omni-TTS {name} must be finite and in (0, 1]")
        if name == "total_gpu_memory_fraction":
            component_fraction = value
        else:
            process_fraction = value
    if streaming_cuda_graph and process_fraction is None:
        raise ValueError(
            "Ming-Omni-TTS streaming AudioVAE CUDA graph requires "
            "process_total_gpu_memory_fraction"
        )
    if (
        component_fraction is not None
        and process_fraction is not None
        and process_fraction < component_fraction
    ):
        raise ValueError(
            "Ming-Omni-TTS process_total_gpu_memory_fraction must be greater "
            "than or equal to total_gpu_memory_fraction"
        )

    if gpu_id is not None:
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int):
            raise TypeError("Ming-Omni-TTS gpu_id must be an integer")
        if gpu_id < 0:
            raise ValueError("Ming-Omni-TTS gpu_id must be non-negative")
        resolved_device = torch.device("cuda", gpu_id)
    else:
        try:
            resolved_device = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(
                f"Invalid Ming-Omni-TTS audio decode device: {device!r}"
            ) from exc
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(
            "Ming-Omni-TTS fixed AudioVAE serving requires an available CUDA device"
        )
    logical_gpu_id = resolved_device.index
    if logical_gpu_id is None:
        logical_gpu_id = torch.cuda.current_device()
    if logical_gpu_id < 0 or logical_gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"Ming-Omni-TTS audio decode GPU {logical_gpu_id} is not visible"
        )
    resolved_device = torch.device("cuda", logical_gpu_id)

    resolved_dtype = _resolve_audio_vae_dtype(dtype)
    if resolved_dtype != torch.bfloat16:
        raise ValueError(
            "Ming-Omni-TTS fixed AudioVAE serving requires bfloat16, "
            f"got {resolved_dtype}"
        )

    device_info = get_gpu_device_info(logical_gpu_id)
    pre_process_bytes = get_process_gpu_memory_bytes(logical_gpu_id)

    checkpoint_dir = _resolve_checkpoint(model_path)
    config = _load_ming_tts_config(checkpoint_dir)

    audio_config = resolve_ming_tts_audio_vae_config(
        config.audio_tokenizer_config,
        attn_implementation=MING_TTS_AUDIO_VAE_ATTN_IMPLEMENTATION,
    )
    patch_size = int(config.audio_patch_size)
    latent_dim = int(config.latent_dim)
    if patch_size <= 0 or latent_dim <= 0:
        raise ValueError(
            "Ming-Omni-TTS audio patch size and latent dimension must be positive"
        )
    decoder_latent_dim = audio_config.dec_kwargs.get("latent_dim")
    if decoder_latent_dim is None:
        raise ValueError("Ming-Omni-TTS AudioVAE decoder config is missing latent_dim")
    try:
        decoder_latent_dim = int(decoder_latent_dim)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Ming-Omni-TTS AudioVAE decoder latent_dim must be an integer"
        ) from exc
    if decoder_latent_dim != latent_dim:
        raise ValueError(
            "Ming-Omni-TTS upstream and AudioVAE decoder latent dimensions "
            f"must match, got {latent_dim} and {decoder_latent_dim}"
        )
    max_step_latents = max(initial_chunk_patches, steady_chunk_patches) * patch_size

    audio_vae = _load_ming_tts_audio_vae(
        checkpoint_dir,
        audio_config,
        device=resolved_device,
        dtype=resolved_dtype,
    )

    decoder = MingAudioDecoder(
        audio_vae,
        stream_capacity=stream_slots,
        max_stream_step_latents=max_step_latents,
        streaming_cuda_graph_required=streaming_cuda_graph,
    )

    try:
        scheduler = MingTTSStreamingVocoderScheduler(
            decoder,
            patch_size=patch_size,
            latent_dim=latent_dim,
            initial_chunk_patches=initial_chunk_patches,
            steady_chunk_patches=steady_chunk_patches,
            keep_latents=keep_latents,
        )
    except Exception:
        try:
            decoder.close()
        except Exception:
            logger.exception("Ming-Omni-TTS AudioVAE construction cleanup failed")
        raise
    try:
        scheduler.warmup_now()
        post_process_bytes = get_process_gpu_memory_bytes(logical_gpu_id)
        process_delta_bytes = (
            post_process_bytes - pre_process_bytes
            if pre_process_bytes is not None and post_process_bytes is not None
            else None
        )
        process_budget_bytes = None
        if process_fraction is None:
            memory_verification = "not_requested"
        elif post_process_bytes is None or device_info.total_memory_bytes is None:
            memory_verification = "unavailable"
            logger.warning(
                "ming_tts_audio_decode_memory stage=audio_decode "
                "memory_verification=unavailable process_post_bytes=%s "
                "device_total_bytes=%s process_fraction=%s",
                post_process_bytes,
                device_info.total_memory_bytes,
                process_fraction,
            )
        else:
            process_budget_bytes = int(
                device_info.total_memory_bytes * process_fraction
            )
            if post_process_bytes > process_budget_bytes:
                raise RuntimeError(
                    "Ming-Omni-TTS audio decode process GPU memory exceeds its "
                    "cumulative budget: "
                    f"used={format_bytes_gib(post_process_bytes)}, "
                    f"budget={format_bytes_gib(process_budget_bytes)}, "
                    f"process_fraction={process_fraction}"
                )
            memory_verification = "verified"

        logger.info(
            "ming_tts_audio_decode_ready stage=audio_decode device=%s dtype=%s "
            "attention_backend=%s streaming_backend=%s "
            "streaming_cuda_graph_required=%s stream_slots=%d "
            "initial_chunk_patches=%d steady_chunk_patches=%d "
            "audio_patch_size=%d max_step_latents=%d latent_dim=%d "
            "component_fraction=%s process_fraction=%s "
            "process_pre_bytes=%s process_post_bytes=%s "
            "audio_factory_process_delta_bytes=%s process_budget_bytes=%s "
            "memory_verification=%s",
            resolved_device,
            str(resolved_dtype).removeprefix("torch."),
            MING_TTS_AUDIO_VAE_ATTN_IMPLEMENTATION,
            "cuda_graph" if streaming_cuda_graph else "eager",
            streaming_cuda_graph,
            stream_slots,
            initial_chunk_patches,
            steady_chunk_patches,
            patch_size,
            max_step_latents,
            latent_dim,
            component_fraction,
            process_fraction,
            pre_process_bytes,
            post_process_bytes,
            process_delta_bytes,
            process_budget_bytes,
            memory_verification,
        )
    except Exception:
        try:
            scheduler.stop()
        except Exception:
            logger.exception("Ming-Omni-TTS AudioVAE construction cleanup failed")
        raise
    return scheduler


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
