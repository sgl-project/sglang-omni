# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the dots.tts pipeline.

``tts_engine`` runs on the OmniScheduler with SGLang owning the Qwen2 AR
backbone's KV cache and batching; the other three stages are pure pre/post
processing and stay on ``SimpleScheduler``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from sglang_omni.models.dots_tts.hf_config import (
    DOTS_TTS_SPEAKER_ENCODER_FILENAME,
    DOTS_TTS_VOCODER_FILENAME,
    load_dots_tts_model_config,
)
from sglang_omni.models.dots_tts.payload_types import (
    DOTS_TTS_DEFAULT_MAX_AUDIO_PATCHES,
    DOTS_TTS_DEFAULT_NUM_STEPS,
)
from sglang_omni.models.dots_tts.prompt_builder import DOTS_TTS_MAX_SEQUENCE_LENGTH
from sglang_omni.models.dots_tts.request_builders import preprocess_dots_tts_payload
from sglang_omni.models.dots_tts.weight_loading import load_dots_tts_module_weights
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)

DEFAULT_PRECISION = "bfloat16"
DEFAULT_EOS_THRESHOLD = 0.8


def create_preprocessing_executor(max_concurrency: int = 1) -> SimpleScheduler:
    return SimpleScheduler(preprocess_dots_tts_payload, max_concurrency=max_concurrency)


def create_reference_encode_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = DEFAULT_PRECISION,
    max_sequence_length: int = DOTS_TTS_MAX_SEQUENCE_LENGTH,
    max_concurrency: int = 1,
    ref_audio_cache_max_items: int = 64,
    ref_audio_cache_max_bytes: int = 512 * 1024 * 1024,
) -> SimpleScheduler:
    # note (chenyang): the AudioVAE encoder and CAM++ both run in float32
    # upstream, so dtype only selects the engine precision and is ignored here.
    del dtype

    from transformers import AutoTokenizer

    from sglang_omni.models.dots_tts.components.vocoder.vocoder_inference import (
        VocoderInference,
    )
    from sglang_omni.models.dots_tts.reference_encode import DotsTtsReferenceEncoder

    checkpoint_dir = resolve_checkpoint(model_path)
    model_config = load_dots_tts_model_config(checkpoint_dir)
    resolved_device = torch.device(_resolve_device(device, gpu_id))
    audio_vae = _load_audio_vae(checkpoint_dir, model_config, resolved_device)
    speaker_encoder = _load_speaker_encoder(
        checkpoint_dir, model_config, resolved_device
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)

    encoder = DotsTtsReferenceEncoder(
        tokenizer=tokenizer,
        vocoder_inference=VocoderInference(audio_vae),
        speaker_encoder=speaker_encoder,
        sample_rate=int(model_config.vocoder.sample_rate),
        patch_size=int(model_config.patch_size),
        samples_per_patch=_samples_per_patch(model_config),
        model_identity=checkpoint_dir,
        device=resolved_device,
        max_sequence_length=max_sequence_length,
        cache_max_items=ref_audio_cache_max_items,
        cache_max_bytes=ref_audio_cache_max_bytes,
    )
    return SimpleScheduler(encoder.encode_payload, max_concurrency=max_concurrency)


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = DEFAULT_PRECISION,
    context_length: int = DOTS_TTS_MAX_SEQUENCE_LENGTH,
    num_steps: int = DOTS_TTS_DEFAULT_NUM_STEPS,
    eos_threshold: float = DEFAULT_EOS_THRESHOLD,
    max_audio_patches: int = DOTS_TTS_DEFAULT_MAX_AUDIO_PATCHES,
    max_running_requests: int = 16,
    server_args_overrides: dict[str, Any] | None = None,
    total_gpu_memory_fraction: float | None = None,
) -> Any:
    from sglang_omni.models.dots_tts.engine_builder import DotsTtsEngineBuilder

    user_overrides = dict(server_args_overrides or {})
    if int(user_overrides.pop("tp_size", 1)) != 1:
        raise ValueError("dots.tts tts_engine is single-GPU only")
    context_length = int(user_overrides.pop("context_length", context_length))

    return DotsTtsEngineBuilder(
        context_length=context_length,
        num_steps=num_steps,
        eos_threshold=eos_threshold,
        max_audio_patches=max_audio_patches,
        max_running_requests=max_running_requests,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=user_overrides,
    )


def create_tts_engine_executor(*args: Any, **kwargs: Any) -> Any:
    return create_sglang_tts_engine_executor(*args, **kwargs)


def create_audio_decode_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = DEFAULT_PRECISION,
    keep_latents: bool = False,
    max_batch_size: int = 1,
    max_batch_wait_ms: int = 0,
) -> SimpleScheduler:
    # note (chenyang): the AudioVAE decoder stays float32 upstream.
    del dtype

    from sglang_omni.models.dots_tts.audio_decode import DotsTtsBatchVocoder
    from sglang_omni.models.dots_tts.components.vocoder.vocoder_inference import (
        VocoderInference,
    )

    checkpoint_dir = resolve_checkpoint(model_path)
    model_config = load_dots_tts_model_config(checkpoint_dir)
    resolved_device = torch.device(_resolve_device(device, gpu_id))
    audio_vae = _load_audio_vae(checkpoint_dir, model_config, resolved_device)

    vocoder = DotsTtsBatchVocoder(
        VocoderInference(audio_vae),
        sample_rate=int(model_config.vocoder.sample_rate),
        device=resolved_device,
        keep_latents=keep_latents,
    )
    return vocoder.build_scheduler(
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


def _load_audio_vae(
    checkpoint_dir: str, model_config: Any, device: torch.device
) -> torch.nn.Module:
    from sglang_omni.models.dots_tts.components.vocoder.bigvgan import AudioVAE

    audio_vae = AudioVAE(model_config.vocoder).eval()
    # note (chenyang): the checkpoint was saved after weight-norm removal, so
    # mirror that here before loading or the parameter names will not line up.
    audio_vae.remove_weight_norm()
    count = load_dots_tts_module_weights(
        audio_vae, f"{checkpoint_dir}/{DOTS_TTS_VOCODER_FILENAME}"
    )
    logger.info("dots.tts AudioVAE loaded: tensors=%s device=%s", count, device)
    return audio_vae.to(device=device).eval()


def _load_speaker_encoder(
    checkpoint_dir: str, model_config: Any, device: torch.device
) -> torch.nn.Module:
    from sglang_omni.models.dots_tts.components.speaker.encoder import (
        SpeakerXVectorFeatures,
    )

    speaker_encoder = SpeakerXVectorFeatures(
        sample_rate=int(model_config.vocoder.sample_rate),
        campplus_embedding_size=int(model_config.campplus_embedding_size),
        max_audio_seconds=float(model_config.xvec_max_audio_seconds),
    ).eval()
    count = load_dots_tts_module_weights(
        speaker_encoder, f"{checkpoint_dir}/{DOTS_TTS_SPEAKER_ENCODER_FILENAME}"
    )
    logger.info("dots.tts speaker encoder loaded: tensors=%s device=%s", count, device)
    return speaker_encoder.to(device=device).eval()


def _samples_per_patch(model_config: Any) -> int:
    hop_size = int(np.prod(model_config.vocoder.downsample_rates))
    return int(model_config.patch_size) * hop_size


def _resolve_device(device: str, gpu_id: int | None) -> str:
    if gpu_id is not None:
        return f"cuda:{int(gpu_id)}"
    return device


__all__ = [
    "create_audio_decode_executor",
    "create_preprocessing_executor",
    "create_reference_encode_executor",
    "create_sglang_tts_engine_executor",
    "create_tts_engine_executor",
]
