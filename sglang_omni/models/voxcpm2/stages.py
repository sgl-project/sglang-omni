# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 stage factories: preprocessing, reference encode, engine and vocoder."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer

from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.models.voxcpm2.components.audio_vae import AudioVAE, AudioVAEConfig
from sglang_omni.models.voxcpm2.hf_config import (
    VoxCPM2RuntimeConfig,
    load_voxcpm2_config,
)
from sglang_omni.models.voxcpm2.reference_encode import VoxCPM2ReferenceEncoder
from sglang_omni.models.voxcpm2.request_builders import (
    VoxCPM2PreprocessingContext,
    preprocess_voxcpm2_payload,
    set_voxcpm2_preprocessing_context,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)

_AUDIO_VAE_FILES = ("audiovae.safetensors", "audiovae.pth")


def _load_audio_vae(
    checkpoint: str, config: VoxCPM2RuntimeConfig, *, device: str
) -> AudioVAE:
    """Build the AudioVAE and load its standalone checkpoint file."""
    root = Path(checkpoint)
    for name in _AUDIO_VAE_FILES:
        path = root / name
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(path), device="cpu")
        else:
            loaded = torch.load(str(path), map_location="cpu", weights_only=True)
            state_dict = loaded.get("state_dict", loaded)
        vae = AudioVAE(AudioVAEConfig(**config.audio_vae))
        vae.load_state_dict(state_dict, assign=True)
        # note (Xinhao Tan): do not fold the VAE into the bfloat16 cast the rest
        # of the model takes. Upstream casts the whole model to the config dtype
        # and then casts the VAE back to float32; matching that is the only way
        # to reproduce its audio.
        return vae.eval().to(device=device, dtype=torch.float32)

    raise FileNotFoundError(
        f"VoxCPM2 AudioVAE checkpoint not found under {root}; "
        f"expected one of {list(_AUDIO_VAE_FILES)}"
    )


@lru_cache(maxsize=2)
def _resolved(model_path: str) -> tuple[str, VoxCPM2RuntimeConfig]:
    checkpoint = resolve_checkpoint(model_path)
    return checkpoint, load_voxcpm2_config(checkpoint)


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 8,
) -> SimpleScheduler:
    checkpoint, config = _resolved(model_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    set_voxcpm2_preprocessing_context(
        VoxCPM2PreprocessingContext(config=config, tokenizer=tokenizer)
    )
    return SimpleScheduler(preprocess_voxcpm2_payload, max_concurrency=max_concurrency)


def create_reference_encode_executor(
    model_path: str,
    *,
    device: str = "cuda",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    max_concurrency: int = 8,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 10,
) -> SimpleScheduler:
    del dtype, max_batch_size, max_batch_wait_ms  # the VAE encoder runs per request
    checkpoint, config = _resolved(model_path)
    worker_device = device if gpu_id is None else f"{device}:{gpu_id}"
    encoder = VoxCPM2ReferenceEncoder(
        _load_audio_vae(checkpoint, config, device=worker_device),
        patch_size=config.patch_size,
        cache_model_identity=str(model_path),
    )
    return SimpleScheduler(encoder.encode_payload, max_concurrency=max_concurrency)


def create_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    inference_timesteps: int = C.DEFAULT_INFERENCE_TIMESTEPS,
    cfg_value: float = C.DEFAULT_CFG_VALUE,
    min_len: int = C.DEFAULT_MIN_LEN,
    max_len: int = C.DEFAULT_MAX_LEN,
    max_running_requests: int = 1,
) -> object:
    del min_len, max_len  # per-request, resolved in the runner from the payload
    from sglang_omni.models.voxcpm2.engine_builder import VoxCPM2EngineBuilder

    builder = VoxCPM2EngineBuilder(
        inference_timesteps=inference_timesteps,
        cfg_value=cfg_value,
        max_running_requests=max_running_requests,
    )
    return builder.build(
        model_path,
        device=device,
        gpu_id=0 if gpu_id is None else gpu_id,
        dtype=dtype,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    gpu_id: int | None = None,
    max_batch_size: int = 1,
    stream_stride: int = 8,
    stream_followup_stride: int = 4,
    overlap_patches: int = C.DEFAULT_STREAMING_PREFIX_LEN - 1,
) -> object:
    from sglang_omni.models.voxcpm2.streaming_vocoder import VoxCPM2StreamingVocoder

    checkpoint, config = _resolved(model_path)
    worker_device = device if gpu_id is None else f"{device}:{gpu_id}"
    return VoxCPM2StreamingVocoder(
        _load_audio_vae(checkpoint, config, device=worker_device),
        device=worker_device,
        patch_size=config.patch_size,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        overlap_patches=overlap_patches,
        max_batch_size=max_batch_size,
    )


__all__ = [
    "create_preprocessing_executor",
    "create_reference_encode_executor",
    "create_tts_engine_executor",
    "create_vocoder_executor",
]
