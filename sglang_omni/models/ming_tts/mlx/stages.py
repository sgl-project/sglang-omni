# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


def create_mlx_audio_decode_executor(
    model_path: str,
    *,
    keep_latents: bool,
    initial_chunk_patches: int,
    steady_chunk_patches: int,
) -> Any:
    from sglang_omni.models.ming_tts.streaming_vocoder import (
        MingTTSStreamingVocoderScheduler,
    )
    from sglang_omni.utils.checkpoint import resolve_checkpoint

    from .audio_io import MingMlxAudioDecoder
    from .config import ModelConfig
    from .loading import load_ming_audio_vae, read_config

    path = resolve_checkpoint(model_path)
    config = ModelConfig.from_dict(read_config(path))
    decoder = MingMlxAudioDecoder(load_ming_audio_vae(path, component="decoder"))
    scheduler = MingTTSStreamingVocoderScheduler(
        decoder,
        patch_size=config.patch_size,
        latent_dim=config.latent_dim,
        initial_chunk_patches=initial_chunk_patches,
        steady_chunk_patches=steady_chunk_patches,
        keep_latents=keep_latents,
    )
    scheduler.warmup_now()
    return scheduler
