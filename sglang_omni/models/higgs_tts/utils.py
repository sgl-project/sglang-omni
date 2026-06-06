# SPDX-License-Identifier: Apache-2.0
"""Utilities shared across the Higgs TTS pipeline.

- :func:`truncate_rope_to_bf16` matches sglang's fp32 RoPE cache to Higgs's
  bf16 training-time RoPE.
- Stage helpers: checkpoint snapshot, codec cache, ref-audio loading from path /
  URL / bytes / base64.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download

from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
from sglang_omni.models.higgs_tts.codebook_layout import (
    BOC_ID,
    EOC_ID,
    apply_delay_pattern,
    reverse_delay_pattern,
    to_codes_TN,
)
from sglang_omni.preprocessing.audio import AudioMediaIO
from sglang_omni.preprocessing.base import _is_url
from sglang_omni.preprocessing.resource_connector import global_http_connection

# Shared between audio_encoder + vocoder; one codec load saves ~1 GB VRAM.
_CODEC_CACHE: dict[tuple[str, str, str], HiggsAudioCodec] = {}


def truncate_rope_to_bf16(model: torch.nn.Module) -> None:
    """bf16-truncate sglang's fp32 ``cos_sin_cache`` in-place (stored as fp32)
    to match Higgs's bf16 training-time RoPE.
    """
    for module in model.modules():
        if hasattr(module, "cos_sin_cache"):
            cache = module.cos_sin_cache
            truncated = cache.to(torch.bfloat16).to(cache.dtype)
            cache.copy_(truncated)


def resolve_checkpoint(checkpoint: str) -> str:
    """Local dir or HF repo id → local snapshot path."""
    if Path(checkpoint).is_dir():
        return checkpoint
    return snapshot_download(checkpoint)


def get_or_load_codec(path: str, device: str, dtype: str) -> HiggsAudioCodec:
    """Process-wide cached :class:`HiggsAudioCodec` per (path, device, dtype)."""
    key = (str(path), str(device), str(dtype))
    cached = _CODEC_CACHE.get(key)
    if cached is not None:
        return cached
    codec = HiggsAudioCodec.from_pretrained(
        path, device=device, dtype=getattr(torch, dtype)
    )
    _CODEC_CACHE[key] = codec
    return codec


def load_audio_to_24k(reference_audio: Any) -> tuple[np.ndarray, int]:
    """Load ``inputs["reference_audio"]`` as 24 kHz mono float32.

    Accepts local path, HTTP/HTTPS URL, or ``{audio_path|path|bytes|base64|data}`` dict.
    """
    io = AudioMediaIO(target_sr=HiggsAudioCodec.SAMPLE_RATE)

    def _load_path_or_url(src: str | Path) -> tuple[np.ndarray, int]:
        if isinstance(src, str) and _is_url(src):
            response = global_http_connection.get_sync_client().get(src)
            response.raise_for_status()
            audio, sr = io.load_bytes(response.content)
        else:
            audio, sr = io.load_file(Path(src))
        return np.asarray(audio, dtype=np.float32), int(sr)

    if isinstance(reference_audio, (str, Path)):
        return _load_path_or_url(reference_audio)

    if "audio_path" in reference_audio or "path" in reference_audio:
        return _load_path_or_url(
            reference_audio.get("audio_path") or reference_audio["path"]
        )
    if "bytes" in reference_audio:
        audio, sr = io.load_bytes(reference_audio["bytes"])
        return np.asarray(audio, dtype=np.float32), int(sr)
    media_type = reference_audio.get("media_type", "audio/wav")
    data = reference_audio.get("base64") or reference_audio["data"]
    audio, sr = io.load_base64(media_type, data)
    return np.asarray(audio, dtype=np.float32), int(sr)


__all__ = [
    "BOC_ID",
    "EOC_ID",
    "apply_delay_pattern",
    "get_or_load_codec",
    "load_audio_to_24k",
    "resolve_checkpoint",
    "reverse_delay_pattern",
    "to_codes_TN",
    "truncate_rope_to_bf16",
]
