# SPDX-License-Identifier: Apache-2.0
"""Utilities shared across the Higgs TTS pipeline.

- Delay pattern: :func:`apply_delay_pattern` (forward) shifts codebook ``c``
  by ``c`` steps with BOC/EOC padding inside the codebook vocab (ids 1024 /
  1025 for the default 1026 vocab). The reverse (de-delay) transform is shared
  in :func:`sglang_omni.utils.codec_delay.reverse_delay_pattern`.
- :func:`truncate_rope_to_bf16` matches sglang's fp32 RoPE cache to Higgs's
  bf16 training-time RoPE.
- Stage helpers: checkpoint snapshot, codec cache, ref-codes coercion,
  ref-audio loading from path / URL / bytes / base64.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
from sglang_omni.preprocessing.audio import AudioMediaIO
from sglang_omni.preprocessing.base import _is_url
from sglang_omni.preprocessing.resource_connector import global_http_connection
from sglang_omni.utils.checkpoint import resolve_checkpoint

# Codec-vocab specials (inside the [N*V] codebook space, NOT the text vocab).
BOC_ID = 1024
EOC_ID = 1025

# Shared between audio_encoder + vocoder; one codec load saves ~1 GB VRAM.
_CODEC_CACHE: dict[tuple[str, str, str], HiggsAudioCodec] = {}


def apply_delay_pattern(codes_TN: torch.Tensor) -> torch.Tensor:
    """``[T, N]`` raw codes → ``[T + N - 1, N]`` delayed, BOC/EOC padded."""
    if codes_TN.ndim != 2:
        raise ValueError(
            f"codes_TN must be 2-D [T, N], got shape {tuple(codes_TN.shape)}"
        )
    T, N = codes_TN.shape
    out = torch.full(
        (T + N - 1, N), EOC_ID, device=codes_TN.device, dtype=codes_TN.dtype
    )
    t_idx = torch.arange(T + N - 1, device=codes_TN.device)
    for c in range(N):
        out[t_idx < c, c] = BOC_ID
        out[c : c + T, c] = codes_TN[:, c]
    return out


def delay_pattern_action_mask(
    delayed_LN: torch.Tensor,
    *,
    eoc_id: int = EOC_ID,
) -> torch.Tensor:
    """``[L, N]`` boolean mask: ``True`` at trainable real-audio actions."""
    if delayed_LN.ndim != 2:
        raise ValueError(
            f"delayed_LN must be 2-D [L, N], got shape {tuple(delayed_LN.shape)}"
        )
    L, N = delayed_LN.shape
    device = delayed_LN.device

    cb0_eoc_rows = (delayed_LN[:, 0] == eoc_id).nonzero(as_tuple=False)
    t_raw = int(cb0_eoc_rows[0].item()) if cb0_eoc_rows.numel() > 0 else L

    r = torch.arange(L, device=device).unsqueeze(1)  # [L, 1]
    c = torch.arange(N, device=device).unsqueeze(0)  # [1, N]
    return (c <= r) & (r < c + t_raw)


def truncate_rope_to_bf16(model: torch.nn.Module) -> None:
    """bf16-truncate sglang's fp32 ``cos_sin_cache`` in-place (stored as fp32)
    to match Higgs's bf16 training-time RoPE.
    """
    for module in model.modules():
        if hasattr(module, "cos_sin_cache"):
            cache = module.cos_sin_cache
            truncated = cache.to(torch.bfloat16).to(cache.dtype)
            cache.copy_(truncated)


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


def to_codes_TN(raw: Any, num_codebooks: int) -> torch.Tensor | None:
    """Coerce client-supplied ``reference_codes`` to a ``[T, N]`` int64 tensor."""
    if raw is None:
        return None
    t = raw if isinstance(raw, torch.Tensor) else torch.tensor(raw)
    if t.numel() == 0:
        return None
    if t.ndim != 2 or t.shape[1] != num_codebooks:
        raise ValueError(
            f"reference_codes must have shape [T, {num_codebooks}], got {tuple(t.shape)}"
        )
    return t.to(torch.long)


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

    if "bytes" in reference_audio:
        audio, sr = io.load_bytes(reference_audio["bytes"])
        return np.asarray(audio, dtype=np.float32), int(sr)
    data = reference_audio.get("base64") or reference_audio.get("data")
    if data is not None:
        media_type = reference_audio.get("media_type", "audio/wav")
        audio, sr = io.load_base64(media_type, data)
        return np.asarray(audio, dtype=np.float32), int(sr)
    if "audio_path" in reference_audio or "path" in reference_audio:
        return _load_path_or_url(
            reference_audio.get("audio_path") or reference_audio["path"]
        )
    raise ValueError("reference_audio must include audio_path, path, bytes, or data")


__all__ = [
    "BOC_ID",
    "EOC_ID",
    "apply_delay_pattern",
    "delay_pattern_action_mask",
    "get_or_load_codec",
    "load_audio_to_24k",
    "resolve_checkpoint",
    "to_codes_TN",
    "truncate_rope_to_bf16",
]
