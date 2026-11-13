# SPDX-License-Identifier: Apache-2.0
"""Constants and stage helpers for the CSM TTS pipeline.

Mimi is 24 kHz / 12.5 Hz, 32 RVQ. EOS is codebooks 0-30 all 0.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sglang_omni.models.csm_tts.audio_codec import (
    CsmMimiCodec,
    trim_at_first_all_zero_frame,
)
from sglang_omni.preprocessing.audio import AudioMediaIO
from sglang_omni.preprocessing.base import is_url
from sglang_omni.preprocessing.resource_connector import global_http_connection
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_snapshot

NUM_CODEBOOKS = 32
CODEBOOK_VOCAB = 2051
MIMI_CODEBOOK_SIZE = 2048
CODEBOOK_EOS = 0
CODEBOOK_PAD = 2050
STOP_CODE = -1
AUDIO_TOKEN_ID = 128002
AUDIO_EOS_TOKEN_ID = 128003
BOS_TOKEN_ID = 128000
EOT_TOKEN_ID = 128001
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920
FRAME_RATE = 12.5
DEFAULT_MAX_FRAMES = 125
BACKBONE_CTX = 2048
K_MAX = 2051

_MIMI_KERNEL_SIZES = (7, 3, 1, 8, 3, 1, 10, 3, 1, 12, 3, 1, 16, 3, 4)
_MIMI_STRIDES = (1, 1, 1, 4, 1, 1, 5, 1, 1, 6, 1, 1, 8, 1, 2)
_MIMI_DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

_CODEC_CACHE: dict[tuple[str, str, str], CsmMimiCodec] = {}

_TF_INDEX_NAME = "transformers.safetensors.index.json"
_TF_SHARD_PREFIX = "transformers-"


def resolve_checkpoint(checkpoint: str) -> str:
    """Local dir or HF repo id, preferring the Transformers export."""
    snapshot = Path(_resolve_snapshot(checkpoint))
    return str(stage_transformers_only_view(snapshot))


def stage_transformers_only_view(snapshot: Path) -> Path:
    """Hide original sesame shards so the loader only sees Transformers weights."""
    tf_index = snapshot / _TF_INDEX_NAME
    if not tf_index.exists():
        return snapshot  # plain layout (e.g. a clean export dir): nothing to do
    import hashlib
    import json

    view = (
        Path.home()
        / ".cache"
        / "sglang_omni"
        / "csm_tfview"
        / hashlib.sha256(str(snapshot.resolve()).encode()).hexdigest()[:16]
    )
    view.mkdir(parents=True, exist_ok=True)

    with open(tf_index) as f:
        index = json.load(f)
    rename = {
        shard: (
            "model-" + shard[len(_TF_SHARD_PREFIX) :]
            if shard.startswith(_TF_SHARD_PREFIX)
            else shard
        )
        for shard in sorted(set(index["weight_map"].values()))
    }
    index["weight_map"] = {k: rename[v] for k, v in index["weight_map"].items()}

    def _link(dst: Path, src: Path) -> None:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())

    for src_name, dst_name in rename.items():
        _link(view / dst_name, snapshot / src_name)
    for entry in snapshot.iterdir():
        name = entry.name
        if name == _TF_INDEX_NAME or name in rename:
            continue
        if name.endswith((".safetensors", ".bin", ".pt", ".pth")) or name in (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ):
            continue
        _link(view / name, entry)
    with open(view / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)
    return view


def get_or_load_codec(checkpoint_dir: str, device: str, dtype: str) -> CsmMimiCodec:
    """Process-wide cached CsmMimiCodec per (path, device, dtype)."""
    key = (str(checkpoint_dir), str(device), str(dtype))
    cached = _CODEC_CACHE.get(key)
    if cached is not None:
        return cached
    codec = CsmMimiCodec.from_pretrained(
        checkpoint_dir, device=device, dtype=getattr(torch, dtype)
    )
    _CODEC_CACHE[key] = codec
    return codec


def encoded_audio_length(num_samples: int) -> int:
    """Closed-form Mimi frame count for raw 24 kHz samples."""
    cur_length = num_samples
    for kernel_size, stride, dilation in zip(
        _MIMI_KERNEL_SIZES, _MIMI_STRIDES, _MIMI_DILATIONS
    ):
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = kernel_size - stride

        n_frames = (cur_length - effective_kernel_size + padding_total) / stride + 1
        n_frames = math.ceil(n_frames) - 1
        ideal_length = n_frames * stride + kernel_size - padding_total
        extra_padding = ideal_length - cur_length

        cur_length = cur_length + padding_total + extra_padding
        cur_length = (cur_length - dilation * (kernel_size - 1) - 1) // stride + 1
    return cur_length


def to_codes_F32(obj: Any) -> torch.Tensor | None:
    """Coerce pre-encoded context codes to [F, 32] int64."""
    if obj is None:
        return None
    t = obj if isinstance(obj, torch.Tensor) else torch.tensor(obj)
    if t.numel() == 0:
        return None
    if t.ndim != 2 or t.shape[1] != NUM_CODEBOOKS:
        raise ValueError(
            f"context codes must have shape [F, {NUM_CODEBOOKS}], got {tuple(t.shape)}"
        )
    return t.to(torch.long)


def stack_frames(frames: list[torch.Tensor]) -> torch.Tensor:
    """Stack per-step [32] rows into [F, 32]."""
    if not frames:
        return torch.empty(0, NUM_CODEBOOKS, dtype=torch.long)
    return torch.stack([f.reshape(NUM_CODEBOOKS).to(torch.long) for f in frames], dim=0)


def load_audio_to_24k(reference_audio: Any) -> tuple[np.ndarray, int]:
    """Load context audio as 24 kHz mono float32."""
    io = AudioMediaIO(target_sr=SAMPLE_RATE)

    def _load_path_or_url(src: str | Path) -> tuple[np.ndarray, int]:
        if isinstance(src, str) and is_url(src):
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
    "AUDIO_EOS_TOKEN_ID",
    "AUDIO_TOKEN_ID",
    "BACKBONE_CTX",
    "BOS_TOKEN_ID",
    "CODEBOOK_EOS",
    "CODEBOOK_PAD",
    "CODEBOOK_VOCAB",
    "DEFAULT_MAX_FRAMES",
    "EOT_TOKEN_ID",
    "FRAME_RATE",
    "K_MAX",
    "MIMI_CODEBOOK_SIZE",
    "NUM_CODEBOOKS",
    "SAMPLES_PER_FRAME",
    "SAMPLE_RATE",
    "STOP_CODE",
    "encoded_audio_length",
    "get_or_load_codec",
    "load_audio_to_24k",
    "resolve_checkpoint",
    "stack_frames",
    "to_codes_F32",
    "trim_at_first_all_zero_frame",
]
