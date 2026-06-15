# SPDX-License-Identifier: Apache-2.0
# Shared CSM / Kyutai-Mimi constants and HF-Transformers-export staging; the 24 kHz audio
# loader follows the sglang-omni Higgs loader. Mimi framing per transformers.MimiModel (Apache-2.0).
"""Constants + stage helpers shared across the CSM TTS pipeline.

Key facts:

- Mimi: 24 kHz, 12.5 Hz frame rate ⇒ one frame = 80 ms = 1920 samples,
  32 quantizers, codebook size 2048.
- Audio vocab per codebook is 2051: valid Mimi codes 0..2047; specials
  2048/2049/2050 must NEVER reach Mimi (``CODEBOOK_PAD=2050``).
- EOS frame: codebooks 0..30 all == 0 (cb31 excluded); audio trim at the
  first all-32-zero frame; default generation cap 125 frames.
- **No delay-pattern helpers** — CSM has none (unlike higgs).


"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download

from sglang_omni.models.csm_tts.audio_codec import (
    CsmMimiCodec,
    trim_at_first_all_zero_frame,
)
from sglang_omni.preprocessing.audio import AudioMediaIO
from sglang_omni.preprocessing.base import _is_url
from sglang_omni.preprocessing.resource_connector import global_http_connection

# --- CSM constants (real values, pinned by) -----------------------
NUM_CODEBOOKS = 32
CODEBOOK_VOCAB = 2051  # per-codebook logits width (2048 Mimi + 3 specials)
MIMI_CODEBOOK_SIZE = 2048  # valid Mimi code range is [0, 2047]
CODEBOOK_EOS = 0
CODEBOOK_PAD = 2050
STOP_CODE = -1  # frozen-row sentinel emitted after generation_done
AUDIO_TOKEN_ID = 128002  # <|AUDIO|> — real in-vocab id (no -100 sentinel)
AUDIO_EOS_TOKEN_ID = 128003  # <|audio_eos|>
BOS = 128000
EOT = 128001
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920  # 24 kHz / 12.5 Hz
FRAME_RATE = 12.5
DEFAULT_MAX_FRAMES = 125
BACKBONE_CTX = 2048
K_MAX = 2051  # fixed-shape top-k width for the branchless sampler

# Mimi encoder causal-conv ladder (HF processing_csm.py:57-60); the stride
# product is 4*5*6*8*2 = 1920 = SAMPLES_PER_FRAME.
_MIMI_KERNEL_SIZES = (7, 3, 1, 8, 3, 1, 10, 3, 1, 12, 3, 1, 16, 3, 4)
_MIMI_STRIDES = (1, 1, 1, 4, 1, 1, 5, 1, 1, 6, 1, 1, 8, 1, 2)
_MIMI_DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

# Shared between audio_encoder + vocoder; one Mimi load saves ~0.4 GB fp32.
_CODEC_CACHE: dict[tuple[str, str, str], CsmMimiCodec] = {}


def resolve_checkpoint(checkpoint: str) -> str:
    """Local dir or HF repo id → local snapshot path, curated to the
    HF-transformers export when the snapshot mixes checkpoint formats."""
    if Path(checkpoint).is_dir():
        snapshot = Path(checkpoint)
    else:
        snapshot = Path(snapshot_download(checkpoint))
    return str(_stage_transformers_only_view(snapshot))


_TF_INDEX_NAME = "transformers.safetensors.index.json"
_TF_SHARD_PREFIX = "transformers-"


def _stage_transformers_only_view(snapshot: Path) -> Path:
    """Expose ONLY the HF-transformers export to the sglang weight loader.

    The sesame/csm-1b snapshot can contain BOTH the orig-format
    ``model.safetensors`` (un-permuted-RoPE Q/K — must never be read) and the ``transformers-*.safetensors`` shards. sglang's
    ``DefaultModelLoader`` globs ``*.safetensors``, so a mixed snapshot feeds
    orig-format tensors into the strict remap census (and would corrupt Q/K
    if they were ever mapped). Stage a symlink view with the transformers
    shards renamed to canonical ``model-0000i-of-0000n.safetensors`` plus a
    rewritten ``model.safetensors.index.json`` so the loader's index filter
    selects exactly the 538-tensor ship artifact. ``audio_codec.py`` falls
    back to the rewritten index name, and tokenizer/config sidecars are
    symlinked through unchanged.
    """
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
            continue  # orig-format weights / stale indexes stay OUT of the view
        _link(view / name, entry)
    with open(view / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)
    return view


def get_or_load_codec(checkpoint_dir: str, device: str, dtype: str) -> CsmMimiCodec:
    """Process-wide cached :class:`CsmMimiCodec` per ``(path, device, dtype)``.

    One Mimi load serves both audio_encoder (encode) and vocoder (decode).

    Returns:
        ``CsmMimiCodec`` (fp32 by default — conv-transpose decode stability).
    """
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
    """Closed-form Mimi frame count for ``num_samples`` raw 24 kHz samples.

    Port of HF ``processing_csm.py:93-129`` (causal-conv ladder ceil;
    ≈ ``ceil(num_samples / 1920)``). Used ONLY for pre-encode budget /
    admission estimates on raw waveforms — the pre-encoded fast path needs no
    formula (F is just ``codes.shape[0]``), and the raw-waveform path defers
    the actual placeholder count to the real Mimi encode so
    prompts can never mismatch. Do NOT promote this estimate to prompt
    assembly.

    Returns:
        int — number of 80 ms frames F.
    """
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

        # use_causal_conv=True: full padding on the left, extra on the right.
        cur_length = cur_length + padding_total + extra_padding
        cur_length = (cur_length - dilation * (kernel_size - 1) - 1) // stride + 1
    return cur_length


def to_codes_F32(obj: Any) -> torch.Tensor | None:
    """Coerce client-supplied pre-encoded context codes to ``[F, 32]`` int64.

    Accepts ``torch.Tensor`` / nested lists / numpy; ``None`` / empty → None.
    Raises ``ValueError`` unless the result is 2-D with 32 columns.

    Returns:
        ``torch.LongTensor [F, 32]`` or ``None``.
    """
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
    """Delay-free frame assembly: list of per-step int64 ``[32]`` rows →
    ``[F, 32]`` code matrix (CSM has no delay pattern — rows stack as-is).

    Returns:
        ``torch.LongTensor [F, 32]`` (``[0, 32]`` for an empty list).
    """
    if not frames:
        return torch.empty(0, NUM_CODEBOOKS, dtype=torch.long)
    return torch.stack([f.reshape(NUM_CODEBOOKS).to(torch.long) for f in frames], dim=0)


# NOTE: trim_at_first_all_zero_frame is defined ONCE in audio_codec.py (the
# fence-#3 home) and re-exported here for stage-side callers.


def load_audio_to_24k(reference_audio: Any) -> tuple[np.ndarray, int]:
    """Load context audio as 24 kHz mono float32 (higgs loader copy).

    Accepts local path, HTTP/HTTPS URL, or ``{audio_path|path|bytes|base64|
    data}`` dict.

    Returns:
        ``(audio_float32 [S], sample_rate)`` — caller reshapes to ``[1, 1, S]``.
    """
    io = AudioMediaIO(target_sr=SAMPLE_RATE)

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
    "AUDIO_EOS_TOKEN_ID",
    "AUDIO_TOKEN_ID",
    "BACKBONE_CTX",
    "BOS",
    "CODEBOOK_EOS",
    "CODEBOOK_PAD",
    "CODEBOOK_VOCAB",
    "DEFAULT_MAX_FRAMES",
    "EOT",
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
