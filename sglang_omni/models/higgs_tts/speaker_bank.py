"""Speaker-level fingerprint via WavLM embedding + cosine-threshold bank.

Replaces the bit-exact hash of ``reference_codes_delayed`` with a
**speaker-aware** fingerprint so two different recordings of the same
speaker share the same radix cache prefix. Improves cache hit rate in
RL-rollout / multi-recording-per-speaker scenarios from ~50% to ~95%.

Three-tier cache (cheapest tier first):

1. **Bytes-hash cache** — same waveform bytes → reuse previously computed
   embedding, no encoder forward.
2. **Speaker bank** — embedding-vs-bank cosine match (threshold 0.5 sweet
   spot per PoC); if any existing speaker matches, reuse its fingerprint.
3. **Register new** — fresh speaker, append to bank with a new id.

Threshold 0.5 derived from PoC on seed-tts en (100% recall, 0% false
collision on N=50 pairs). Optimal value is encoder-dependent.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


_DEFAULT_THRESHOLD = 0.5

# Module-level singleton guarded by lock.
_LOCK = threading.Lock()
_ENCODER: Any | None = None
_BANK: list[tuple[str, np.ndarray]] = []  # [(spk_id, emb)]
_BYTES_CACHE: dict[str, np.ndarray] = {}  # bytes_hash → emb
_THRESHOLD = _DEFAULT_THRESHOLD


def configure(
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    encoder_ckpt: str = "/ceph/data/higgs_audio_eval/assets/wavlm_large_finetune.pth",
    s3prl_path: str = "/ceph/data/higgs_audio_eval/assets/s3prl",
    device: str = "cuda:0",
) -> None:
    """Load the WavLM speaker encoder (idempotent) and set the bank threshold."""
    global _ENCODER, _THRESHOLD
    with _LOCK:
        _THRESHOLD = float(threshold)
        if _ENCODER is not None:
            return

        # Lazy import — higgs_mm is heavy; only pulled when speaker bank is enabled.
        import sys

        sys.path.insert(0, "/ceph/workspace/huapeng/higgs-mm")
        from higgs_mm.eval.tts_metrics import WavLMWrapper

        logger.info("Loading WavLM speaker encoder (%s) on %s", encoder_ckpt, device)
        _ENCODER = WavLMWrapper(
            ckpt_path=encoder_ckpt, s3prl_path=s3prl_path, device=device
        )


def _embed(waveform: torch.Tensor) -> np.ndarray:
    """Encode a 24 kHz mono waveform → 256-d numpy embedding."""
    # WavLM expects 16 kHz; downsample
    import torchaudio.functional as F_audio

    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)  # [1, L]
    if waveform.shape[0] != 1:
        waveform = waveform.mean(0, keepdim=True)
    wav_16k = F_audio.resample(waveform, 24000, 16000).to(_ENCODER.device)
    with torch.no_grad():
        emb = _ENCODER.get_embedding(wav_16k).squeeze().cpu().numpy()
    return emb


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def fingerprint(waveform: torch.Tensor) -> str:
    """Return a speaker fingerprint string suitable for ``Req.extra_key``.

    Same speaker → same fingerprint (different recordings still match).
    Different speaker → different fingerprint.
    """
    if _ENCODER is None:
        raise RuntimeError("speaker_bank.configure() must be called first")

    # Tier 1: bytes hash cache
    wav_bytes = waveform.detach().cpu().numpy().tobytes()
    bytes_hash = hashlib.blake2b(wav_bytes, digest_size=8).hexdigest()
    cached_emb = _BYTES_CACHE.get(bytes_hash)
    if cached_emb is not None:
        emb = cached_emb
    else:
        # Tier 2: encode + cache the embedding
        emb = _embed(waveform)
        with _LOCK:
            _BYTES_CACHE[bytes_hash] = emb
            # Bound cache size
            if len(_BYTES_CACHE) > 4096:
                # Drop oldest insertion order entry
                _BYTES_CACHE.pop(next(iter(_BYTES_CACHE)))

    # Tier 3: bank lookup (cosine match)
    with _LOCK:
        for spk_id, spk_emb in _BANK:
            if _cos_sim(emb, spk_emb) > _THRESHOLD:
                return spk_id
        # Register new speaker
        new_id = f"spk_{len(_BANK):06d}"
        _BANK.append((new_id, emb))
        logger.debug(
            "speaker_bank: new speaker %s registered (bank size now %d)",
            new_id,
            len(_BANK),
        )
        return new_id


def stats() -> dict[str, int]:
    """Diagnostic: current bank + cache sizes."""
    with _LOCK:
        return {
            "bank_size": len(_BANK),
            "bytes_cache_size": len(_BYTES_CACHE),
            "threshold": _THRESHOLD,
        }


def reset() -> None:
    """Wipe bank + cache. Mainly for tests."""
    global _BANK, _BYTES_CACHE
    with _LOCK:
        _BANK = []
        _BYTES_CACHE = {}


__all__ = ["configure", "fingerprint", "stats", "reset"]
