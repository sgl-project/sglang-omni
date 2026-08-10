# SPDX-License-Identifier: Apache-2.0
"""Cross-request Whisper encoder-output cache (keying + cache wrapper).

The Whisper encoder is a deterministic, stateless forward pass: the same
audio feature map always yields the same ``encoder_states``. When the same
audio (e.g. a reference clip) is requested repeatedly, re-running the encoder
is pure wasted compute. This module provides:

* :func:`feature_digest` / :func:`encoder_cache_key` — torch-free, pure keying
  logic (unit-testable without torch/sglang).
* :class:`EncoderOutputCache` — the real cache wrapper. It is backend-agnostic
  (any dict-like object with ``get``/``put`` works), defaulting to the repo's
  :class:`StageOutputCache` so production uses LRU + byte-cap + CPU storage.
  Keeping it backend-injectable lets the integration test exercise the exact
  hit/miss logic on a real GPU with a stub encoder and no sglang dependency.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "feature_digest",
    "encoder_cache_key",
    "EncoderOutputCache",
]


def feature_digest(feature: Any) -> str:
    """Compute a stable digest for one audio feature tensor/mapping.

    ``feature`` is expected to be a tensor-like object exposing ``.shape`` and
    a deterministic byte representation (``.detach().cpu().numpy().tobytes()``
    for torch tensors, or a numpy array). The digest combines shape + content
    so two audibly-different clips can never collide, and the same clip always
    hashes identically.
    """
    shape = tuple(getattr(feature, "shape", None) or ())
    data = _to_bytes(feature)
    blob = hashlib.blake2b(data, digest_size=16)
    blob.update(b"|")
    blob.update(str(shape).encode("utf-8"))
    return blob.hexdigest()


def _to_bytes(feature: Any) -> bytes:
    # torch tensor path
    to_cpu = getattr(feature, "detach", None)
    if callable(to_cpu):
        try:
            import numpy as np

            arr = to_cpu().cpu().numpy()
            return arr.tobytes()
        except Exception:
            pass
    # numpy path
    tobytes = getattr(feature, "tobytes", None)
    if callable(tobytes):
        try:
            return tobytes()
        except Exception:
            pass
    # last resort: repr (never ideal, but keeps the function side-effect free)
    return repr(feature).encode("utf-8")


def encoder_cache_key(
    model_id: str,
    features: Sequence[Any],
) -> str:
    """Build a cache key for a batch of encoder features.

    The key is the model id plus the sorted per-feature digests, so the cache
    entry is content-addressed and independent of scheduling order. Identical
    audio batches (even across different requests) share one encoder output.
    """
    digests = sorted(feature_digest(f) for f in features)
    blob = hashlib.blake2b("".join(digests).encode("utf-8"), digest_size=16)
    blob.update(b"|")
    blob.update(model_id.encode("utf-8"))
    return blob.hexdigest()


class EncoderOutputCache:
    """Content-addressed cache for Whisper encoder outputs.

    Parameters
    ----------
    model_id:
        Identifier mixed into the cache key (different models must not share
        encoder outputs).
    backend:
        A dict-like store with ``get(key) -> value | None`` and
        ``put(key, value)``. Defaults to :class:`StageOutputCache` (LRU + byte
        cap, CPU-resident). The backend is injectable so tests can use a plain
        ``dict`` and still exercise the real hit/miss logic.
    """

    def __init__(
        self,
        model_id: str,
        backend: Optional[Any] = None,
        *,
        max_size: int = 256,
        max_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.model_id = model_id
        if backend is not None:
            # Accept a plain dict for lightweight injection/testing; wrap it so
            # the get/put contract matches StageOutputCache.
            if isinstance(backend, dict):
                store: dict = backend

                class _DictBackend:
                    def get(self, k):  # noqa: ANN001
                        return store.get(k)

                    def put(self, k, v):  # noqa: ANN001
                        store[k] = v

                self._backend = _DictBackend()
            else:
                self._backend = backend
        else:
            from sglang_omni.scheduling.stage_cache import StageOutputCache

            self._backend = StageOutputCache(
                max_size=max_size,
                max_bytes=max_bytes,
                cache_device="cpu",
            )

    def _key(self, features: Sequence[Any]) -> str:
        return encoder_cache_key(self.model_id, features)

    def get_or_encode(
        self,
        features: Sequence[Any],
        device: Any,
        encode_fn: Callable[[], Any],
    ) -> Any:
        """Return cached encoder states for ``features`` or compute + cache them.

        ``encode_fn`` is only invoked on a cache miss. On a hit the stored
        tensor is moved to ``device`` and returned, and ``encode_fn`` is *not*
        called — this is exactly the "encoder forward runs once" guarantee.
        """
        key = self._key(features)
        cached = self._backend.get(key)
        if cached is not None:
            logger.info("whisper encoder cache HIT (key=%s)", key[:12])
            import torch

            return cached.to(device)
        states = encode_fn()
        self._backend.put(key, states)
        logger.info("whisper encoder cache MISS (key=%s)", key[:12])
        return states
