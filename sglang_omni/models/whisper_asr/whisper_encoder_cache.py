# SPDX-License-Identifier: Apache-2.0
"""Cross-request Whisper encoder-output cache keying (torch-free core).

The Whisper encoder is a deterministic, stateless forward pass: the same
audio feature map always yields the same ``cross_attention_states``. When the
same audio (e.g. a reference clip) is requested repeatedly, re-running the
encoder is pure wasted compute. This module provides the *pure* keying logic
so it can be unit-tested without ``torch`` / ``sglang``; the actual caching
wraps :class:`sglang_omni.scheduling.reference_encoder.ReferenceEncodeService`
which lives in the model file (it imports torch).
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

__all__ = ["encoder_cache_key", "feature_digest"]


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
