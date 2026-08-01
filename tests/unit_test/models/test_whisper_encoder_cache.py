# SPDX-License-Identifier: Apache-2.0
"""Tests for Whisper encoder cross-request cache keying (no torch required).

These exercise the pure keying logic in ``whisper_encoder_cache`` so the
content-addressing behavior of the opt-in encoder cache is covered without
importing torch / sglang. The actual encoder forward + StageOutputCache wiring
is validated on GPU in integration runs.
"""

from __future__ import annotations

from sglang_omni.models.whisper_asr.whisper_encoder_cache import (
    encoder_cache_key,
    feature_digest,
)


class _FakeTensor:
    """Minimal torch.tensor stand-in: exposes shape + deterministic bytes."""

    def __init__(self, data: bytes, shape: tuple[int, ...]) -> None:
        self._data = data
        self.shape = shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        # emulate numpy .tobytes via a tiny shim
        class _Arr:
            def __init__(self, b: bytes) -> None:
                self._b = b

            def tobytes(self) -> bytes:
                return self._b

        return _Arr(self._data)


def _feat(seed: int, n: int = 8) -> _FakeTensor:
    return _FakeTensor(bytes([(seed + i) % 256 for i in range(n)]), (1, n))


def test_same_feature_same_digest() -> None:
    a = _feat(1)
    b = _feat(1)
    assert feature_digest(a) == feature_digest(b)


def test_different_feature_different_digest() -> None:
    assert feature_digest(_feat(1)) != feature_digest(_feat(2))


def test_shape_changes_digest() -> None:
    a = _FakeTensor(b"\x01\x02\x03\x04", (1, 4))
    b = _FakeTensor(b"\x01\x02\x03\x04", (1, 2))
    assert feature_digest(a) != feature_digest(b)


def test_encoder_cache_key_order_independent() -> None:
    k1 = encoder_cache_key("whisper", [_feat(1), _feat(2)])
    k2 = encoder_cache_key("whisper", [_feat(2), _feat(1)])
    assert k1 == k2


def test_encoder_cache_key_model_aware() -> None:
    k1 = encoder_cache_key("whisper-base", [_feat(1)])
    k2 = encoder_cache_key("whisper-large", [_feat(1)])
    assert k1 != k2


def test_encoder_cache_key_same_audio_reuses() -> None:
    # Two separate requests with the *same* audio should share a key.
    k_a = encoder_cache_key("m", [_feat(7)])
    k_b = encoder_cache_key("m", [_feat(7)])
    assert k_a == k_b
