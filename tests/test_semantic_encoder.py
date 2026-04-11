# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MingSemanticEncoder."""

import pytest


def test_semantic_encoder_init():
    """MingSemanticEncoder can be instantiated."""
    from sglang_omni.models.ming_omni.diffusion.semantic_encoder import (
        MingSemanticEncoder,
    )

    encoder = MingSemanticEncoder()
    assert encoder._model is None
    assert encoder._tokenizer is None
    assert encoder._device is None


def test_semantic_encoder_encode_not_loaded():
    """encode() raises RuntimeError if model not loaded."""
    from sglang_omni.models.ming_omni.diffusion.semantic_encoder import (
        MingSemanticEncoder,
    )

    encoder = MingSemanticEncoder()
    with pytest.raises(RuntimeError, match="not loaded"):
        encoder.encode("test prompt")


def test_semantic_encoder_unload_safe():
    """unload() is safe to call when nothing is loaded."""
    from sglang_omni.models.ming_omni.diffusion.semantic_encoder import (
        MingSemanticEncoder,
    )

    encoder = MingSemanticEncoder()
    encoder.unload()  # Should not raise
    assert encoder._model is None


def test_zimage_backend_has_semantic_encoder():
    """ZImageBackend exposes semantic encoding mode."""
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    backend = ZImageBackend()
    assert hasattr(backend, "_semantic_encoder")
    assert backend._semantic_encoder is None
