# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MingSemanticEncoder."""

import pytest


def test_semantic_encoder_init():
    """MingSemanticEncoder can be instantiated."""
    from sglang_omni.models.ming_omni.diffusion.semantic_encoder import (
        MingSemanticEncoder,
    )

    encoder = MingSemanticEncoder()
    assert encoder._llm is None
    assert encoder._tokenizer is None
    assert encoder._connector is None
    assert encoder._proj_in is None
    assert encoder._proj_out is None
    assert encoder._query_tokens is None
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
    assert encoder._llm is None


def test_zimage_backend_has_semantic_encoder():
    """ZImageBackend exposes semantic encoding mode."""
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    backend = ZImageBackend()
    assert hasattr(backend, "_semantic_encoder")
    assert backend._semantic_encoder is None


def test_helper_find_consecutive_ones():
    """_find_first_index_of_consecutive_ones works correctly."""
    from sglang_omni.models.ming_omni.diffusion.semantic_encoder import (
        _find_first_index_of_consecutive_ones,
    )

    assert _find_first_index_of_consecutive_ones([0, 1, 1, 0, 1, 0]) == [1, 4]
    assert _find_first_index_of_consecutive_ones([1, 1, 1]) == [0]
    assert _find_first_index_of_consecutive_ones([0, 0, 0]) == []
    assert _find_first_index_of_consecutive_ones([]) == []


def test_helper_merge_consecutive_ones():
    """_merge_consecutive_ones works correctly."""
    from sglang_omni.models.ming_omni.diffusion.semantic_encoder import (
        _merge_consecutive_ones,
    )

    assert _merge_consecutive_ones([0, 1, 0], 1) == [0, 1, 0]
    assert _merge_consecutive_ones([1, 1, 1, 1], 2) == [1, 1]
    assert _merge_consecutive_ones([0, 1, 1, 0, 1, 1, 1, 1], 2) == [0, 1, 0, 1, 1]
