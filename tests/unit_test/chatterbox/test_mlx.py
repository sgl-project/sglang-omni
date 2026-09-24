# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from sglang_omni.models.chatterbox.mlx.loader import _split_fused_qkv
from sglang_omni.models.chatterbox.mlx.model import NoPE


def test_nope_is_identity() -> None:
    rope = NoPE()
    x = mx.ones((2, 4, 8))
    assert mx.array_equal(rope(x), x).item()
    assert rope.dims == 0
    assert rope.traditional is True


def test_split_fused_qkv_weight() -> None:
    weight = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    parts = _split_fused_qkv("h.0.attn.c_attn.weight", weight)

    assert set(parts) == {
        "h.0.attn.q_proj.weight",
        "h.0.attn.k_proj.weight",
        "h.0.attn.v_proj.weight",
    }
    for projection in parts.values():
        assert projection.shape == (2, 2)


def test_split_fused_qkv_bias() -> None:
    bias = torch.arange(6, dtype=torch.float32)
    parts = _split_fused_qkv("h.0.attn.c_attn.bias", bias)

    assert set(parts) == {
        "h.0.attn.q_proj.bias",
        "h.0.attn.k_proj.bias",
        "h.0.attn.v_proj.bias",
    }
    for projection in parts.values():
        assert projection.shape == (2,)
