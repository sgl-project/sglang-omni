# SPDX-License-Identifier: Apache-2.0
"""Tests for SGLang encoder runner runtime performance patches."""
from __future__ import annotations

import torch
import torch.nn as nn

from sglang_omni.model_runner.sglang_encoder_runner import (
    _optimize_conv3d_patch_embeds,
    _use_no_compile_vision_rotary,
)


class _PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            3,
            7,
            kernel_size=(2, 4, 4),
            stride=(2, 4, 4),
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, 3, 2, 4, 4)
        return self.proj(hidden_states).view(-1, 7)


class _VisionLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(hidden_states)


def test_optimize_conv3d_patch_embed_preserves_output_and_proj_api():
    model = _VisionLike().eval()
    hidden_states = torch.randn(11, 3 * 2 * 4 * 4)

    with torch.no_grad():
        expected = model(hidden_states)
        count = _optimize_conv3d_patch_embeds(model)
        actual = model(hidden_states)

    assert count == 1
    assert hasattr(model.patch_embed, "proj")
    assert hasattr(model.patch_embed, "linearized_proj")
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_no_compile_vision_rotary_patch_is_idempotent(monkeypatch):
    import sglang.srt.layers.attention.vision as vision_attention

    def _sentinel_rotary(q, k, cos, sin, unsqueeze_dim=1):
        del cos, sin, unsqueeze_dim
        return q, k

    monkeypatch.setattr(vision_attention, "apply_rotary_pos_emb", _sentinel_rotary)

    assert _use_no_compile_vision_rotary() is True
    patched = vision_attention.apply_rotary_pos_emb
    assert getattr(patched, "_sglang_omni_no_compile", False) is True
    assert _use_no_compile_vision_rotary() is False
    assert vision_attention.apply_rotary_pos_emb is patched

    q = torch.randn(5, 3, 4, dtype=torch.bfloat16)
    k = torch.randn(5, 3, 4, dtype=torch.bfloat16)
    cos = torch.randn(5, 4)
    sin = torch.randn(5, 4)
    q_out, k_out = patched(q, k, cos, sin)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert q_out.dtype == q.dtype
    assert k_out.dtype == k.dtype
