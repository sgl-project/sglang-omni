# SPDX-License-Identifier: Apache-2.0
"""Tests for sharing audio-attention segment splits across encoder layers."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.audio_encoder import (
    _SegmentSplits,
    _share_segment_splits,
)

HEADS, HEAD_DIM = 4, 16
DIM = HEADS * HEAD_DIM


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, nn.Linear(DIM, DIM, bias=False))
        self.num_heads = HEADS
        self.scaling = HEAD_DIM**-0.5
        self.attention_dropout = 0.0
        self.config = hf_modeling.Qwen3OmniMoeAudioEncoderConfig(
            d_model=DIM, encoder_attention_heads=HEADS
        )
        self.config._attn_implementation = "sdpa"

    def forward(self, hidden_states, cu_seqlens, **kwargs):
        """Stands in for the stock per-layer device-to-host split."""
        seq_length, _ = hidden_states.size()
        q, k, v = (
            proj(hidden_states)
            .reshape(seq_length, self.num_heads, -1)
            .transpose(0, 1)
            .unsqueeze(0)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        fn = hf_modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, hf_modeling.eager_attention_forward
        )
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        splits = [torch.split(t, lengths, dim=2) for t in (q, k, v)]
        outs = [
            fn(
                self,
                a,
                b,
                c,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0,
                is_causal=False,
            )[0]
            for a, b, c in zip(*splits)
        ]
        out = torch.cat(outs, dim=1).reshape(seq_length, -1).contiguous()
        return self.out_proj(out)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()


class _Tower(nn.Module):
    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Layer() for _ in range(n))


def _inputs(segments: list[int]):
    total = sum(segments)
    hs = torch.randn(total, DIM)
    cu = torch.tensor([0, *segments], dtype=torch.int32).cumsum(0).to(torch.int32)
    return hs, cu


def test_share_segment_splits_patches_every_layer() -> None:
    tower, splits = _Tower(), _SegmentSplits()
    _share_segment_splits(tower, splits)
    for layer in tower.layers:
        assert layer.self_attn._omni_segment_splits is splits
        assert hasattr(layer.self_attn, "_omni_unshared_forward")


def test_shared_splits_match_the_per_layer_split_bitwise() -> None:
    torch.manual_seed(0)
    tower, splits = _Tower(), _SegmentSplits()
    segments = [104, 104, 52]
    hs, cu = _inputs(segments)
    with torch.no_grad():
        reference = [layer.self_attn(hs, cu) for layer in tower.layers]
    _share_segment_splits(tower, splits)
    splits.value = segments
    with torch.no_grad():
        shared = [layer.self_attn(hs, cu) for layer in tower.layers]
    for got, want in zip(shared, reference):
        assert torch.equal(got, want)


def test_missing_splits_fall_back_to_the_stock_path() -> None:
    torch.manual_seed(0)
    tower, splits = _Tower(), _SegmentSplits()
    hs, cu = _inputs([104, 26])
    with torch.no_grad():
        reference = tower.layers[0].self_attn(hs, cu)
    _share_segment_splits(tower, splits)
    splits.value = None
    with torch.no_grad():
        assert torch.equal(tower.layers[0].self_attn(hs, cu), reference)


def test_mismatched_splits_fall_back_instead_of_corrupting() -> None:
    torch.manual_seed(0)
    tower, splits = _Tower(), _SegmentSplits()
    hs, cu = _inputs([104, 26])
    with torch.no_grad():
        reference = tower.layers[0].self_attn(hs, cu)
    _share_segment_splits(tower, splits)
    splits.value = [64, 64]
    with torch.no_grad():
        assert torch.equal(tower.layers[0].self_attn(hs, cu), reference)


def test_shared_splits_do_not_copy_from_device_per_layer() -> None:
    """The whole point is one host round-trip per request, not one per layer."""
    tower, splits = _Tower(), _SegmentSplits()
    _share_segment_splits(tower, splits)
    segments = [104, 26]
    hs, cu = _inputs(segments)
    splits.value = segments
    calls = {"n": 0}
    original = torch.Tensor.tolist

    def counting_tolist(self):
        calls["n"] += 1
        return original(self)

    torch.Tensor.tolist = counting_tolist
    try:
        with torch.no_grad():
            for layer in tower.layers:
                layer.self_attn(hs, cu)
    finally:
        torch.Tensor.tolist = original
    assert calls["n"] == 0
