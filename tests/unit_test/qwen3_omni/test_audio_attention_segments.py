# SPDX-License-Identifier: Apache-2.0
"""Tests for sharing audio-attention segment splits across encoder layers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.audio_attention import (
    FusedAudioAttention,
    project_audio_qkv,
)
from sglang_omni.models.qwen3_omni.components.audio_encoder import (
    SegmentSplits,
    share_segment_splits,
)

HEADS, HEAD_DIM = 4, 16
DIM = HEADS * HEAD_DIM


class Attention(nn.Module):
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
        self.config._attn_implementation = (
            "sdpa"  # noqa: leading-underscore  # production name
        )

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
            self.config._attn_implementation,
            hf_modeling.eager_attention_forward,  # noqa: leading-underscore  # production name
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


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention()


class Tower(nn.Module):
    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Layer() for _ in range(n))


def inputs(segments: list[int]):
    total = sum(segments)
    hs = torch.randn(total, DIM)
    cu = torch.tensor([0, *segments], dtype=torch.int32).cumsum(0).to(torch.int32)
    return hs, cu


def test_share_segment_splits_patches_every_layer() -> None:
    tower, splits = Tower(), SegmentSplits()
    share_segment_splits(tower, splits)
    for layer in tower.layers:
        assert (
            layer.self_attn._omni_segment_splits is splits
        )  # noqa: leading-underscore  # production name
        assert hasattr(layer.self_attn, "_omni_unshared_forward")


def test_shared_splits_match_the_per_layer_split_bitwise() -> None:
    torch.manual_seed(0)
    tower, splits = Tower(), SegmentSplits()
    segments = [104, 104, 52]
    hs, cu = inputs(segments)
    with torch.no_grad():
        reference = [layer.self_attn(hs, cu) for layer in tower.layers]
    share_segment_splits(tower, splits)
    splits.value = segments
    with torch.no_grad():
        shared = [layer.self_attn(hs, cu) for layer in tower.layers]
    for got, want in zip(shared, reference):
        assert torch.equal(got, want)


def test_missing_splits_fall_back_to_the_stock_path() -> None:
    torch.manual_seed(0)
    tower, splits = Tower(), SegmentSplits()
    hs, cu = inputs([104, 26])
    with torch.no_grad():
        reference = tower.layers[0].self_attn(hs, cu)
    share_segment_splits(tower, splits)
    splits.value = None
    with torch.no_grad():
        assert torch.equal(tower.layers[0].self_attn(hs, cu), reference)


def test_mismatched_splits_fall_back_instead_of_corrupting() -> None:
    torch.manual_seed(0)
    tower, splits = Tower(), SegmentSplits()
    hs, cu = inputs([104, 26])
    with torch.no_grad():
        reference = tower.layers[0].self_attn(hs, cu)
    share_segment_splits(tower, splits)
    splits.value = [64, 64]
    with torch.no_grad():
        assert torch.equal(tower.layers[0].self_attn(hs, cu), reference)


def test_shared_splits_do_not_copy_from_device_per_layer() -> None:
    """The whole point is one host round-trip per request, not one per layer."""
    tower, splits = Tower(), SegmentSplits()
    share_segment_splits(tower, splits)
    segments = [104, 26]
    hs, cu = inputs(segments)
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


@pytest.mark.parametrize(
    "biases", [(True, True, True), (False, False, False), (True, False, True)]
)
def test_fused_projection_preserves_loaded_weights_and_dtype_moves(
    biases: tuple[bool, bool, bool],
) -> None:
    torch.manual_seed(0)
    config = hf_modeling.Qwen3OmniMoeAudioEncoderConfig(
        d_model=DIM, encoder_attention_heads=HEADS
    )
    attention = hf_modeling.Qwen3OmniMoeAudioAttention(config).eval()
    for projection, has_bias in zip(
        (attention.q_proj, attention.k_proj, attention.v_proj), biases
    ):
        if not has_bias:
            projection.bias = None
        else:
            pass
    checkpoint = {
        name: torch.randn_like(parameter)
        for name, parameter in attention.state_dict().items()
    }
    attention.load_state_dict(checkpoint, strict=True)
    fused = FusedAudioAttention(attention, SegmentSplits()).to(torch.float64)
    attention = attention.to(torch.float64)
    hidden = torch.randn(19, DIM * 2, dtype=torch.float64)[:, ::2]
    with torch.no_grad():
        expected = project_audio_qkv(attention, hidden)
        actual = project_audio_qkv(fused, hidden)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=1e-12, rtol=1e-12)
    expected_parameters = sum(parameter.numel() for parameter in attention.parameters())
    missing_bias_elements = DIM * biases.count(False) if any(biases) else 0
    assert sum(parameter.numel() for parameter in fused.parameters()) == (
        expected_parameters + missing_bias_elements
    )


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("shared_splits", [[31, 17, 5], None, [8, 8]])
def test_fused_attention_matches_hf_with_shared_or_fallback_segments(
    implementation: str, shared_splits: list[int] | None
) -> None:
    torch.manual_seed(0)
    config = hf_modeling.Qwen3OmniMoeAudioEncoderConfig(
        d_model=DIM, encoder_attention_heads=HEADS
    )
    config._attn_implementation = implementation
    attention = hf_modeling.Qwen3OmniMoeAudioAttention(config).eval()
    splits = SegmentSplits()
    splits.value = shared_splits
    fused = FusedAudioAttention(attention, splits)
    hidden, cu_seqlens = _inputs([31, 17, 5])
    with torch.no_grad():
        reference = attention(hidden, cu_seqlens)
        actual = fused(hidden, cu_seqlens)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)


def test_fused_shared_segments_avoid_per_layer_host_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = hf_modeling.Qwen3OmniMoeAudioEncoderConfig(
        d_model=DIM, encoder_attention_heads=HEADS
    )
    config._attn_implementation = "sdpa"
    splits = SegmentSplits()
    splits.value = [31, 17, 5]
    fused = FusedAudioAttention(
        hf_modeling.Qwen3OmniMoeAudioAttention(config).eval(), splits
    )
    hidden, cu_seqlens = _inputs(splits.value)

    def reject_tolist(tensor: torch.Tensor) -> list[int]:
        raise AssertionError(
            "shared attention unexpectedly copied segment lengths to host"
        )

    monkeypatch.setattr(torch.Tensor, "tolist", reject_tolist)
    with torch.no_grad():
        result = fused(hidden, cu_seqlens)
    assert result.shape == hidden.shape


@pytest.mark.parametrize("shared_splits", [None, [8, 8]])
def test_fused_flash_fallback_preserves_variable_length_attention(
    monkeypatch: pytest.MonkeyPatch, shared_splits: list[int] | None
) -> None:
    torch.manual_seed(0)
    config = hf_modeling.Qwen3OmniMoeAudioEncoderConfig(
        d_model=DIM, encoder_attention_heads=HEADS
    )
    config._attn_implementation = "flash_attention_2"
    attention = hf_modeling.Qwen3OmniMoeAudioAttention(config).eval()
    splits = SegmentSplits()
    splits.value = shared_splits
    fused = FusedAudioAttention(attention, splits)
    hidden, cu_seqlens = _inputs([31, 17, 5])
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def flash_attention(
        module: hf_modeling.Qwen3OmniMoeAudioAttention | FusedAudioAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        max_length_q: torch.Tensor,
        max_length_k: torch.Tensor,
        is_causal: bool,
    ) -> tuple[torch.Tensor, None]:
        calls.append((cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k))
        return query.transpose(1, 2), None

    monkeypatch.setitem(
        hf_modeling.ALL_ATTENTION_FUNCTIONS, "flash_attention_2", flash_attention
    )
    with torch.no_grad():
        reference = attention(hidden, cu_seqlens)
        actual = fused(hidden, cu_seqlens)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
    assert len(calls) == 2
    for query_bounds, key_bounds, query_max, key_max in calls:
        assert torch.equal(query_bounds, cu_seqlens)
        assert torch.equal(key_bounds, cu_seqlens)
        assert query_max.item() == 31
        assert key_max.item() == 31
