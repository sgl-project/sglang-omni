# SPDX-License-Identifier: Apache-2.0
"""Tests for the audio encoder layer-stack CUDA graph runner."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components import audio_layer_graph
from sglang_omni.models.qwen3_omni.components.audio_attention import (
    FusedAudioAttention,
    SegmentSplits,
)
from sglang_omni.models.qwen3_omni.components.audio_layer_graph import (
    DEFAULT_TOKEN_BUCKETS,
    AudioLayerGraphRunner,
    packed_attention_backend,
    packed_attention_forward,
    resolve_packed_attention,
)

WINDOW = 104
HEADS = 4
HEAD_DIM = 64


class Config:
    def __init__(self, d_model: int) -> None:
        self.d_model = d_model


class Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.out_proj = nn.Linear(dim, dim, bias=False)


class Layer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = Attention(dim)


class Tower(nn.Module):
    def __init__(self, dim: int = 8, n: int = 2) -> None:
        super().__init__()
        self.config = Config(dim)
        self.layers = nn.ModuleList(Layer(dim) for _ in range(n))


def make_runner(**kwargs) -> AudioLayerGraphRunner:
    tower = Tower()
    return AudioLayerGraphRunner(
        tower, device=torch.device("cuda", 0), window=WINDOW, **kwargs
    )


def test_cpu_device_is_rejected() -> None:
    with pytest.raises(ValueError):
        AudioLayerGraphRunner(Tower(), device=torch.device("cpu"), window=WINDOW)


def test_runner_without_captured_graphs_declines() -> None:
    runner = make_runner()
    assert runner.has_graphs is False
    hidden = torch.zeros(4, 8)
    assert runner.maybe_replay(hidden, torch.zeros(3), [2, 2]) is None


def test_segment_slots_cover_every_batch_row() -> None:
    runner = make_runner(max_batch_rows=32)
    # A bucket of 256 tokens holds 2 windows, but 32 rows can each add one more.
    assert runner.segment_slots(256) >= 256 // WINDOW + 32


def test_window_segments_bound_each_segment() -> None:
    runner = make_runner()
    assert runner.window_segments(0) == []
    assert runner.window_segments(23) == [23]
    assert runner.window_segments(126) == [WINDOW, 22]
    assert runner.window_segments(WINDOW * 2) == [WINDOW, WINDOW]


def test_bucket_selection_picks_the_smallest_that_fits() -> None:
    runner = make_runner()
    runner.graphs = {
        b: type("C", (), {"segment_slots": 64})() for b in DEFAULT_TOKEN_BUCKETS
    }
    assert runner.select(100, [25, 25, 25, 25]) == 128
    assert runner.select(130, [104, 26]) == 256
    assert runner.select(600, [104, 104, 104, 104, 104, 80]) == 1024


def test_bucket_selection_declines_beyond_the_largest_bucket() -> None:
    runner = make_runner()
    runner.graphs = {
        b: type("C", (), {"segment_slots": 64})() for b in DEFAULT_TOKEN_BUCKETS
    }
    tokens = max(DEFAULT_TOKEN_BUCKETS) + 1
    assert runner.select(tokens, runner.window_segments(tokens)) is None


def test_bucket_selection_declines_when_segments_exceed_slots() -> None:
    runner = make_runner()
    runner.graphs = {
        b: type("C", (), {"segment_slots": 4})() for b in DEFAULT_TOKEN_BUCKETS
    }
    assert runner.select(100, [1] * 100) is None


def test_bucket_selection_counts_split_padding_segments() -> None:
    runner = make_runner(token_buckets=(256,))
    runner.graphs = {256: type("C", (), {"segment_slots": 3})()}
    # 129 live tokens occupy two segments. The 127 padding rows need two more
    # window-bounded segments, so a three-slot capture cannot serve the replay.
    assert runner.select(129, [104, 25]) is None


@pytest.mark.parametrize("segments", ([104, -4], [104, 1], [WINDOW + 1]))
def test_bucket_selection_declines_invalid_live_segments(segments: list[int]) -> None:
    runner = make_runner()
    runner.graphs = {
        b: type("C", (), {"segment_slots": 64})() for b in DEFAULT_TOKEN_BUCKETS
    }
    assert runner.select(100, segments) is None


def test_disabled_runner_declines_even_with_graphs() -> None:
    runner = make_runner()
    runner.graphs = {128: type("C", (), {"segment_slots": 64})()}
    runner.disabled_reason = "capture failed"
    assert runner.has_graphs is False
    assert runner.maybe_replay(torch.zeros(4, 8), torch.zeros(3), [2, 2]) is None


@pytest.mark.parametrize(
    ("capability", "backend"),
    (
        ((9, 0), "fa3"),
        ((8, 9), "triton_attn"),
        ((10, 0), "triton_attn"),
        ((12, 0), "triton_attn"),
    ),
)
def test_packed_attention_backend_follows_the_device_capability(
    capability: tuple[int, int], backend: str
) -> None:
    assert packed_attention_backend(capability, is_hip=False) == backend


def test_packed_attention_backend_keeps_hip_off_fa3() -> None:
    """PyTorch HIP reports gfx950 as (9, 5); that is not NVIDIA Hopper."""
    assert packed_attention_backend((9, 5), is_hip=True) == "triton_attn"
    assert packed_attention_backend((9, 0), is_hip=True) == "triton_attn"


def test_an_unresolvable_kernel_stack_stays_eager_with_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def resolve(device: torch.device) -> tuple[nn.Module, str]:
        raise ImportError("no flash attention build for this torch")

    monkeypatch.setattr(audio_layer_graph, "resolve_packed_attention", resolve)
    runner = make_runner()
    runner.capture_all()
    assert runner.has_graphs is False
    assert runner.graphs == {}
    assert "no flash attention build" in runner.disabled_reason


def test_capture_segments_fit_the_declared_window() -> None:
    runner = make_runner(max_batch_rows=32)
    for bucket in DEFAULT_TOKEN_BUCKETS:
        segments = runner.capture_segments(bucket)
        assert len(segments) == runner.segment_slots(bucket)
        assert sum(segments) == bucket
        assert max(segments) <= WINDOW


class RealAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.num_heads = HEADS
        self.scaling = HEAD_DIM**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)


class RealLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = RealAttention(dim)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        self.activation_fn = nn.GELU()


class RealTower(nn.Module):
    def __init__(self, dim: int = HEADS * HEAD_DIM, n: int = 2) -> None:
        super().__init__()
        self.config = Config(dim)
        self.layers = nn.ModuleList(RealLayer(dim) for _ in range(n))


def segmented_sdpa(q, k, v, cu_seqlens, softmax_scale):
    bounds = cu_seqlens.tolist()
    outputs = [
        F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1),
            k[start:end].transpose(0, 1),
            v[start:end].transpose(0, 1),
            scale=softmax_scale,
        ).transpose(0, 1)
        for start, end in zip(bounds, bounds[1:])
    ]
    return torch.cat(outputs, dim=0)


class SegmentedSdpa(nn.Module):
    def forward(self, q, k, v, cu_seqlens, bsz, seq_len, softmax_scale, **kwargs):
        return segmented_sdpa(q, k, v, cu_seqlens, softmax_scale)


def make_cu_seqlens(segments: list[int], device: torch.device) -> torch.Tensor:
    return (
        torch.tensor([0, *segments], dtype=torch.int32)
        .cumsum(0)
        .to(torch.int32)
        .to(device)
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_attention_matches_fp32_sdpa_per_segment_and_head() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda", 0)
    packed_attention, _ = resolve_packed_attention(device)
    segments = [104, 104, 49, 37, 90]
    q, k, v = (
        torch.randn(sum(segments), HEADS, HEAD_DIM, device=device).to(torch.bfloat16)
        for _ in range(3)
    )
    cu_seqlens = make_cu_seqlens(segments, device)
    with torch.no_grad():
        packed = packed_attention(
            q,
            k,
            v,
            cu_seqlens,
            bsz=1,
            seq_len=q.shape[0],
            softmax_scale=HEAD_DIM**-0.5,
            max_seqlen=WINDOW,
        )
        with sdpa_kernel(SDPBackend.MATH):
            reference = segmented_sdpa(
                q.float(), k.float(), v.float(), cu_seqlens, HEAD_DIM**-0.5
            )
    assert packed.shape == reference.shape
    torch.testing.assert_close(packed.float(), reference, rtol=1e-2, atol=1e-2)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_layer_stack_matches_segmented_sdpa() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda", 0)
    tower = RealTower().to(device, torch.bfloat16)
    runner = AudioLayerGraphRunner(tower, device=device, window=WINDOW)
    runner.resolve_attention()
    assert runner.disabled_reason is None, runner.disabled_reason
    segments = [104, 104, 49, 37, 90]
    hidden = torch.randn(sum(segments), tower.config.d_model, device=device).to(
        torch.bfloat16
    )
    cu_seqlens = make_cu_seqlens(segments, device)
    with torch.no_grad():
        packed = runner.run_layers(hidden, cu_seqlens, WINDOW)
        runner.packed_attention = SegmentedSdpa()
        reference = runner.run_layers(hidden, cu_seqlens, WINDOW)
    torch.testing.assert_close(packed, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_capture_all_records_a_graph_for_every_bucket() -> None:
    tower = RealTower().to(torch.device("cuda", 0), torch.bfloat16)
    runner = AudioLayerGraphRunner(tower, device=torch.device("cuda", 0), window=WINDOW)
    runner.capture_all()
    assert runner.has_graphs, runner.disabled_reason
    assert sorted(runner.graphs) == sorted(DEFAULT_TOKEN_BUCKETS)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_replay_matches_the_uncaptured_packed_stack_across_bucket_boundaries() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda", 0)
    tower = RealTower().to(device, torch.bfloat16)
    runner = AudioLayerGraphRunner(tower, device=device, window=WINDOW)
    runner.capture_all()
    assert runner.has_graphs, runner.disabled_reason

    cases = (
        [104, 23],
        [104, 24],
        [104, 25],
        [104, 104, 47],
        [104, 104, 48],
        [104, 104, 49],
        [37, 90],
    )
    with torch.no_grad():
        for segments in cases:
            tokens = sum(segments)
            hidden = torch.randn(tokens, tower.config.d_model, device=device).to(
                torch.bfloat16
            )
            cu_seqlens = make_cu_seqlens(segments, device)
            uncaptured = runner.run_layers(hidden, cu_seqlens, WINDOW)
            replayed = runner.maybe_replay(hidden, cu_seqlens, segments)
            assert replayed is not None
            torch.testing.assert_close(replayed, uncaptured)


def test_fused_packed_attention_matches_unfused_strided_inputs() -> None:
    torch.manual_seed(0)
    config = hf_modeling.Qwen3OmniMoeAudioEncoderConfig(
        d_model=HEADS * HEAD_DIM, encoder_attention_heads=HEADS
    )
    attention = hf_modeling.Qwen3OmniMoeAudioAttention(config).eval()
    fused = FusedAudioAttention(attention, SegmentSplits())
    segments = [31, 17, 5]
    hidden = torch.randn(sum(segments), config.d_model)
    cu_seqlens = _cu_seqlens(segments, torch.device("cpu"))
    with torch.no_grad():
        expected = packed_attention_forward(
            attention, _SegmentedSdpa(), hidden, cu_seqlens, WINDOW
        )
        actual = packed_attention_forward(
            fused, _SegmentedSdpa(), hidden, cu_seqlens, WINDOW
        )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_graph_matches_unfused_stack_across_bucket_boundaries() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda", 0)
    config = hf_modeling.Qwen3OmniMoeAudioEncoderConfig(
        d_model=HEADS * HEAD_DIM,
        encoder_attention_heads=HEADS,
        encoder_ffn_dim=2 * HEADS * HEAD_DIM,
    )
    tower = _RealTower()
    tower.layers = nn.ModuleList(
        hf_modeling.Qwen3OmniMoeAudioEncoderLayer(config) for _ in range(2)
    )
    tower = tower.to(device, torch.bfloat16).eval()
    runner = AudioLayerGraphRunner(
        tower, device=device, window=WINDOW, token_buckets=(128, 256, 512)
    )
    runner.resolve_attention()
    assert runner.disabled_reason is None, runner.disabled_reason
    cases = (
        [104, 23],
        [104, 24],
        [104, 25],
        [104, 104, 47],
        [104, 104, 48],
        [104, 104, 49],
        [37, 90],
    )
    hidden_states = [
        torch.randn(sum(segments), config.d_model, device=device, dtype=torch.bfloat16)
        for segments in cases
    ]
    with torch.no_grad():
        expected = [
            runner.run_layers(hidden, _cu_seqlens(segments, device), WINDOW)
            for hidden, segments in zip(hidden_states, cases)
        ]
        splits = SegmentSplits()
        for layer in tower.layers:
            layer.self_attn = FusedAudioAttention(layer.self_attn, splits)
        runner.capture_all()
        assert runner.has_graphs, runner.disabled_reason
        for hidden, segments, reference in zip(hidden_states, cases, expected):
            cu_seqlens = _cu_seqlens(segments, device)
            eager = runner.run_layers(hidden, cu_seqlens, WINDOW)
            replayed = runner.maybe_replay(hidden, cu_seqlens, segments)
            assert replayed is not None
            torch.testing.assert_close(eager, reference, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(replayed, reference, rtol=2e-2, atol=2e-2)
