# SPDX-License-Identifier: Apache-2.0
"""Tests for the audio encoder layer-stack CUDA graph runner."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sglang_omni.models.qwen3_omni.components.audio_layer_graph import (
    DEFAULT_TOKEN_BUCKETS,
    AudioLayerGraphRunner,
)

WINDOW = 104


class _Config:
    def __init__(self, d_model: int) -> None:
        self.d_model = d_model
        self._attn_implementation = "sdpa"


class _Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.config = _Config(dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)


class _Layer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = _Attention(dim)


class _Tower(nn.Module):
    def __init__(self, dim: int = 8, n: int = 2) -> None:
        super().__init__()
        self.config = _Config(dim)
        self.layers = nn.ModuleList(_Layer(dim) for _ in range(n))


def _runner(**kwargs) -> AudioLayerGraphRunner:
    tower = _Tower()
    return AudioLayerGraphRunner(
        tower, device=torch.device("cuda", 0), window=WINDOW, **kwargs
    )


def test_cpu_device_is_rejected() -> None:
    with pytest.raises(ValueError):
        AudioLayerGraphRunner(_Tower(), device=torch.device("cpu"), window=WINDOW)


def test_runner_without_captured_graphs_declines() -> None:
    runner = _runner()
    assert runner.has_graphs is False
    hidden = torch.zeros(4, 8)
    assert runner.maybe_replay(hidden, torch.zeros(3), [2, 2]) is None


def test_segment_slots_cover_every_batch_row() -> None:
    runner = _runner(max_batch_rows=32)
    # A bucket of 256 tokens holds 2 windows, but 32 rows can each add one more.
    assert runner._segment_slots(256) >= 256 // WINDOW + 32


def test_window_segments_bound_each_segment() -> None:
    runner = _runner()
    assert runner._window_segments(0) == []
    assert runner._window_segments(23) == [23]
    assert runner._window_segments(126) == [WINDOW, 22]
    assert runner._window_segments(WINDOW * 2) == [WINDOW, WINDOW]


def test_bucket_selection_picks_the_smallest_that_fits() -> None:
    runner = _runner()
    runner._graphs = {
        b: type("C", (), {"segment_slots": 64})() for b in DEFAULT_TOKEN_BUCKETS
    }
    assert runner._select(100, [25, 25, 25, 25]) == 128
    assert runner._select(130, [104, 26]) == 256
    assert runner._select(600, [104, 104, 104, 104, 104, 80]) == 1024


def test_bucket_selection_declines_beyond_the_largest_bucket() -> None:
    runner = _runner()
    runner._graphs = {
        b: type("C", (), {"segment_slots": 64})() for b in DEFAULT_TOKEN_BUCKETS
    }
    tokens = max(DEFAULT_TOKEN_BUCKETS) + 1
    assert runner._select(tokens, runner._window_segments(tokens)) is None


def test_bucket_selection_declines_when_segments_exceed_slots() -> None:
    runner = _runner()
    runner._graphs = {
        b: type("C", (), {"segment_slots": 4})() for b in DEFAULT_TOKEN_BUCKETS
    }
    assert runner._select(100, [1] * 100) is None


def test_bucket_selection_counts_split_padding_segments() -> None:
    runner = _runner(token_buckets=(256,))
    runner._graphs = {256: type("C", (), {"segment_slots": 3})()}
    # 129 live tokens occupy two segments. The 127 padding rows need two more
    # window-bounded segments, so a three-slot capture cannot serve the replay.
    assert runner._select(129, [104, 25]) is None


@pytest.mark.parametrize("segments", ([104, -4], [104, 1], [WINDOW + 1]))
def test_bucket_selection_declines_invalid_live_segments(segments: list[int]) -> None:
    runner = _runner()
    runner._graphs = {
        b: type("C", (), {"segment_slots": 64})() for b in DEFAULT_TOKEN_BUCKETS
    }
    assert runner._select(100, segments) is None


def test_disabled_runner_declines_even_with_graphs() -> None:
    runner = _runner()
    runner._graphs = {128: type("C", (), {"segment_slots": 64})()}
    runner._disabled_reason = "capture failed"
    assert runner.has_graphs is False
    assert runner.maybe_replay(torch.zeros(4, 8), torch.zeros(3), [2, 2]) is None


def test_varlen_install_does_not_mutate_the_shared_config() -> None:
    """The config is shared with the rest of the thinker; flipping it in place
    would switch attention for unrelated components."""
    tower = _Tower()
    shared = tower.layers[0].self_attn.config
    tower.layers[1].self_attn.config = shared
    tower.config = shared
    runner = AudioLayerGraphRunner(tower, device=torch.device("cuda", 0), window=WINDOW)
    runner._install_varlen()
    assert shared._attn_implementation == "sdpa"
    for layer in tower.layers:
        assert layer.self_attn._omni_varlen_config is not shared
        assert (
            layer.self_attn._omni_varlen_config._attn_implementation
            == "flash_attention_2"
        )


def test_capture_segments_fit_the_declared_window() -> None:
    runner = _runner(max_batch_rows=32)
    for bucket in DEFAULT_TOKEN_BUCKETS:
        segments = runner._capture_segments(bucket)
        assert len(segments) == runner._segment_slots(bucket)
        assert sum(segments) == bucket
        assert max(segments) <= WINDOW


class _RealAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.config = _Config(dim)
        self.num_heads = 1
        self.scaling = dim**-0.5
        self.attention_dropout = 0.0
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)


class _RealLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = _RealAttention(dim)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        self.activation_fn = nn.GELU()


class _RealTower(nn.Module):
    def __init__(self, dim: int = 64, n: int = 2) -> None:
        super().__init__()
        self.config = _Config(dim)
        self.layers = nn.ModuleList(_RealLayer(dim) for _ in range(n))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_capture_all_records_a_graph_for_every_bucket() -> None:
    tower = _RealTower().to(torch.device("cuda", 0), torch.bfloat16)
    runner = AudioLayerGraphRunner(tower, device=torch.device("cuda", 0), window=WINDOW)
    runner.capture_all()
    assert runner.has_graphs, runner._disabled_reason
    assert sorted(runner._graphs) == sorted(DEFAULT_TOKEN_BUCKETS)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_replay_matches_eager_varlen_across_bucket_boundaries() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda", 0)
    tower = _RealTower().to(device, torch.bfloat16)
    runner = AudioLayerGraphRunner(tower, device=device, window=WINDOW)
    runner.capture_all()
    assert runner.has_graphs, runner._disabled_reason

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
            cu_seqlens = (
                torch.tensor([0, *segments], dtype=torch.int32)
                .cumsum(0)
                .to(torch.int32)
                .to(device)
            )
            eager = runner._run_layers(hidden, cu_seqlens, WINDOW)
            replayed = runner.maybe_replay(hidden, cu_seqlens, segments)
            assert replayed is not None
            torch.testing.assert_close(replayed, eager)
