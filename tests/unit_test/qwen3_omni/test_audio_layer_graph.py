# SPDX-License-Identifier: Apache-2.0
"""Tests for the audio encoder layer-stack CUDA graph runner."""

from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from sglang.srt.layers.attention.vision import (
    VisionTritonAttention,
    prepare_vision_attention_metadata,
)
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components import audio_encoder
from sglang_omni.models.qwen3_omni.components.audio_layer_graph import (
    DEFAULT_TOKEN_BUCKETS,
    AudioLayerGraphRunner,
    stack_qkv_projection,
)
from sglang_omni.models.qwen3_omni.components.packed_attention import (
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
        tower,
        device=torch.device("cuda", 0),
        window=WINDOW,
        packed_attention=SegmentedSdpa(),
        **kwargs,
    )


def test_cpu_device_is_rejected() -> None:
    with pytest.raises(ValueError):
        AudioLayerGraphRunner(
            Tower(),
            device=torch.device("cpu"),
            window=WINDOW,
            packed_attention=SegmentedSdpa(),
        )


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
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)


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


def real_tower(device: torch.device) -> RealTower:
    tower = RealTower().to(device, torch.bfloat16)
    for layer in tower.layers:
        stack_qkv_projection(layer.self_attn)
    return tower


def test_stacked_projection_holds_q_k_and_v_once() -> None:
    torch.manual_seed(0)
    attention = RealAttention(HEADS * HEAD_DIM)
    projections = (attention.q_proj, attention.k_proj, attention.v_proj)
    hidden = torch.randn(7, HEADS * HEAD_DIM)
    with torch.no_grad():
        before = [projection(hidden) for projection in projections]
        stack_qkv_projection(attention)
        after = [projection(hidden) for projection in projections]
        stacked = attention.qkv_proj(hidden)
    for got, want in zip(after, before):
        assert torch.equal(got, want)
    torch.testing.assert_close(stacked, torch.cat(before, dim=-1))
    for projection in projections:
        assert (
            projection.weight.untyped_storage().data_ptr()
            == attention.qkv_proj.weight.untyped_storage().data_ptr()
        )
        assert (
            projection.bias.untyped_storage().data_ptr()
            == attention.qkv_proj.bias.untyped_storage().data_ptr()
        )


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
    packed_attention = resolve_packed_attention(device, torch.bfloat16)
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
            forward_metadata=prepare_vision_attention_metadata(
                cu_seqlens, device, max_seqlen=WINDOW
            ),
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
    tower = real_tower(device)
    runner = AudioLayerGraphRunner(
        tower,
        device=device,
        window=WINDOW,
        packed_attention=resolve_packed_attention(device, torch.bfloat16),
    )
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
    device = torch.device("cuda", 0)
    tower = real_tower(device)
    runner = AudioLayerGraphRunner(
        tower,
        device=device,
        window=WINDOW,
        packed_attention=resolve_packed_attention(device, torch.bfloat16),
    )
    runner.capture_all()
    assert runner.has_graphs, runner.disabled_reason
    assert sorted(runner.graphs) == sorted(DEFAULT_TOKEN_BUCKETS)


class DeadGraphCycle:
    def __init__(self, device: torch.device) -> None:
        self.cycle = self
        self.inputs = torch.zeros(8, device=device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = self.inputs + 1


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_capture_survives_a_collection_that_frees_dead_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture succeeds when a garbage collection runs inside it while dead graphs wait in a reference cycle."""
    device = torch.device("cuda", 0)
    runner = AudioLayerGraphRunner(
        real_tower(device),
        device=device,
        window=WINDOW,
        packed_attention=resolve_packed_attention(device, torch.bfloat16),
    )
    run_layers = runner.run_layers

    def run_layers_collecting(
        hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            gc.collect()
        else:
            pass
        return run_layers(hidden_states, cu_seqlens, max_seqlen)

    monkeypatch.setattr(runner, "run_layers", run_layers_collecting)
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        DeadGraphCycle(device)
        runner.capture_all()
    finally:
        if was_enabled:
            gc.enable()
        else:
            pass

    assert runner.has_graphs, runner.disabled_reason
    assert sorted(runner.graphs) == sorted(DEFAULT_TOKEN_BUCKETS)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend", ["platform", "triton"])
def test_replay_matches_the_uncaptured_packed_stack_across_bucket_boundaries(
    backend: Literal["platform", "triton"],
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda", 0)
    tower = real_tower(device)
    runner = AudioLayerGraphRunner(
        tower,
        device=device,
        window=WINDOW,
        packed_attention=(
            resolve_packed_attention(device, torch.bfloat16)
            if backend == "platform"
            else VisionTritonAttention()
        ),
    )
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


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("dtype", "enable_layer_cuda_graph", "replays", "tolerance"),
    [
        ("bf16", True, [True], 5e-3),
        ("bf16", False, [], 5e-3),
        ("fp32", True, [], 1e-5),
    ],
    ids=["bf16-graphs", "bf16-eager", "fp32"],
)
def test_cuda_audio_encoder_matches_the_fp32_transformers_tower(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dtype: Literal["bf16", "fp32"],
    enable_layer_cuda_graph: bool,
    replays: list[bool],
    tolerance: float,
) -> None:
    """A CUDA audio encoder encodes in fp32, and in bf16 with SGLANG_VIT_ENABLE_CUDA_GRAPH set."""
    monkeypatch.setenv("SGLANG_VIT_ENABLE_CUDA_GRAPH", "1")
    config = hf_modeling.Qwen3OmniMoeAudioEncoderConfig(
        d_model=256,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=512,
        downsample_hidden_size=16,
        output_dim=64,
    )
    torch.manual_seed(0)
    reference = hf_modeling.Qwen3OmniMoeAudioEncoder(config).eval()
    save_file(
        {
            f"thinker.audio_tower.{name}": tensor.contiguous()
            for name, tensor in reference.state_dict().items()
        },
        tmp_path / "model.safetensors",
    )
    monkeypatch.setattr(
        audio_encoder,
        "load_thinker_config",
        Mock(return_value=SimpleNamespace(audio_config=config)),
    )
    model = audio_encoder.Qwen3OmniAudioEncoder(
        str(tmp_path),
        device="cuda",
        dtype=dtype,
        enable_layer_cuda_graph=enable_layer_cuda_graph,
    )
    runner = model.layer_graph_runner
    assert (runner is not None) == bool(replays)
    recorded_replays: list[bool] = []
    if runner is not None:
        replay = runner.maybe_replay

        def recorded_replay(
            hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, segments: list[int]
        ) -> torch.Tensor | None:
            replayed = replay(hidden_states, cu_seqlens, segments)
            recorded_replays.append(replayed is not None)
            return replayed

        monkeypatch.setattr(runner, "maybe_replay", recorded_replay)
    else:
        pass
    reference = reference.to("cuda", torch.float32)
    lengths = torch.tensor([400, 850, 99])
    features = torch.randn(3, config.num_mel_bins, 850)

    with torch.no_grad():
        actual = model(
            input_features=features,
            feature_attention_mask=torch.arange(850) < lengths.unsqueeze(1),
        )["audio_embeds"]
        with sdpa_kernel(SDPBackend.MATH):
            expected = reference(
                torch.cat(
                    [row[:, :length] for row, length in zip(features, lengths)], dim=-1
                ).to("cuda"),
                feature_lens=lengths.to("cuda"),
            ).last_hidden_state

    assert recorded_replays == replays
    torch.testing.assert_close(actual.float(), expected, rtol=tolerance, atol=tolerance)
