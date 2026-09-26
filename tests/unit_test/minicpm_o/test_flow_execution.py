# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o Flow execution preserves valid-frame outputs."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT
from sglang_omni.models.minicpm_o.components.token2wav.fixed_packed import (
    build_fixed_packed_layout,
)
from sglang_omni.models.minicpm_o.components.token2wav.flow import CausalConditionalCFM
from sglang_omni.models.minicpm_o.components.token2wav.flow_cuda_graph import (
    CapturedFlowGraph,
    CapturedPackedFlowGraph,
    FlowCudaGraphRunner,
)
from sglang_omni.models.minicpm_o.components.token2wav.flow_graph_shapes import (
    build_default_flow_cuda_graph_shapes,
)
from sglang_omni.models.minicpm_o.components.token2wav.packed_dit_cuda_graph import (
    CapturedPackedDiTGraph,
    PackedDiTCudaGraphRunner,
    PackedDiTWorkspace,
)


def small_decoder(device: str = "cpu") -> CausalConditionalCFM:
    dit = DiT(
        in_channels=16,
        out_channels=4,
        depth=1,
        num_heads=2,
        head_dim=16,
        hidden_size=32,
    )
    torch.nn.init.normal_(dit.blocks[0].adaLN_modulation[-1].weight, std=0.01)
    torch.nn.init.normal_(dit.final_layer.linear.weight, std=0.01)
    return CausalConditionalCFM(dit).to(device).eval()


def flow_inputs(
    device: str, frames: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = torch.randn(1, 4, frames, device=device)
    mask = torch.ones(1, 1, frames, device=device)
    spks = torch.randn(1, 4, device=device)
    cond = torch.randn_like(mu)
    return mu, mask, spks, cond


def test_default_graph_shapes_cover_dense_region_and_sparse_tails() -> None:
    shapes = build_default_flow_cuda_graph_shapes()
    singleton = [shape for shape in shapes if shape[0] == 1]
    assert len(shapes) == 141
    assert len(singleton) == 44
    assert shapes[0] == (1, 128)
    assert singleton[-1] == (1, 1024)
    assert all((1, frames) in shapes for frames in range(272, 721, 16))
    assert (
        max(right[1] - left[1] for left, right in zip(singleton, singleton[1:])) == 32
    )
    assert (2, 336) in shapes and (8, 784) in shapes


def test_flow_graph_fit_uses_nearest_mel_bucket() -> None:
    decoder = small_decoder()
    runner = FlowCudaGraphRunner(decoder, device=torch.device("cuda:0"))
    runner.device = torch.device("cpu")
    runner.graphs = {(1, 32): MagicMock(), (2, 32): MagicMock()}
    assert runner.fit(torch.zeros(1, 4, 31)) == (1, 32)
    assert runner.fit(torch.zeros(2, 4, 31)) == (2, 32)
    assert runner.fit(torch.zeros(1, 4, 33)) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_packed_graph_state_matches_eager_type_promotion(dtype: torch.dtype) -> None:
    torch.manual_seed(29)
    decoder = small_decoder().to(dtype=dtype)
    runner = FlowCudaGraphRunner(decoder, device=torch.device("cuda:0"))
    runner.device = torch.device("cpu")
    captured_state = runner.capture_inputs(2, 32, is_packed=True)[0]
    eager_state = decoder.rand_noise[:, :, :32].expand(2, -1, -1).clone()
    time_span = torch.linspace(0, 1, 11, dtype=dtype)
    time_span = 1 - torch.cos(time_span * 0.5 * torch.pi)
    velocities = torch.randn(10, 2, 4, 32, dtype=torch.bfloat16)
    for step, velocity in enumerate(velocities):
        dt = time_span[step + 1] - time_span[step]
        eager_state = eager_state + dt * velocity
        captured_state.copy_(captured_state + dt * velocity)
    assert captured_state.dtype == eager_state.dtype
    torch.testing.assert_close(captured_state, eager_state, rtol=0, atol=0)


def test_right_padding_preserves_valid_flow_frames() -> None:
    torch.manual_seed(7)
    decoder = small_decoder()
    mu, mask, spks, cond = flow_inputs("cpu", 13)
    eager = decoder(mu, mask, spks, cond)
    padded = decoder(
        F.pad(mu, (0, 3)),
        F.pad(mask, (0, 3)),
        spks,
        F.pad(cond, (0, 3)),
    )[:, :, :13]
    torch.testing.assert_close(padded, eager, atol=1e-5, rtol=1e-5)


def test_flow_graph_state_is_reset_and_outputs_remain_independent() -> None:
    torch.manual_seed(13)
    decoder = small_decoder()
    static_inputs = (
        torch.zeros(1, 4, 16),
        torch.zeros(1),
        torch.tensor(0.1),
        torch.zeros(2, 4, 16),
        torch.ones(2, 1, 16),
        torch.zeros(2, 4),
        torch.zeros(2, 4, 16),
    )
    graph = MagicMock()
    graph.replay.side_effect = lambda: static_inputs[0].copy_(
        decoder.euler_step(*static_inputs)
    )
    runner = FlowCudaGraphRunner(decoder, device=torch.device("cuda:0"))
    runner.device = torch.device("cpu")
    runner.graphs = {
        (1, 16): CapturedFlowGraph(
            graph=graph, inputs=static_inputs, output=static_inputs[0]
        )
    }
    previous: list[tuple[torch.Tensor, torch.Tensor]] = []
    for frames in (13, 16, 13):
        inputs = flow_inputs("cpu", frames)
        expected = decoder(*inputs)
        decoder.graph_runner = runner
        actual = decoder(*inputs)
        decoder.graph_runner = None
        previous.append((actual, expected))
    for actual, expected in previous:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert runner.graph_replays == 30


def test_packed_flow_composes_graphs_and_resets_solver_state() -> None:
    torch.manual_seed(37)
    decoder = small_decoder()
    static_inputs = (
        torch.zeros(2, 4, 16),
        torch.zeros(2),
        torch.tensor(0.1),
        torch.zeros(4, 4, 16),
        torch.ones(4, 1, 16),
        torch.zeros(4, 4),
        torch.zeros(4, 4, 16),
    )
    prediction = torch.empty_like(static_inputs[0])
    order: list[str] = []
    pre_graph, core_graph, post_graph = MagicMock(), MagicMock(), MagicMock()
    pre_graph.replay.side_effect = lambda: order.append("pre")

    def run_core() -> None:
        order.append("dit")
        decoder.estimator.enable_variable_length = False
        prediction.copy_(decoder.euler_step(*static_inputs))
        decoder.estimator.enable_variable_length = True

    def run_post() -> None:
        order.append("post")
        static_inputs[0].copy_(prediction)

    core_graph.replay.side_effect = run_core
    post_graph.replay.side_effect = run_post
    flow_runner = FlowCudaGraphRunner(decoder, device=torch.device("cuda:0"))
    flow_runner.device = torch.device("cpu")
    flow_runner.graphs[(2, 16)] = CapturedPackedFlowGraph(
        pre_graph=pre_graph,
        post_graph=post_graph,
        inputs=static_inputs,
        output=static_inputs[0],
        positions=torch.zeros(64, dtype=torch.long),
        padding_mask=torch.zeros(64, dtype=torch.bool),
    )
    packed_runner = PackedDiTCudaGraphRunner(
        decoder.estimator, device=torch.device("cuda:0")
    )
    layout = build_fixed_packed_layout(
        torch.tensor([15, 13] * 2, dtype=torch.int32), 16, 64, 2
    )
    packed_runner.graphs[(2, 64)] = CapturedPackedDiTGraph(
        graph=core_graph,
        inputs=(
            layout.source_positions,
            layout.packed_padding_mask,
            layout.row_ids,
            layout.cu_seqlens,
            layout.conv_positions,
            layout.conv_valid,
        ),
        output=prediction,
        workspace=PackedDiTWorkspace(
            dense_input=torch.empty(64, 32),
            conditioning=torch.zeros(5, 32),
            packed_output=torch.empty(64, 4),
        ),
    )
    previous: list[tuple[torch.Tensor, torch.Tensor]] = []
    for frames in (15, 16, 15):
        mu = torch.randn(2, 4, frames)
        mask = torch.ones(2, 1, frames)
        mask[1, :, frames - 2 :] = 0
        spks, cond = torch.randn(2, 4), torch.randn_like(mu)
        expected = decoder(mu, mask, spks, cond)
        decoder.estimator.enable_variable_length = True
        decoder.estimator.packed_graph_runner = packed_runner
        decoder.graph_runner = flow_runner
        actual = decoder(mu, mask, spks, cond, packed_valid_frames=2 * (2 * frames - 2))
        decoder.estimator.enable_variable_length = False
        decoder.estimator.packed_graph_runner = None
        decoder.graph_runner = None
        previous.append((actual, expected))
    for actual, expected in previous:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert order == ["pre", "dit", "post"] * 30
    assert flow_runner.graph_replays == 60
    assert flow_runner.packed_step_replays == 30
    assert packed_runner.graph_replays == 30


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flow_graph_replays_changed_inputs_and_falls_back() -> None:
    torch.manual_seed(11)
    decoder = small_decoder("cuda")
    decoder.estimator.enable_variable_length = True
    runner = FlowCudaGraphRunner(decoder, device=torch.device("cuda:0"))
    runner.capture(((1, 16), (2, 32)))
    assert set(runner.graphs) == {(1, 16)}

    previous: list[tuple[torch.Tensor, torch.Tensor]] = []
    for frames in (13, 16, 13, 17):
        inputs = flow_inputs("cuda", frames)
        eager = decoder(*inputs)
        decoder.graph_runner = runner
        actual = decoder(*inputs)
        decoder.graph_runner = None
        torch.testing.assert_close(actual, eager, atol=1e-4, rtol=1e-4)
        previous.append((actual, eager))
    for actual, expected in previous:
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    assert runner.graph_replays == 30
    assert runner.graph_misses == 1
