# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity packed DiT preserves ragged outputs across graph replays."""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from unittest.mock import MagicMock

import pytest
import torch
from torch._dynamo.exc import Unsupported

from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT
from sglang_omni.models.minicpm_o.components.token2wav.fixed_packed import (
    build_fixed_packed_layout,
    pack_fixed_capacity,
    unpack_fixed_capacity,
)
from sglang_omni.models.minicpm_o.components.token2wav.flow import CausalConditionalCFM
from sglang_omni.models.minicpm_o.components.token2wav.flow_cuda_graph import (
    FlowCudaGraphRunner,
)
from sglang_omni.models.minicpm_o.components.token2wav.packed_dit_cuda_graph import (
    CapturedPackedDiTGraph,
    PackedDiTCudaGraphRunner,
    PackedDiTWorkspace,
)


@pytest.mark.parametrize("row_lengths,capacity", [((4, 2), 8), ((8, 1, 8, 1), 20)])
def test_fixed_packing_preserves_valid_rows(
    row_lengths: tuple[int, ...], capacity: int
) -> None:
    frames = max(row_lengths)
    source = torch.arange(
        1, len(row_lengths) * frames * 3 + 1, dtype=torch.float32
    ).reshape(len(row_lengths), frames, 3)
    layout = build_fixed_packed_layout(
        torch.tensor(row_lengths, dtype=torch.int32), frames, capacity, 2
    )
    packed = pack_fixed_capacity(source, layout)
    static_input = torch.full_like(packed, float("nan"))
    actual = pack_fixed_capacity(source, layout, out=static_input)
    assert actual.data_ptr() == static_input.data_ptr()
    torch.testing.assert_close(actual, packed)
    valid_rows = torch.cat(
        [source[index, :length] for index, length in enumerate(row_lengths)]
    )
    torch.testing.assert_close(packed[: sum(row_lengths)], valid_rows)
    torch.testing.assert_close(
        packed[sum(row_lengths) :], torch.zeros_like(packed[sum(row_lengths) :])
    )
    restored = unpack_fixed_capacity(packed, layout)
    for index, length in enumerate(row_lengths):
        torch.testing.assert_close(restored[index, :length], source[index, :length])
        torch.testing.assert_close(
            restored[index, length:], torch.zeros_like(source[index, length:])
        )


def test_packed_graph_capacity_fit_is_independent_of_mel_width() -> None:
    dit = DiT(in_channels=16, out_channels=4, depth=1, hidden_size=32)
    runner = PackedDiTCudaGraphRunner(dit, device=torch.device("cuda:0"))
    runner.graphs = {
        (2, 128): MagicMock(),
        (2, 160): MagicMock(),
        (2, 4096): MagicMock(),
    }
    assert runner.fit(2, 32, 102) == 128
    assert runner.fit(2, 48, 122) == 128
    assert runner.fit(2, 40, 146) == 160
    assert runner.fit(2, 32, 80) is None
    assert runner.fit(2, 1025, 4000) is None


def test_capture_rejects_capacity_beyond_workspace_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dit = DiT(in_channels=16, out_channels=4, depth=1, hidden_size=32)
    runner = PackedDiTCudaGraphRunner(dit, device=torch.device("cuda:0"))
    current_stream = MagicMock(side_effect=AssertionError("CUDA must not be used"))
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    with pytest.raises(ValueError, match="capacities"):
        runner.capture(((2, 8192),))
    current_stream.assert_not_called()


@pytest.fixture
def cpu_capture_runner(monkeypatch: pytest.MonkeyPatch) -> PackedDiTCudaGraphRunner:
    dit = DiT(in_channels=16, out_channels=4, depth=1, hidden_size=32)
    runner = PackedDiTCudaGraphRunner(dit, device=torch.device("cuda:0"))
    runner.device = torch.device("cpu")
    stream = MagicMock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: (1, 1))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (10 * 1024**3, 0))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", MagicMock)
    monkeypatch.setattr(torch.cuda, "graph", lambda **kwargs: nullcontext())
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    return runner


def test_compiler_warmup_error_fails_capture_startup(
    cpu_capture_runner: PackedDiTCudaGraphRunner,
) -> None:
    cpu_capture_runner.estimator.run_packed_blocks = MagicMock(
        side_effect=RuntimeError("compiler warmup failed")
    )
    with pytest.raises(RuntimeError, match="compiler warmup failed"):
        cpu_capture_runner.capture(((2, 128),))


def test_compiler_error_inside_capture_is_not_an_eager_fallback(
    cpu_capture_runner: PackedDiTCudaGraphRunner,
) -> None:
    cpu_capture_runner.estimator.run_packed_blocks = MagicMock(
        side_effect=[torch.zeros(128, 4)] * 3 + [Unsupported("compiler guard failed")]
    )
    with pytest.raises(Unsupported, match="compiler guard failed"):
        cpu_capture_runner.capture(((2, 128),))


def test_graph_capture_failure_retains_eager_fallback(
    cpu_capture_runner: PackedDiTCudaGraphRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_capture_runner.estimator.run_packed_blocks = MagicMock(
        return_value=torch.zeros(128, 4)
    )
    monkeypatch.setattr(
        torch.cuda, "graph", MagicMock(side_effect=RuntimeError("capture unsupported"))
    )
    cpu_capture_runner.capture(((2, 128),))
    assert cpu_capture_runner.graphs == {}


def test_packed_workspace_reuses_capacity_across_dense_widths() -> None:
    dit = DiT(in_channels=16, out_channels=4, depth=1, num_heads=1, hidden_size=4)
    runner = PackedDiTCudaGraphRunner(dit, device=torch.device("cuda:0"))
    dit.packed_graph_runner = runner
    workspace = PackedDiTWorkspace(
        dense_input=torch.full((4 * 64, 4), float("nan")),
        conditioning=torch.zeros(5, 4),
        packed_output=torch.full((160, 4), float("nan")),
    )

    def replay_identity(inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        packed = workspace.dense_input[inputs[0]]
        packed = packed + workspace.conditioning[inputs[2]]
        output.copy_(packed.masked_fill(inputs[1][:, None], 0))

    for capacity in (128, 160):
        layout = build_fixed_packed_layout(
            torch.tensor([16] * 4, dtype=torch.int32), 16, capacity, 2
        )
        inputs = (
            layout.source_positions,
            layout.packed_padding_mask,
            layout.row_ids,
            layout.cu_seqlens,
            layout.conv_positions,
            layout.conv_valid,
        )
        graph = MagicMock()
        output = workspace.packed_output[:capacity]
        graph.replay.side_effect = partial(replay_identity, inputs, output)
        runner.graphs[(2, capacity)] = CapturedPackedDiTGraph(
            graph=graph, inputs=inputs, output=output, workspace=workspace
        )

    previous: list[tuple[torch.Tensor, torch.Tensor]] = []
    for lengths in ((32, 19), (36, 25), (40, 33), (32, 19)):
        frames = max(lengths)
        capacity = runner.fit(2, frames, 2 * sum(lengths))
        layout = build_fixed_packed_layout(
            torch.tensor(lengths * 2, dtype=torch.int32), frames, capacity, 2
        )
        source = torch.randn(4, frames, 4)
        conditioning = torch.randn(4, 1, 4)
        captured = runner.prepare(layout)
        actual = dit.forward_packed_fixed(source, conditioning, layout, captured)
        expected = (
            (source + conditioning)
            .masked_fill(layout.padding_mask[:, :, None], 0)
            .transpose(1, 2)
        )
        previous.append((actual, expected))
    for actual, expected in previous:
        torch.testing.assert_close(actual, expected)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_flow_graphs", [False, True])
def test_packed_graph_replays_changed_layout_and_falls_back(
    dtype: torch.dtype, with_flow_graphs: bool
) -> None:
    torch.manual_seed(29)
    dit = DiT(
        in_channels=16,
        out_channels=4,
        depth=1,
        num_heads=2,
        head_dim=16,
        hidden_size=32,
        enable_variable_length=True,
    )
    torch.nn.init.normal_(dit.blocks[0].adaLN_modulation[-1].weight, std=0.1)
    torch.nn.init.normal_(dit.final_layer.linear.weight, std=0.1)
    decoder = CausalConditionalCFM(dit).to(device="cuda", dtype=dtype).eval()
    decoder.requires_grad_(False)
    runner = PackedDiTCudaGraphRunner(dit, device=torch.device("cuda:0"))
    runner.capture(((2, 128), (2, 160)))
    assert set(runner.graphs) == {(2, 128), (2, 160)}
    if with_flow_graphs:
        dit.packed_graph_runner = runner
        flow_runner = FlowCudaGraphRunner(decoder, device=torch.device("cuda:0"))
        flow_runner.capture(((2, 32), (2, 48)))
        assert set(flow_runner.graphs) == {(2, 32), (2, 48)}
        dit.packed_graph_runner = None
    else:
        flow_runner = None

    spks = torch.randn(2, 4, device="cuda", dtype=dtype)
    requests: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    for profile in ((32, 19), (36, 25), (40, 33), (50, 29), (64, 60)):
        frames = max(profile)
        mu = torch.randn(2, 4, frames, device="cuda", dtype=dtype)
        cond = torch.randn_like(mu)
        widths = torch.tensor(profile, device="cuda")
        mask = (
            torch.arange(frames, device="cuda")[None, None, :] < widths[:, None, None]
        ).to(mu)
        requests.append((mu, mask, cond, 2 * sum(profile)))

    previous: list[tuple[torch.Tensor, torch.Tensor]] = []
    for mu, mask, cond, valid_frames in (
        requests[0],
        requests[1],
        requests[0],
        requests[2],
        requests[3],
        requests[4],
    ):
        eager = decoder(mu, mask, spks, cond, packed_valid_frames=valid_frames)
        initial_noise = decoder.rand_noise[:, :, : mu.shape[2]].expand_as(eager)
        assert not torch.equal(eager, initial_noise)
        dit.packed_graph_runner = runner
        decoder.graph_runner = flow_runner
        actual = decoder(mu, mask, spks, cond, packed_valid_frames=valid_frames)
        dit.packed_graph_runner = None
        decoder.graph_runner = None
        torch.testing.assert_close(actual, eager, atol=0.03, rtol=0.03)
        previous.append((actual, eager))
    for actual, expected in previous:
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    assert runner.graph_replays == 50
    assert runner.graph_misses == 1
    if flow_runner is not None:
        assert flow_runner.graph_replays == 80
        assert flow_runner.packed_step_replays == 40
        assert flow_runner.graph_misses == 1
    else:
        pass
