# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o Flow execution preserves valid-frame outputs."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT, DiTBlock
from sglang_omni.models.minicpm_o.components.token2wav.flow import (
    CausalConditionalCFM,
    FlowCudaGraphRunner,
    build_default_flow_cuda_graph_shapes,
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


def test_compiled_blocks_share_one_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    dit = DiT(in_channels=16, out_channels=4, depth=2, hidden_size=32)
    compiled: list[tuple[object, bool]] = []

    def compile_forward(forward: object, *, dynamic: bool) -> object:
        compiled.append((forward, dynamic))
        return forward

    monkeypatch.setattr(torch, "compile", compile_forward)
    dit.enable_compiled_blocks()
    assert compiled == [(DiTBlock.forward, True)]
    assert all(block.forward.__func__ is DiTBlock.forward for block in dit.blocks)


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


def test_flow_graph_fit_uses_mel_table_for_packed_epilogue() -> None:
    decoder = small_decoder()
    decoder.estimator.enable_variable_length = True
    runner = FlowCudaGraphRunner(decoder, device=torch.device("cuda:0"))
    runner.device = torch.device("cpu")
    runner.graphs = {(1, 32): MagicMock()}
    runner.epilogues = {(2, 32): MagicMock()}
    assert runner.fit(torch.zeros(1, 4, 31)) == (1, 32)
    assert runner.fit(torch.zeros(2, 4, 31)) == (2, 32)


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


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compile_blocks", [False, True])
def test_flow_graph_replays_changed_inputs_and_falls_back(
    compile_blocks: bool,
) -> None:
    torch.manual_seed(11)
    decoder = small_decoder("cuda")
    if compile_blocks:
        decoder.estimator.enable_compiled_blocks()
    runner = FlowCudaGraphRunner(decoder, device=torch.device("cuda:0"))
    runner.capture(((1, 16),))

    for frames in (13, 16, 17):
        inputs = flow_inputs("cuda", frames)
        eager = decoder(*inputs)
        decoder.graph_runner = runner
        actual = decoder(*inputs)
        decoder.graph_runner = None
        torch.testing.assert_close(actual, eager, atol=1e-4, rtol=1e-4)
    assert runner.graph_replays == 20
    assert runner.graph_misses == 1
