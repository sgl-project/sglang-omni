# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity packed DiT keeps variable-length outputs graph-safe."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from sglang_omni.models.minicpm_o.components.token2wav.dit import (
    DiT,
    PackedDiTCudaGraphRunner,
)
from sglang_omni.models.minicpm_o.components.token2wav.fixed_packed import (
    build_fixed_packed_layout,
    pack_fixed_capacity,
    unpack_fixed_capacity,
)
from sglang_omni.models.minicpm_o.components.token2wav.flow import (
    CausalConditionalCFM,
    FlowCudaGraphRunner,
)


def test_fixed_packing_preserves_valid_rows() -> None:
    lengths = torch.tensor([4, 2], dtype=torch.int32)
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1)
    layout = build_fixed_packed_layout(
        lengths, padded_length=4, capacity=8, guard_width=2
    )
    packed = pack_fixed_capacity(source, layout)
    torch.testing.assert_close(
        packed[:6, 0], torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32)
    )
    torch.testing.assert_close(packed[6:, 0], torch.zeros(2))
    torch.testing.assert_close(
        layout.cu_seqlens, torch.tensor([0, 4, 6, 8], dtype=torch.int32)
    )
    restored = unpack_fixed_capacity(packed, layout)
    torch.testing.assert_close(restored[0], source[0])
    torch.testing.assert_close(restored[1, :2], source[1, :2])
    torch.testing.assert_close(restored[1, 2:], torch.zeros_like(source[1, 2:]))


def test_packed_graph_capacity_fit_is_independent_of_mel_width() -> None:
    dit = DiT(in_channels=16, out_channels=4, depth=1, hidden_size=32)
    runner = PackedDiTCudaGraphRunner(dit, device=torch.device("cuda:0"))
    runner.graphs = {(2, 128): MagicMock(), (2, 160): MagicMock()}
    assert runner.fit(2, 32, 102) == 128
    assert runner.fit(2, 48, 122) == 128
    assert runner.fit(2, 16, 146) == 160
    assert runner.fit(2, 5, 122) is None
    assert runner.fit(2, 1025, 122) is None


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fixed_packed_dit_graph_replays_changed_lengths() -> None:
    torch.manual_seed(17)
    dit = (
        DiT(
            in_channels=16,
            out_channels=4,
            depth=1,
            num_heads=2,
            head_dim=16,
            hidden_size=32,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    x = torch.randn(4, 32, 32, device="cuda", dtype=torch.bfloat16)
    conditioning = torch.randn(4, 1, 32, device="cuda", dtype=torch.bfloat16)
    profiles = ((32, 19, 32, 19), (32, 25, 32, 25))
    static_lengths = torch.tensor(profiles[0], device="cuda", dtype=torch.int32)
    current_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current_stream)
    with torch.cuda.stream(stream):
        for _ in range(3):
            dit.forward_packed_fixed(x, conditioning, static_lengths, capacity=128)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_output = dit.forward_packed_fixed(
                x, conditioning, static_lengths, capacity=128
            )
    current_stream.wait_stream(stream)

    for profile in profiles:
        lengths = torch.tensor(profile, device="cuda", dtype=torch.int32)
        expected = dit.forward_packed(x, conditioning, lengths)
        static_lengths.copy_(lengths)
        graph.replay()
        torch.testing.assert_close(captured_output, expected, atol=0.02, rtol=0.02)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fixed_packed_euler_step_graph_replays_changed_masks() -> None:
    torch.manual_seed(23)
    dit = (
        DiT(
            in_channels=16,
            out_channels=4,
            depth=1,
            num_heads=2,
            head_dim=16,
            hidden_size=32,
            enable_variable_length=True,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    torch.nn.init.normal_(dit.blocks[0].adaLN_modulation[-1].weight, std=0.01)
    torch.nn.init.normal_(dit.final_layer.linear.weight, std=0.01)
    decoder = CausalConditionalCFM(dit).eval().requires_grad_(False)
    x = torch.randn(2, 4, 32, device="cuda", dtype=torch.bfloat16)
    t = torch.zeros(2, device="cuda", dtype=torch.bfloat16)
    dt = torch.tensor(0.1, device="cuda", dtype=torch.bfloat16)
    mu = torch.randn(4, 4, 32, device="cuda", dtype=torch.bfloat16)
    spks = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    cond = torch.randn_like(mu)
    mask = torch.empty(4, 1, 32, device="cuda", dtype=torch.bfloat16)
    profiles = ((32, 19), (32, 25))

    def set_mask(lengths: tuple[int, int]) -> None:
        widths = torch.tensor(lengths * 2, device="cuda")
        mask.copy_(
            (torch.arange(32, device="cuda")[None, None, :] < widths[:, None, None]).to(
                mask
            )
        )

    set_mask(profiles[0])
    current_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current_stream)
    with torch.cuda.stream(stream):
        for _ in range(3):
            decoder.euler_step(x, t, dt, mu, mask, spks, cond, packed_capacity=128)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_output = decoder.euler_step(
                x, t, dt, mu, mask, spks, cond, packed_capacity=128
            )
    current_stream.wait_stream(stream)

    for profile in profiles:
        set_mask(profile)
        expected = decoder.euler_step(x, t, dt, mu, mask, spks, cond)
        graph.replay()
        torch.testing.assert_close(captured_output, expected, atol=0.02, rtol=0.02)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flow_runner_replays_packed_batch_with_changed_lengths() -> None:
    torch.manual_seed(29)
    dit = (
        DiT(
            in_channels=16,
            out_channels=4,
            depth=1,
            num_heads=2,
            head_dim=16,
            hidden_size=32,
            enable_variable_length=True,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    torch.nn.init.normal_(dit.blocks[0].adaLN_modulation[-1].weight, std=0.01)
    torch.nn.init.normal_(dit.final_layer.linear.weight, std=0.01)
    decoder = (
        CausalConditionalCFM(dit)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
        .requires_grad_(False)
    )
    runner = PackedDiTCudaGraphRunner(dit, device=torch.device("cuda:0"))
    runner.capture(((2, 128),))
    assert set(runner.graphs) == {(2, 128)}
    flow_runner = FlowCudaGraphRunner(decoder, device=torch.device("cuda:0"))
    flow_runner.capture(((2, 32), (2, 48)))
    assert set(flow_runner.epilogues) == {(2, 32), (2, 48)}

    spks = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
    for profile in ((32, 19), (36, 25)):
        frames = max(profile)
        mu = torch.randn(2, 4, frames, device="cuda", dtype=torch.bfloat16)
        cond = torch.randn_like(mu)
        widths = torch.tensor(profile, device="cuda")
        mask = (
            torch.arange(frames, device="cuda")[None, None, :] < widths[:, None, None]
        ).to(mu)
        packed_valid_frames = 2 * sum(profile)
        eager = decoder(
            mu,
            mask,
            spks,
            cond,
            packed_valid_frames=packed_valid_frames,
        )
        dit.packed_graph_runner = runner
        decoder.graph_runner = flow_runner
        actual = decoder(
            mu,
            mask,
            spks,
            cond,
            packed_valid_frames=packed_valid_frames,
        )
        dit.packed_graph_runner = None
        decoder.graph_runner = None
        torch.testing.assert_close(actual, eager, atol=0.03, rtol=0.03)
    assert runner.graph_replays == 20
    assert runner.graph_misses == 0
    assert flow_runner.graph_replays == 20
    assert flow_runner.graph_misses == 0
