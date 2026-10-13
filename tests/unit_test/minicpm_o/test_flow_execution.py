# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o Flow execution preserves valid-frame outputs."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT, DiTState
from sglang_omni.models.minicpm_o.components.token2wav.fixed_packed import (
    build_fixed_packed_layout,
)
from sglang_omni.models.minicpm_o.components.token2wav.flow import CausalConditionalCFM
from sglang_omni.models.minicpm_o.components.token2wav.flow_cuda_graph import (
    CapturedPackedFlowGraph,
    FlowCudaGraphRunner,
)
from sglang_omni.models.minicpm_o.components.token2wav.packed_dit_cuda_graph import (
    CapturedPackedDiTGraph,
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
    device: str, frames: int, batch_size: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = torch.randn(batch_size, 4, frames, device=device)
    mask = torch.ones(batch_size, 1, frames, device=device)
    speaker_embeddings = torch.randn(batch_size, 4, device=device)
    mel_conditioning = torch.randn_like(mu)
    return mu, mask, speaker_embeddings, mel_conditioning


def test_flow_graph_fit_uses_nearest_mel_bucket() -> None:
    decoder = small_decoder()
    runner = FlowCudaGraphRunner(
        decoder.euler_step,
        decoder.rand_noise,
        estimator=decoder.estimator,
        inference_cfg_rate=decoder.inference_cfg_rate,
    )
    runner.device = torch.device("cpu")
    runner.graphs = {(1, 64): MagicMock(), (1, 32): MagicMock(), (2, 32): MagicMock()}
    assert runner.fit(torch.zeros(1, 4, 31), torch.zeros(2)) == (1, 32)
    assert runner.fit(torch.zeros(2, 4, 31), torch.zeros(2)) == (2, 32)
    assert runner.fit(torch.zeros(1, 4, 33), torch.zeros(2)) == (1, 64)
    assert runner.fit(torch.zeros(1, 4, 65), torch.zeros(2)) is None
    assert runner.fit(torch.zeros(3, 4, 31), torch.zeros(2)) is None
    assert (
        runner.fit(torch.zeros(1, 4, 31), torch.zeros(2, dtype=torch.float16)) is None
    )


def test_right_padding_preserves_valid_flow_frames() -> None:
    torch.manual_seed(7)
    decoder = small_decoder()
    mu, mask, speaker_embeddings, mel_conditioning = flow_inputs("cpu", 13)
    eager, _ = decoder(mu, mask, speaker_embeddings, mel_conditioning)
    padded, _ = decoder(
        F.pad(mu, (0, 3)),
        F.pad(mask, (0, 3)),
        speaker_embeddings,
        F.pad(mel_conditioning, (0, 3)),
    )
    torch.testing.assert_close(padded[:, :, :13], eager, atol=1e-5, rtol=1e-5)


def test_packed_flow_prepares_state_and_orders_replays() -> None:
    decoder = small_decoder()
    runner = FlowCudaGraphRunner(
        decoder.euler_step,
        decoder.rand_noise,
        estimator=decoder.estimator,
        inference_cfg_rate=decoder.inference_cfg_rate,
    )
    runner.device = torch.device("cpu")
    inputs = runner.capture_inputs(2, 32, is_packed=True)
    replay_order = MagicMock()
    captured = CapturedPackedFlowGraph(
        pre_graph=replay_order.pre,
        post_graph=replay_order.post,
        inputs=inputs,
        output=inputs[0],
        positions=torch.empty(128, dtype=torch.long),
        padding_mask=torch.empty(128, dtype=torch.bool),
    )
    runner.graphs[(2, 32)] = captured
    packed_graph = MagicMock(spec=CapturedPackedDiTGraph)
    for frames in (31, 29):
        x = torch.randn(2, 4, frames)
        conditioning = torch.randn(4, 4, frames)
        mask = torch.ones(4, 1, frames)
        speakers = torch.randn(4, 4)
        layout = build_fixed_packed_layout(
            torch.full((4,), frames, dtype=torch.int32), 32, 128, 2
        )
        runner.prepare((2, 32), x, conditioning, mask, speakers, conditioning, layout)
        torch.testing.assert_close(inputs[0], F.pad(x, (0, 32 - frames)))
        torch.testing.assert_close(inputs[4], F.pad(mask, (0, 32 - frames)))
        t, dt = torch.tensor(0.3), torch.tensor(0.1)
        runner.run_packed_step(captured, replay_order.packed, packed_graph, t, dt)
        torch.testing.assert_close(inputs[1], t.expand(2))
        torch.testing.assert_close(inputs[2], dt)
    assert (
        replay_order.mock_calls
        == [
            call.pre.replay(),
            call.packed.replay(packed_graph),
            call.post.replay(),
        ]
        * 2
    )


def test_streaming_flow_bypasses_graph_runners() -> None:
    decoder = small_decoder()
    decoder.graph_runner = MagicMock()
    decoder.estimator.forward_chunk = MagicMock(
        return_value=(torch.zeros(2, 4, 16), DiTState())
    )
    _, next_states = decoder(
        *flow_inputs("cpu", 16), states=[DiTState()], n_timesteps=1
    )
    assert len(next_states) == 1
    decoder.estimator.forward_chunk.assert_called_once()
    decoder.graph_runner.fit.assert_not_called()


def test_packed_batch_bypasses_dense_graphs() -> None:
    decoder = small_decoder()
    decoder.estimator.enable_variable_length = True
    decoder.graph_runner = MagicMock()
    decoder.estimator.forward = MagicMock(return_value=torch.zeros(4, 4, 16))
    decoder(*flow_inputs("cpu", 16, 2), n_timesteps=1)
    decoder.graph_runner.fit.assert_not_called()
    decoder.estimator.forward.assert_called_once()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch_size,compile_dense", [(1, False), (1, True), (2, True)])
def test_flow_graph_replays_changed_inputs_and_falls_back(
    batch_size: int, compile_dense: bool
) -> None:
    torch.manual_seed(11)
    decoder = small_decoder("cuda")
    decoder.estimator.enable_variable_length = False
    if compile_dense:
        decoder.half()
        for block in decoder.estimator.blocks:
            block.forward = torch.compile(
                block.forward, dynamic=True, options={"triton.cudagraphs": False}
            )
    else:
        pass
    runner = FlowCudaGraphRunner(
        decoder.euler_step,
        decoder.rand_noise,
        estimator=decoder.estimator,
        inference_cfg_rate=decoder.inference_cfg_rate,
    )
    runner.capture(((batch_size, 16),))
    assert set(runner.graphs) == {(batch_size, 16)}

    previous: list[tuple[torch.Tensor, torch.Tensor]] = []
    for frames in (13, 16, 13, 17):
        inputs = flow_inputs("cuda", frames, batch_size)
        with torch.autocast(
            "cuda", dtype=runner.dtype, enabled=runner.dtype != torch.float32
        ):
            eager, _ = decoder(*inputs)
            decoder.graph_runner = runner
            actual, _ = decoder(*inputs)
        decoder.graph_runner = None
        torch.testing.assert_close(actual, eager, atol=1e-3, rtol=1e-3)
        previous.append((actual, eager))
    for actual, expected in previous:
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    assert runner.graph_replays == 30
    assert runner.graph_misses == 1
