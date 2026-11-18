# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity packed DiT preserves ragged outputs across graph replays."""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
from torch._dynamo.exc import Unsupported

from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT
from sglang_omni.models.minicpm_o.components.token2wav.fixed_packed import (
    build_fixed_packed_layout,
    unpack_fixed_capacity,
)
from sglang_omni.models.minicpm_o.components.token2wav.flow import CausalConditionalCFM
from sglang_omni.models.minicpm_o.components.token2wav.flow_cuda_graph import (
    FlowCudaGraphRunner,
)
from sglang_omni.models.minicpm_o.components.token2wav.packed_dit_cuda_graph import (
    PACKED_DIT_GRAPH_MAX_SEQUENCE_LENGTH,
    PackedDiTCudaGraphRunner,
)


def packed_runner(dit: DiT, device: torch.device) -> PackedDiTCudaGraphRunner:
    return PackedDiTCudaGraphRunner(
        dit.run_packed_blocks,
        device=device,
        hidden_size=dit.in_proj.out_features,
        output_channels=dit.out_channels,
        convolution_guard_frames=dit.blocks[0].conv.kernel_size - 1,
    )


@pytest.mark.parametrize("row_lengths,capacity", [((4, 2), 8), ((8, 1, 8, 1), 20)])
def test_unpack_fixed_capacity_restores_valid_frames(
    row_lengths: tuple[int, ...], capacity: int
) -> None:
    layout = build_fixed_packed_layout(
        torch.tensor(row_lengths, dtype=torch.int32), max(row_lengths), capacity, 2
    )
    packed = torch.arange(capacity * 3, dtype=torch.float32).reshape(capacity, 3)
    restored = unpack_fixed_capacity(packed, layout)
    offset_frames = 0
    for index, length in enumerate(row_lengths):
        torch.testing.assert_close(
            restored[index, :length], packed[offset_frames : offset_frames + length]
        )
        assert torch.count_nonzero(restored[index, length:]) == 0
        offset_frames += length


def test_packed_graph_capacity_fit_is_independent_of_mel_width() -> None:
    dit = DiT(in_channels=16, out_channels=4, depth=1, hidden_size=32)
    runner = packed_runner(dit, torch.device("cuda:0"))
    runner.graphs = {
        (2, 128): MagicMock(),
        (2, 160): MagicMock(),
        (2, 4096): MagicMock(),
    }
    assert runner.fit(2, 32, 102) == 128
    assert runner.fit(2, 48, 122) == 128
    assert runner.fit(2, 40, 146) == 160
    assert runner.fit(2, 32, 128) == 160
    assert runner.fit(2, 32, 80) is None
    assert runner.fit(2, 1025, 4000) is None


def test_capture_rejects_capacity_beyond_workspace_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dit = DiT(in_channels=16, out_channels=4, depth=1, hidden_size=32)
    runner = packed_runner(dit, torch.device("cuda:0"))
    current_stream = MagicMock(side_effect=AssertionError("CUDA must not be used"))
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    with pytest.raises(ValueError, match="capacities"):
        runner.capture(((2, 8192),))
    current_stream.assert_not_called()


@pytest.fixture
def cpu_capture_runner(monkeypatch: pytest.MonkeyPatch) -> PackedDiTCudaGraphRunner:
    dit = DiT(in_channels=16, out_channels=4, depth=1, hidden_size=32)
    runner = packed_runner(dit, torch.device("cuda:0"))
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


@pytest.mark.parametrize("error_type", [RuntimeError, torch.cuda.OutOfMemoryError])
def test_packed_warmup_only_falls_back_on_out_of_memory(
    cpu_capture_runner: PackedDiTCudaGraphRunner, error_type: type[RuntimeError]
) -> None:
    cpu_capture_runner.run_packed_blocks = MagicMock(
        side_effect=error_type("warmup failed")
    )
    if error_type is torch.cuda.OutOfMemoryError:
        cpu_capture_runner.capture(((2, 128),))
        assert cpu_capture_runner.graphs == {}
    else:
        with pytest.raises(RuntimeError, match="warmup failed"):
            cpu_capture_runner.capture(((2, 128),))


def test_compiler_error_inside_capture_is_not_an_eager_fallback(
    cpu_capture_runner: PackedDiTCudaGraphRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_capture_runner.run_packed_blocks = MagicMock(return_value=torch.zeros(128, 4))
    monkeypatch.setattr(
        torch.cuda, "graph", MagicMock(side_effect=Unsupported("compiler guard failed"))
    )
    with pytest.raises(Unsupported, match="compiler guard failed"):
        cpu_capture_runner.capture(((2, 128),))


def test_graph_capture_failure_retains_eager_fallback(
    cpu_capture_runner: PackedDiTCudaGraphRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_capture_runner.run_packed_blocks = MagicMock(return_value=torch.zeros(128, 4))
    monkeypatch.setattr(
        torch.cuda, "graph", MagicMock(side_effect=RuntimeError("capture unsupported"))
    )
    cpu_capture_runner.capture(((2, 128),))
    assert cpu_capture_runner.graphs == {}


def test_capture_keeps_all_sequences_within_attention_length_bound(
    cpu_capture_runner: PackedDiTCudaGraphRunner,
) -> None:
    batch_size = 2
    capacity_frames = (2 * batch_size + 1) * PACKED_DIT_GRAPH_MAX_SEQUENCE_LENGTH - 1
    cpu_capture_runner.run_packed_blocks = MagicMock(
        return_value=torch.zeros(capacity_frames, 4)
    )
    cpu_capture_runner.capture(((batch_size, capacity_frames),))
    arguments = cpu_capture_runner.run_packed_blocks.call_args.args
    cumulative_sequence_lengths, maximum_sequence_length = arguments[3:5]
    assert cumulative_sequence_lengths.diff().max().item() <= maximum_sequence_length


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
    runner = packed_runner(dit, torch.device("cuda:0"))
    runner.capture(((2, 128), (2, 160)))
    assert set(runner.graphs) == {(2, 128), (2, 160)}
    if with_flow_graphs:
        dit.packed_graph_runner = runner
        flow_runner = FlowCudaGraphRunner(
            decoder.euler_step,
            decoder.rand_noise,
            estimator=dit,
            inference_cfg_rate=decoder.inference_cfg_rate,
        )
        flow_runner.capture(((2, 32), (2, 48)))
        assert set(flow_runner.graphs) == {(2, 32), (2, 48)}
        dit.packed_graph_runner = None
    else:
        flow_runner = None

    speaker_embeddings = torch.randn(2, 4, device="cuda", dtype=torch.float32)
    requests: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    for profile in ((32, 19), (36, 25), (40, 33), (50, 29), (64, 60)):
        frames = max(profile)
        mu = torch.randn(2, 4, frames, device="cuda", dtype=torch.float32)
        mel_conditioning = torch.randn_like(mu)
        widths = torch.tensor(profile, device="cuda")
        mask = (
            torch.arange(frames, device="cuda")[None, None, :] < widths[:, None, None]
        ).to(mu)
        requests.append((mu, mask, mel_conditioning, 2 * sum(profile)))

    previous: list[tuple[torch.Tensor, torch.Tensor]] = []
    for mu, mask, mel_conditioning, valid_frames in (
        requests[0],
        requests[1],
        requests[0],
        requests[2],
        requests[3],
        requests[4],
    ):
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            eager, _ = decoder(
                mu,
                mask,
                speaker_embeddings,
                mel_conditioning,
                packed_valid_frames=valid_frames,
            )
        initial_noise = decoder.rand_noise[:, :, : mu.shape[2]].expand_as(eager)
        assert not torch.equal(eager, initial_noise)
        dit.packed_graph_runner = runner
        decoder.graph_runner = flow_runner
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            actual, _ = decoder(
                mu,
                mask,
                speaker_embeddings,
                mel_conditioning,
                packed_valid_frames=valid_frames,
            )
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
