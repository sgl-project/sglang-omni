# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity packed DiT preserves ragged outputs across graph replays."""

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
from sglang_omni.models.minicpm_o.components.token2wav.flow import CausalConditionalCFM


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


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_graph_replays_changed_layout_and_falls_back(dtype: torch.dtype) -> None:
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
    runner.capture(((2, 128),))
    assert set(runner.graphs) == {(2, 128)}

    spks = torch.randn(2, 4, device="cuda", dtype=dtype)
    requests: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    for profile in ((32, 19), (36, 25), (40, 30)):
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
    ):
        eager = decoder(mu, mask, spks, cond, packed_valid_frames=valid_frames)
        initial_noise = decoder.rand_noise[:, :, : mu.shape[2]].expand_as(eager)
        assert not torch.equal(eager, initial_noise)
        dit.packed_graph_runner = runner
        actual = decoder(mu, mask, spks, cond, packed_valid_frames=valid_frames)
        dit.packed_graph_runner = None
        torch.testing.assert_close(actual, eager, atol=0.03, rtol=0.03)
        previous.append((actual, eager))
    for actual, expected in previous:
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    assert runner.graph_replays == 30
    assert runner.graph_misses == 1
