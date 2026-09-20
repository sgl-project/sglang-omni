# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedDiT,
    RaggedRowAttention,
    RowAttention,
    chunk_causal_mask,
    gather_rows,
    pack_rows,
    scatter_rows,
    solve_flow_euler_packed,
)
from sglang_omni.models.fun_cosyvoice3.stages import solve_flow_euler

cosyvoice_dit = pytest.importorskip("cosyvoice.flow.DiT.dit")
cosyvoice_mask = pytest.importorskip("cosyvoice.utils.mask")

CHANNELS = 8
SPEAKER = 8
LENGTHS = (11, 4, 19, 7)
CHUNK = 4
CPU = torch.device("cpu")


def _tiny_dit() -> torch.nn.Module:
    torch.manual_seed(0)
    dit = cosyvoice_dit.DiT(
        dim=32,
        depth=3,
        heads=2,
        dim_head=16,
        ff_mult=2,
        mel_dim=CHANNELS,
        mu_dim=CHANNELS,
        spk_dim=SPEAKER,
        out_channels=CHANNELS,
        static_chunk_size=CHUNK,
        num_decoding_left_chunks=-1,
        long_skip_connection=True,
    ).double()
    with torch.no_grad():
        for parameter in dit.parameters():
            parameter.normal_(0, 0.3)
    return dit.eval()


def _padded_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    rows, width = len(LENGTHS), max(LENGTHS)
    mask = torch.arange(width).unsqueeze(0) < torch.tensor(LENGTHS).unsqueeze(1)
    return {
        "x": torch.randn(rows, CHANNELS, width, dtype=torch.float64),
        "mu": torch.randn(rows, CHANNELS, width, dtype=torch.float64),
        "cond": torch.randn(rows, CHANNELS, width, dtype=torch.float64),
        "spks": torch.randn(rows, SPEAKER, dtype=torch.float64),
        "mask": mask.unsqueeze(1).double(),
        "t": torch.full((1,), 0.37, dtype=torch.float64),
    }


def _packed_inputs(padded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    rows = pack_rows(LENGTHS, CPU)
    return {
        "x": gather_rows(padded["x"].transpose(1, 2), rows),
        "mu": gather_rows(padded["mu"].transpose(1, 2), rows),
        "cond": gather_rows(padded["cond"].transpose(1, 2), rows),
        "spks": padded["spks"][rows.row_ids].unsqueeze(0),
        "t": padded["t"],
        "rows": rows,
    }


def _valid(padded: torch.Tensor) -> list[torch.Tensor]:
    return [padded[index, :, :length] for index, length in enumerate(LENGTHS)]


def test_pack_rows_describes_each_token_by_row_and_position() -> None:
    rows = pack_rows(LENGTHS, CPU)

    assert rows.total == 41
    assert rows.width == 19
    assert rows.starts_host.tolist() == [0, 11, 15, 34, 41]
    assert rows.row_ids.tolist() == [0] * 11 + [1] * 4 + [2] * 19 + [3] * 7
    assert rows.positions.tolist() == (
        list(range(11)) + list(range(4)) + list(range(19)) + list(range(7))
    )


def test_gather_then_scatter_keeps_the_valid_frames_and_zeroes_the_pad() -> None:
    rows = pack_rows(LENGTHS, CPU)
    padded = torch.randn(len(LENGTHS), 19, 3)

    packed = gather_rows(padded, rows)
    restored = scatter_rows(packed, rows, 19)

    assert packed.shape == (1, 41, 3)
    for index, length in enumerate(LENGTHS):
        torch.testing.assert_close(restored[index, :length], padded[index, :length])
        assert torch.count_nonzero(restored[index, length:]) == 0


def test_chunk_causal_mask_matches_cosyvoice() -> None:
    expected = cosyvoice_mask.subsequent_chunk_mask(19, CHUNK, -1, CPU)

    assert torch.equal(chunk_causal_mask(19, CHUNK, CPU), expected)


@pytest.mark.parametrize("chunk_size", [CHUNK, None])
def test_row_attention_matches_dense_attention_per_row(chunk_size: int | None) -> None:
    torch.manual_seed(2)
    rows = pack_rows(LENGTHS, CPU)
    query, key, value = (torch.randn(1, rows.total, 2 * 16) for _ in range(3))

    out = RowAttention(rows, chunk_size=chunk_size, heads=2)(query, key, value)

    for index, length in enumerate(LENGTHS):
        start = int(rows.starts_host[index])
        span = slice(start, start + length)
        heads = lambda part: part[0, span].view(length, 2, 16).transpose(0, 1)
        expected = torch.nn.functional.scaled_dot_product_attention(
            heads(query),
            heads(key),
            heads(value),
            attn_mask=(
                None
                if chunk_size is None
                else chunk_causal_mask(length, chunk_size, CPU)
            ),
        )
        torch.testing.assert_close(
            out[0, span], expected.transpose(0, 1).reshape(length, 32)
        )


@pytest.mark.parametrize("streaming", [True, False])
def test_packed_forward_matches_the_padded_dit_per_row(streaming: bool) -> None:
    dit = _tiny_dit()
    padded = _padded_inputs()
    packed = _packed_inputs(padded)
    estimator = PackedDiT(dit, device=CPU)

    with torch.inference_mode():
        expected = dit(
            padded["x"],
            padded["mask"],
            padded["mu"],
            padded["t"],
            padded["spks"],
            padded["cond"],
            streaming=streaming,
        )
        attention = estimator.row_attention(
            packed["rows"], streaming=streaming, dtype=packed["x"].dtype
        )
        out = estimator.forward(
            packed["x"],
            packed["mu"],
            packed["spks"],
            packed["cond"],
            packed["t"],
            packed["rows"],
            attention,
        )
    out = scatter_rows(out, packed["rows"], 19).transpose(1, 2)

    for actual, reference in zip(_valid(out), _valid(expected), strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("streaming", [True, False])
def test_packed_solve_matches_the_padded_solve_per_row(streaming: bool) -> None:
    dit = _tiny_dit()
    padded = _padded_inputs()
    packed = _packed_inputs(padded)
    decoder = SimpleNamespace(
        estimator=dit,
        inference_cfg_rate=0.7,
        forward_estimator=lambda x, mask, mu, t, spks, cond, streaming: dit(
            x, mask, mu, t, spks, cond, streaming=streaming
        ),
    )
    unit_span = torch.linspace(0, 1, 11, dtype=torch.float64)
    time_span = 1 - torch.cos(unit_span * 0.5 * torch.pi)
    noise = torch.randn(1, CHANNELS, 19, dtype=torch.float64).expand(4, -1, -1)

    with torch.inference_mode():
        expected = solve_flow_euler(
            decoder,
            noise.clone(),
            time_span,
            padded["mu"],
            padded["mask"],
            padded["spks"],
            padded["cond"],
            streaming=streaming,
        )
        out = solve_flow_euler_packed(
            PackedDiT(dit, device=CPU),
            gather_rows(noise.transpose(1, 2), packed["rows"]),
            time_span,
            packed["mu"],
            padded["spks"],
            packed["cond"],
            packed["rows"],
            cfg_rate=0.7,
            streaming=streaming,
        )
    out = scatter_rows(out, packed["rows"], 19).transpose(1, 2)

    for actual, reference in zip(_valid(out), _valid(expected), strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-9)


def test_a_wide_row_does_not_change_the_rows_packed_beside_it() -> None:
    dit = _tiny_dit()
    padded = _padded_inputs()
    packed = _packed_inputs(padded)
    estimator = PackedDiT(dit, device=CPU)

    with torch.inference_mode():
        together = estimator.forward(
            packed["x"],
            packed["mu"],
            packed["spks"],
            packed["cond"],
            packed["t"],
            packed["rows"],
            estimator.row_attention(
                packed["rows"], streaming=True, dtype=packed["x"].dtype
            ),
        )
        for index, length in enumerate(LENGTHS):
            rows = pack_rows((length,), CPU)
            alone = estimator.forward(
                padded["x"][index : index + 1, :, :length].transpose(1, 2),
                padded["mu"][index : index + 1, :, :length].transpose(1, 2),
                padded["spks"][index : index + 1].expand(length, -1).unsqueeze(0),
                padded["cond"][index : index + 1, :, :length].transpose(1, 2),
                padded["t"],
                rows,
                estimator.row_attention(rows, streaming=True, dtype=packed["x"].dtype),
            )
            start = int(packed["rows"].starts_host[index])
            torch.testing.assert_close(
                together[:, start : start + length], alone, rtol=1e-9, atol=1e-9
            )


@pytest.mark.accelerator
@pytest.mark.parametrize("chunk_size", [CHUNK, None])
def test_the_ragged_read_matches_the_padded_read(chunk_size: int | None) -> None:
    from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported

    device = torch.device("cuda")
    if not _is_fa3_supported():
        pytest.skip("FA3 is unavailable on this device")
    torch.manual_seed(2)
    heads = 2
    head_dim = 64
    rows = pack_rows(LENGTHS, device)
    query, key, value = (
        torch.randn(
            1, rows.total, heads * head_dim, device=device, dtype=torch.bfloat16
        )
        for _ in range(3)
    )
    ragged = RaggedRowAttention(
        rows, chunk_size=chunk_size, heads=heads, head_dim=head_dim
    )(query, key, value)
    padded = RowAttention(rows, chunk_size=chunk_size, heads=heads)(query, key, value)
    torch.testing.assert_close(ragged, padded, rtol=2e-2, atol=2e-2)
