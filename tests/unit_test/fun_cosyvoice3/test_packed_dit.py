# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.fun_cosyvoice3.causal_conv import (
    GROUP_CONV_KERNEL_CHANNELS,
    FusedConvPositionEmbedding,
    group_conv_mish,
    pack_group_conv_weight,
)
from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedDiT,
    PackedRows,
    RaggedRowAttention,
    RowAttention,
    WholeRowAttention,
    chunk_causal_mask,
    gather_rows,
    pack_rows,
    rotate_in_place,
    rotated,
    scatter_rows,
    solve_flow_euler_packed,
)
from sglang_omni.models.fun_cosyvoice3.stages import solve_flow_euler

cosyvoice_dit = pytest.importorskip("cosyvoice.flow.DiT.dit")
cosyvoice_mask = pytest.importorskip("cosyvoice.utils.mask")
cosyvoice_modules = pytest.importorskip("cosyvoice.flow.DiT.modules")

CHANNELS = 8
SPEAKER = 8
LENGTHS = (11, 4, 19, 7)
CHUNK = 4
CPU = torch.device("cpu")
CONV_GROUPS = 16
CONV_TAPS = 31


def tiny_dit() -> torch.nn.Module:
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


def padded_inputs() -> dict[str, torch.Tensor]:
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


def packed_inputs(padded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    rows = pack_rows(LENGTHS, CPU)
    return {
        "x": gather_rows(padded["x"].transpose(1, 2), rows),
        "mu": gather_rows(padded["mu"].transpose(1, 2), rows),
        "cond": gather_rows(padded["cond"].transpose(1, 2), rows),
        "spks": padded["spks"][rows.row_ids].unsqueeze(0),
        "t": padded["t"],
        "rows": rows,
    }


def valid(padded: torch.Tensor) -> list[torch.Tensor]:
    return [padded[index, :, :length] for index, length in enumerate(LENGTHS)]


def rope(estimator: PackedDiT, rows: PackedRows) -> tuple[torch.Tensor, torch.Tensor]:
    angles = estimator.rope_angles(rows.width)[:, rows.positions]
    return angles.cos(), angles.sin()


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(actual.double() - expected)
        / torch.linalg.vector_norm(expected)
    )


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


@pytest.mark.parametrize(
    "dtype, bits", [(torch.bfloat16, torch.int16), (torch.float32, torch.int32)]
)
def test_rotated_matches_rotate_in_place_bit_for_bit(
    dtype: torch.dtype, bits: torch.dtype
) -> None:
    torch.manual_seed(4)
    x = torch.randn(1, 41, 2 * 16, dtype=dtype)
    x[0, 0, 20] = -0.0
    angles = torch.randn(1, 41, 8)
    expected = x.clone()
    rotate_in_place(expected, angles.cos(), angles.sin())

    actual = rotated(x, angles.cos(), angles.sin())

    assert torch.equal(actual.view(bits), expected.view(bits))


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
    dit = tiny_dit()
    padded = padded_inputs()
    packed = packed_inputs(padded)
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
            rope(estimator, packed["rows"]),
        )
    out = scatter_rows(out, packed["rows"], 19).transpose(1, 2)

    for actual, reference in zip(valid(out), valid(expected), strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-9)


def test_packed_dit_shares_the_projection_weights_and_keeps_the_dit_forward() -> None:
    dit = tiny_dit()
    padded = padded_inputs()

    def native_forward() -> torch.Tensor:
        with torch.inference_mode():
            return dit(
                padded["x"],
                padded["mask"],
                padded["mu"],
                padded["t"],
                padded["spks"],
                padded["cond"],
                streaming=True,
            )

    before = native_forward()
    estimator = PackedDiT(dit, device=CPU)

    assert torch.equal(native_forward(), before)
    for block, qkv_weight, qkv_bias in zip(
        dit.transformer_blocks,
        estimator.qkv_weights,
        estimator.qkv_biases,
        strict=True,
    ):
        attention = block.attn
        for projection in (attention.to_q, attention.to_k, attention.to_v):
            assert (
                projection.weight.untyped_storage().data_ptr()
                == qkv_weight.untyped_storage().data_ptr()
            )
            assert (
                projection.bias.untyped_storage().data_ptr()
                == qkv_bias.untyped_storage().data_ptr()
            )


@pytest.mark.parametrize("streaming", [True, False])
def test_packed_solve_matches_the_padded_solve_per_row(streaming: bool) -> None:
    dit = tiny_dit()
    padded = padded_inputs()
    packed = packed_inputs(padded)
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

    estimator = PackedDiT(dit, device=CPU)
    twin_rows = pack_rows(LENGTHS * 2, CPU)

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
            estimator,
            gather_rows(noise.transpose(1, 2), packed["rows"]),
            time_span,
            packed["mu"],
            padded["spks"],
            packed["cond"],
            twin_rows,
            estimator.row_attention(
                twin_rows, streaming=streaming, dtype=torch.float64
            ),
            estimator.rope_angles(twin_rows.width),
            cfg_rate=0.7,
        )
    out = scatter_rows(out, packed["rows"], 19).transpose(1, 2)

    for actual, reference in zip(valid(out), valid(expected), strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-9)


def test_packed_compile_requires_ragged_half_precision(monkeypatch) -> None:
    estimator = PackedDiT(tiny_dit(), device=CPU)
    compile_options: list[dict[str, object]] = []

    def fake_compile(function, **kwargs):
        compile_options.append(kwargs)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    assert not estimator.compile(torch.float32)
    assert compile_options == []

    estimator.is_ragged = True
    assert not estimator.compile(torch.float32)
    assert compile_options == []
    assert estimator.compile(torch.bfloat16)
    assert estimator.is_compiled
    assert compile_options == [
        {
            "backend": "inductor",
            "fullgraph": True,
            "options": {
                "triton.autotune_pointwise": False,
                "emulate_precision_casts": True,
            },
        }
    ]


@pytest.mark.parametrize("streaming", [True, False])
def test_rows_packed_beside_an_empty_and_a_trailing_row_are_unchanged(
    streaming: bool,
) -> None:
    dit = tiny_dit()
    padded = padded_inputs()
    packed = packed_inputs(padded)
    estimator = PackedDiT(dit, device=CPU)
    torch.manual_seed(5)
    trailing = 23
    filler = {
        name: torch.randn(1, trailing, CHANNELS, dtype=torch.float64)
        for name in ("x", "mu", "cond", "spks")
    }
    rows = pack_rows((*LENGTHS[:2], 0, *LENGTHS[2:], trailing), CPU)
    together_inputs = [
        torch.cat((packed[name], filler[name]), dim=1)
        for name in ("x", "mu", "spks", "cond")
    ]

    with torch.inference_mode():
        together = estimator.forward(
            *together_inputs,
            packed["t"],
            rows,
            estimator.row_attention(rows, streaming=streaming, dtype=torch.float64),
            rope(estimator, rows),
        )
        for index, length in enumerate(LENGTHS):
            alone_rows = pack_rows((length,), CPU)
            alone = estimator.forward(
                padded["x"][index : index + 1, :, :length].transpose(1, 2),
                padded["mu"][index : index + 1, :, :length].transpose(1, 2),
                padded["spks"][index : index + 1].expand(length, -1).unsqueeze(0),
                padded["cond"][index : index + 1, :, :length].transpose(1, 2),
                padded["t"],
                alone_rows,
                estimator.row_attention(
                    alone_rows, streaming=streaming, dtype=torch.float64
                ),
                rope(estimator, alone_rows),
            )
            start = int(packed["rows"].starts_host[index])
            torch.testing.assert_close(
                together[:, start : start + length], alone, rtol=1e-9, atol=1e-9
            )


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "chunk_size, max_frames_scale", [(CHUNK, 1), (None, 1), (None, 4)]
)
def test_the_ragged_read_matches_the_padded_read(
    chunk_size: int | None, max_frames_scale: int
) -> None:
    from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported

    device = torch.device("cuda")
    if not _is_fa3_supported():
        pytest.skip("FA3 is unavailable on this device")
    torch.manual_seed(2)
    heads = 2
    head_dim = 64
    rows = pack_rows((*LENGTHS[:2], 0, *LENGTHS[2:]), device)
    query, key, value = (
        torch.randn(
            1, rows.total, heads * head_dim, device=device, dtype=torch.bfloat16
        )
        for _ in range(3)
    )
    if chunk_size is None:
        attention = WholeRowAttention(
            rows,
            heads=heads,
            head_dim=head_dim,
            max_frames=max_frames_scale * rows.width,
        )
    else:
        attention = RaggedRowAttention(
            rows, chunk_size=chunk_size, heads=heads, head_dim=head_dim
        )
    ragged = attention(query, key, value)
    padded = RowAttention(rows, chunk_size=chunk_size, heads=heads)(query, key, value)
    torch.testing.assert_close(ragged, padded, rtol=2e-2, atol=2e-2)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("group_channels", GROUP_CONV_KERNEL_CHANNELS)
@pytest.mark.parametrize("frames", [31, 95, 1000])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_group_conv_mish_is_as_close_to_float64_as_the_module(
    group_channels: int, frames: int, dtype: torch.dtype
) -> None:
    channels = CONV_GROUPS * group_channels
    torch.manual_seed(0)
    conv = (
        torch.nn.Conv1d(channels, channels, CONV_TAPS, groups=CONV_GROUPS)
        .cuda()
        .to(dtype)
    )
    with torch.no_grad():
        conv.bias.normal_(0, 0.5)
    x = torch.randn(2, frames, channels, device="cuda").to(dtype)

    with torch.inference_mode():
        fused = group_conv_mish(x, pack_group_conv_weight(conv), conv.bias)
        module = F.mish(conv(x.transpose(1, 2))).transpose(1, 2)
        float64 = F.mish(
            F.conv1d(
                x.double().transpose(1, 2),
                conv.weight.double(),
                conv.bias.double(),
                groups=CONV_GROUPS,
            )
        ).transpose(1, 2)

    assert fused.shape == (2, frames - CONV_TAPS + 1, channels) and fused.dtype == dtype
    assert relative_error(fused, float64) <= relative_error(module, float64)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_the_fused_position_embedding_is_as_close_to_float64_as_the_module() -> None:
    torch.manual_seed(0)
    original = cosyvoice_modules.CausalConvPositionEmbedding(256).cuda().eval()
    float64 = copy.deepcopy(original).double()
    original.to(torch.bfloat16)
    fused = FusedConvPositionEmbedding(original)
    x = torch.randn(2, 77, 256, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        fused_output = fused(x)
        module_output = original(x)
        reference = float64(x.double())

    assert fused_output.shape == module_output.shape
    assert relative_error(fused_output, reference) <= relative_error(
        module_output, reference
    )
