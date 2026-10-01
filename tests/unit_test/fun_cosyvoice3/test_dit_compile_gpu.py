# SPDX-License-Identifier: Apache-2.0

"""GPU compile qualification for native DiT and production PackedDiT paths."""

from __future__ import annotations

import copy

import pytest
import torch
from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    DIT_INDUCTOR_OPTIONS,
    PackedDiT,
    pack_rows,
)

cosyvoice_dit = pytest.importorskip("cosyvoice.flow.DiT.dit")

pytestmark = pytest.mark.accelerator

TOL = 1e-4
COMPILED_OVER_EAGER_ERROR = 2.0


def native_inputs(batch: int, frames: int) -> tuple[torch.Tensor, ...]:
    mask = torch.ones(batch, 1, frames, device="cuda")
    mask[batch // 2 :, :, frames * 3 // 4 :] = 0
    return (
        torch.randn(batch, 80, frames, device="cuda"),
        mask,
        torch.randn(batch, 80, frames, device="cuda"),
        torch.full((batch,), 0.37, device="cuda"),
        torch.randn(batch, 80, device="cuda"),
        torch.randn(batch, 80, frames, device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compile_dit_backbone_matches_eager_beyond_the_warmup_shapes() -> None:
    torch.manual_seed(5)
    dit = (
        cosyvoice_dit.DiT(
            dim=128,
            depth=2,
            heads=2,
            dim_head=64,
            ff_mult=2,
            mel_dim=80,
            mu_dim=80,
            spk_dim=80,
            out_channels=80,
            static_chunk_size=4,
            num_decoding_left_chunks=-1,
            long_skip_connection=True,
        )
        .cuda()
        .eval()
    )
    flow = torch.nn.Module()
    flow.decoder = torch.nn.Module()
    flow.decoder.estimator = dit
    param_names = set(dict(dit.named_parameters()))
    stages.patch_chunk_mask()

    cases = []
    with torch.inference_mode():
        for streaming in (True, False):
            for batch, frames in ((2, 24), (6, 40)):
                inputs = native_inputs(batch, frames)
                cases.append((inputs, streaming, dit(*inputs, streaming=streaming)))

    stages.compile_dit_backbone(flow, warmup_mel_frames=16, warmup_steps=1)

    with torch.inference_mode():
        for inputs, streaming, eager in cases:
            compiled = dit(*inputs, streaming=streaming)
            torch.testing.assert_close(compiled, eager, rtol=TOL, atol=TOL)
    assert set(dict(dit.named_parameters())) == param_names


class ChunkMask(torch.nn.Module):
    def __init__(self, chunk_mask, static_chunk_size: int) -> None:
        super().__init__()
        self.chunk_mask = chunk_mask
        self.static_chunk_size = static_chunk_size

    def forward(self, xs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.chunk_mask(xs, masks, False, False, 0, self.static_chunk_size, -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("static_chunk_size", [50, 0])
def test_the_compiled_chunk_mask_matches_eager(static_chunk_size: int) -> None:
    stages.patch_chunk_mask()
    eager = ChunkMask(cosyvoice_dit.add_optional_chunk_mask, static_chunk_size)
    compiled = torch.compile(eager, fullgraph=True, options=dict(DIT_INDUCTOR_OPTIONS))
    generator = torch.Generator(device="cuda").manual_seed(0)
    for batch, frames in ((1, 7), (2, 128), (5, 301), (16, 1033)):
        lengths = torch.randint(
            1, frames + 1, (batch,), device="cuda", generator=generator
        )
        valid = torch.arange(frames, device="cuda")[None] < lengths[:, None]
        xs = torch.empty(batch, frames, 8, device="cuda")
        expected = eager(xs, valid[:, None].clone())
        assert torch.equal(compiled(xs, valid[:, None].clone()), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_production_packed_dit_compile_is_as_close_to_float32_as_eager() -> None:
    if not _is_fa3_supported():
        pytest.skip("FA3 is unavailable on this device")

    torch.manual_seed(3)
    dit = (
        cosyvoice_dit.DiT(
            dim=128,
            depth=2,
            heads=2,
            dim_head=64,
            ff_mult=2,
            mel_dim=8,
            mu_dim=8,
            spk_dim=8,
            out_channels=8,
            static_chunk_size=4,
            num_decoding_left_chunks=-1,
            long_skip_connection=True,
        )
        .cuda()
        .eval()
    )
    with torch.no_grad():
        for block in dit.transformer_blocks:
            block.attn_norm.linear.bias.fill_(0.5)
        dit.norm_out.linear.bias.fill_(0.5)
    for module in dit.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            module.to(torch.bfloat16)
        else:
            pass
    estimator = PackedDiT(dit, device="cuda")
    assert estimator.is_ragged
    reference = PackedDiT(copy.deepcopy(dit).float(), device="cuda")

    def run(rows, inputs, streaming: bool) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return estimator.forward(
                inputs["x"],
                inputs["mu"],
                inputs["spks"],
                inputs["cond"],
                inputs["t"],
                rows,
                estimator.row_attention(
                    rows, streaming=streaming, dtype=inputs["spks"].dtype
                ),
                estimator.rope(rows),
            )

    def run_float32(rows, inputs, streaming: bool) -> torch.Tensor:
        floats = {name: value.float() for name, value in inputs.items()}
        with torch.inference_mode():
            return reference.forward(
                floats["x"],
                floats["mu"],
                floats["spks"],
                floats["cond"],
                floats["t"],
                rows,
                reference.row_attention(rows, streaming=streaming, dtype=torch.float32),
                reference.rope(rows),
            )

    def error(actual: torch.Tensor, expected: torch.Tensor) -> float:
        return float(
            torch.linalg.vector_norm(actual.float() - expected)
            / torch.linalg.vector_norm(expected)
        )

    cases = []
    for streaming in (True, False):
        for lengths in ((11, 7), (13, 5, 9), (21,)):
            rows = pack_rows(lengths, torch.device("cuda"))
            inputs = {
                name: torch.randn(1, rows.total, 8, device="cuda")
                for name in ("x", "mu", "cond")
            }
            inputs["spks"] = torch.randn(
                1, rows.total, 8, device="cuda", dtype=torch.bfloat16
            )
            inputs["t"] = torch.full((1,), 0.37, device="cuda", dtype=torch.bfloat16)
            cases.append(
                (
                    rows,
                    inputs,
                    streaming,
                    run(rows, inputs, streaming),
                    run_float32(rows, inputs, streaming),
                )
            )

    assert estimator.rope(cases[0][0])[0].dtype == torch.float32
    assert estimator.compile(torch.bfloat16)
    for rows, inputs, streaming, eager, float32 in cases:
        compiled = run(rows, inputs, streaming)
        assert error(compiled, float32) <= COMPILED_OVER_EAGER_ERROR * error(
            eager, float32
        ), (rows.lengths, streaming)
