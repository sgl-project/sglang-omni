# SPDX-License-Identifier: Apache-2.0

"""GPU compile qualification for native DiT and production PackedDiT paths."""

from __future__ import annotations

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
PACKED_COMPILE_REL_L2_TOL = 2e-2


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
def test_production_packed_dit_compile_has_bounded_drift_from_eager() -> None:
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
    for module in dit.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            module.to(torch.bfloat16)
        else:
            pass
    estimator = PackedDiT(dit, device="cuda")
    assert estimator.is_ragged

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
            cases.append((rows, inputs, streaming, run(rows, inputs, streaming)))

    assert estimator.rope(cases[0][0])[0].dtype == torch.float32
    assert estimator.compile(torch.bfloat16)
    for rows, inputs, streaming, eager in cases:
        compiled = run(rows, inputs, streaming)
        torch.cuda.synchronize()
        assert compiled.shape == eager.shape
        assert compiled.dtype == eager.dtype
        assert torch.isfinite(compiled).all()
        relative_l2 = torch.linalg.vector_norm(
            compiled.float() - eager.float()
        ) / torch.linalg.vector_norm(eager.float())
        assert relative_l2 < PACKED_COMPILE_REL_L2_TOL
