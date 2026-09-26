# SPDX-License-Identifier: Apache-2.0

"""GPU compile qualification for native DiT and production PackedDiT paths."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.accelerator

TOL = 1e-4


class TinyDiT(torch.nn.Module):
    """Minimal stand-in for cosyvoice.flow.DiT.dit.DiT."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        del mu, t, spks, cond, streaming
        # The real DiT transposes to [batch, time, channels] first.
        x = x.transpose(1, 2)
        # Mirrors the never-taken .item() guard in add_optional_chunk_mask.
        if mask.sum().item() < 0:
            x = x * 0
        out = self.norm(self.proj(x))
        return out.transpose(1, 2)


def make_inputs(estimator, t: int) -> tuple[torch.Tensor, ...]:
    device = next(estimator.parameters()).device
    return (
        torch.randn(2, 16, t, device=device),
        torch.ones(2, 1, t, device=device),
        torch.randn(2, 16, t, device=device),
        torch.zeros(2, device=device),
        torch.randn(2, 16, device=device),
        torch.randn(2, 16, t, device=device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compile_dit_backbone_dynamic_shapes_match_eager() -> None:
<<<<<<< HEAD
    pass

=======
>>>>>>> upstream/main
    estimator = TinyDiT().cuda().eval()
    original_forward = estimator.forward
    param_names = set(dict(estimator.named_parameters()))

    torch._inductor.config.fx_graph_cache = (
        True  # noqa: leading-underscore  # production name
    )
    if hasattr(
        torch._dynamo.config, "cache_size_limit"
    ):  # noqa: leading-underscore  # production name
        torch._dynamo.config.cache_size_limit = (
            1024  # noqa: leading-underscore  # production name
        )
    if hasattr(
        torch._dynamo.config, "accumulated_cache_size_limit"
    ):  # noqa: leading-underscore  # production name
        torch._dynamo.config.accumulated_cache_size_limit = (
            1024  # noqa: leading-underscore  # production name
        )
    estimator.forward = torch.compile(estimator.forward, dynamic=True)

    with torch.no_grad():
        # Two lengths on the same inputs prove the symbolic-length graph is reused.
        for t in (32, 48):
            x, mask, mu, timestep, spks, cond = make_inputs(estimator, t)
            compiled = estimator(x, mask, mu, timestep, spks, cond, streaming=False)
            eager = original_forward(x, mask, mu, timestep, spks, cond, streaming=False)
            assert torch.allclose(compiled, eager, atol=TOL, rtol=TOL)

    # Bound-method compile keeps parameter names stable (no _orig_mod prefix).
    assert set(dict(estimator.named_parameters())) == param_names


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("streaming", [True, False])
def test_production_packed_dit_compile_matches_eager(streaming: bool) -> None:
    cosyvoice_dit = pytest.importorskip("cosyvoice.flow.DiT.dit")
    from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported

    from sglang_omni.models.fun_cosyvoice3.packed_dit import PackedDiT, pack_rows

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
        .to(dtype=torch.bfloat16)
        .eval()
    )
    estimator = PackedDiT(dit, device="cuda")
    assert estimator.is_ragged
    assert estimator.compile(torch.bfloat16)

    for lengths in ((11, 7), (13, 5, 9)):
        rows = pack_rows(lengths, torch.device("cuda"))
        inputs = {
            "x": torch.randn(1, rows.total, 8, device="cuda", dtype=torch.bfloat16),
            "mu": torch.randn(1, rows.total, 8, device="cuda", dtype=torch.bfloat16),
            "spks": torch.randn(1, rows.total, 8, device="cuda", dtype=torch.bfloat16),
            "cond": torch.randn(1, rows.total, 8, device="cuda", dtype=torch.bfloat16),
            "t": torch.full((1,), 0.37, device="cuda", dtype=torch.bfloat16),
        }

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            eager_attention = estimator.row_attention(
                rows, streaming=streaming, dtype=torch.bfloat16
            )
            eager = estimator.forward(
                inputs["x"],
                inputs["mu"],
                inputs["spks"],
                inputs["cond"],
                inputs["t"],
                rows,
                eager_attention,
            )
            compiled_attention = estimator.row_attention(
                rows, streaming=streaming, dtype=torch.bfloat16
            )
            compiled = estimator.forward_for_mode(
                streaming,
                attention=compiled_attention,
            )(
                inputs["x"],
                inputs["mu"],
                inputs["spks"],
                inputs["cond"],
                inputs["t"],
                rows,
                compiled_attention,
            )
        torch.cuda.synchronize()
        assert torch.equal(compiled, eager)
