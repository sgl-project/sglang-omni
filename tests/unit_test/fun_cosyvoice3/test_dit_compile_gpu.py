# SPDX-License-Identifier: Apache-2.0

"""GPU smoke test: torch.compile(dynamic=True) on a tiny DiT-shaped module.

Verifies eager-equivalent output across two sequence lengths without the
CosyVoice checkout or checkpoint.
"""

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
    pass

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
