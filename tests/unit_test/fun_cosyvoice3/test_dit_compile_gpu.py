# SPDX-License-Identifier: Apache-2.0

"""GPU smoke test: torch.compile(dynamic=True) on a tiny DiT-shaped module.

Verifies eager-equivalent output across two sequence lengths without the
CosyVoice checkout or checkpoint.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.accelerator

_TOL = 1e-4


class _TinyDiT(torch.nn.Module):
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


def _make_inputs(estimator, t: int) -> tuple[torch.Tensor, ...]:
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
    import sglang_omni.models.fun_cosyvoice3.stages as stages

    estimator = _TinyDiT().cuda().eval()
    original_forward = estimator.forward
    param_names = set(dict(estimator.named_parameters()))

    stages._configure_dit_torch_compile()
    estimator.forward = torch.compile(estimator.forward, dynamic=True)

    with torch.no_grad():
        # Two lengths on the same inputs prove the symbolic-length graph is reused.
        for t in (32, 48):
            x, mask, mu, timestep, spks, cond = _make_inputs(estimator, t)
            compiled = estimator(x, mask, mu, timestep, spks, cond, streaming=False)
            eager = original_forward(x, mask, mu, timestep, spks, cond, streaming=False)
            assert torch.allclose(compiled, eager, atol=_TOL, rtol=_TOL)

    # Bound-method compile keeps parameter names stable (no _orig_mod prefix).
    assert set(dict(estimator.named_parameters())) == param_names
