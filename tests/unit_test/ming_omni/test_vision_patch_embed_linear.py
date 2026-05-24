# SPDX-License-Identifier: Apache-2.0
# (wenyao) Asserts Conv3d vs F.linear patch_embed match within bf16 precision
# at the substitution boundary. End-to-end correctness is verified separately
# via MMMU regression: 27 downstream blocks in bf16 amplify input noise, so
# end-to-end bit equivalence is the wrong guarantee — equal answer quality is.
from __future__ import annotations
import os
import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


_MODEL_PATH = os.environ.get(
    "MING_MODEL_PATH", "/data/repo/vllm-omni/Ming-flash-omni-2.0"
)


def _build_encoder():
    from sglang_omni.models.ming_omni.components.image_encoder import (
        MingImageEncoder,
    )
    enc = MingImageEncoder(
        model_path=_MODEL_PATH, device="cuda", dtype="bfloat16"
    )
    return enc.visual


@torch.no_grad()
def test_patch_embed_linear_matches_conv3d():
    """Conv3d vs F.linear with identical weights yield bf16-equivalent outputs."""
    import torch.nn.functional as F

    enc = _build_encoder()
    pe = enc.patch_embed
    seq_len = 840
    patch_dim = (
        pe.in_channels * pe.temporal_patch_size * pe.patch_size * pe.patch_size
    )
    x = torch.randn(
        seq_len, patch_dim, dtype=torch.bfloat16, device="cuda"
    )

    conv_out = pe(x).view(seq_len, pe.embed_dim)
    linear_out = F.linear(
        x.view(seq_len, -1),
        pe.proj.weight.view(pe.embed_dim, -1),
        pe.proj.bias,
    )

    assert conv_out.shape == linear_out.shape
    assert not torch.isnan(linear_out).any()
    max_diff = (conv_out.float() - linear_out.float()).abs().max().item()
    # (wenyao) bf16 has ~0.78% relative precision; 2× slack covers
    # cuDNN-conv vs cuBLAS-GEMM accumulation-order differences.
    tol = 0.02 * conv_out.abs().max().item()
    assert max_diff <= tol, (
        f"patch_embed Conv3d vs F.linear diverged beyond bf16 precision: "
        f"max_abs_diff={max_diff:.4e}, tol={tol:.4e} "
        f"(output max_abs={conv_out.abs().max().item():.4e})"
    )
