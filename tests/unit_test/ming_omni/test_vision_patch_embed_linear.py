# SPDX-License-Identifier: Apache-2.0
"""Patch embedding equivalence tests."""

from __future__ import annotations

import torch
import torch.nn as nn

from sglang_omni.models.ming_omni.components.vision_encoder import _linear_patch_embed


class _TinyPatchEmbed(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        temporal_patch_size: int,
        patch_size: int,
        embed_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            bias=True,
            device=device,
            dtype=dtype,
        )


@torch.no_grad()
def test_patch_embed_linear_matches_conv3d():
    """Conv3d vs F.linear with identical weights yield equivalent outputs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    torch.manual_seed(0)
    pe = _TinyPatchEmbed(
        in_channels=3,
        temporal_patch_size=2,
        patch_size=14,
        embed_dim=16,
        device=device,
        dtype=dtype,
    )
    seq_len = 7
    patch_dim = pe.in_channels * pe.temporal_patch_size * pe.patch_size * pe.patch_size
    x = torch.randn(seq_len, patch_dim, dtype=dtype, device=device)

    conv_out = pe.proj(
        x.view(
            seq_len,
            pe.in_channels,
            pe.temporal_patch_size,
            pe.patch_size,
            pe.patch_size,
        )
    ).view(seq_len, pe.embed_dim)
    linear_out = _linear_patch_embed(pe, x)

    torch.testing.assert_close(
        linear_out,
        conv_out,
        rtol=2e-2 if dtype is torch.bfloat16 else 1e-5,
        atol=2e-2 if dtype is torch.bfloat16 else 1e-5,
    )
