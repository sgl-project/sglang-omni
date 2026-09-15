# SPDX-License-Identifier: Apache-2.0
# Ported from OpenBMB/VoxCPM (Apache-2.0), src/voxcpm/modules/layers/ and model/voxcpm2.py.
"""The projections and the stop head that sit between VoxCPM2's stacks."""

from __future__ import annotations

import torch
from torch import nn


class ScalarQuantizationLayer(nn.Module):
    """Bottlenecks a hidden state through a tanh-bounded, rounded latent."""

    def __init__(self, in_dim: int, out_dim: int, latent_dim: int = 64, scale: int = 9):
        super().__init__()
        self.scale = scale
        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, out_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.in_proj(hidden))
        hidden = torch.round(hidden * self.scale) / self.scale
        return self.out_proj(hidden)


class VoxCPM2Projections(nn.Module):
    """Everything the decode loop runs between the two AR stacks and the DiT."""

    def __init__(
        self,
        *,
        lm_hidden_size: int,
        encoder_hidden_size: int,
        dit_hidden_size: int,
        quantization_latent_dim: int,
        quantization_scale: int,
    ) -> None:
        super().__init__()
        self.fsq_layer = ScalarQuantizationLayer(
            lm_hidden_size,
            lm_hidden_size,
            quantization_latent_dim,
            quantization_scale,
        )
        self.enc_to_lm_proj = nn.Linear(encoder_hidden_size, lm_hidden_size)
        self.lm_to_dit_proj = nn.Linear(lm_hidden_size, dit_hidden_size)
        self.res_to_dit_proj = nn.Linear(lm_hidden_size, dit_hidden_size)
        self.fusion_concat_proj = nn.Linear(lm_hidden_size * 2, lm_hidden_size)

        self.stop_proj = nn.Linear(lm_hidden_size, lm_hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(lm_hidden_size, 2, bias=False)

    def quantize(self, lm_hidden: torch.Tensor) -> torch.Tensor:
        return self.fsq_layer(lm_hidden)

    def fuse(self, lm_hidden: torch.Tensor, patch_embed: torch.Tensor) -> torch.Tensor:
        """Build the residual stack's input from the base hidden and the patch."""
        return self.fusion_concat_proj(torch.cat((lm_hidden, patch_embed), dim=-1))

    def to_dit(
        self, lm_hidden: torch.Tensor, residual_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate both stacks' contributions into the DiT conditioning."""
        return torch.cat(
            (self.lm_to_dit_proj(lm_hidden), self.res_to_dit_proj(residual_hidden)),
            dim=-1,
        )

    def stop_logits(self, lm_hidden: torch.Tensor) -> torch.Tensor:
        return self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))


__all__ = ["ScalarQuantizationLayer", "VoxCPM2Projections"]
