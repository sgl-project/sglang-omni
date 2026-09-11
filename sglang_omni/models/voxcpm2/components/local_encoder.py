# SPDX-License-Identifier: Apache-2.0
# Ported from OpenBMB/VoxCPM (Apache-2.0), src/voxcpm/modules/locenc/local_encoder.py.
"""VoxCPM2 local encoder: one latent patch to one AR-step embedding."""

from __future__ import annotations

import torch
from torch import nn

from sglang_omni.models.voxcpm2.components.minicpm import MiniCPM4Config, MiniCPMModel


class VoxCPMLocEnc(nn.Module):
    def __init__(self, config: MiniCPM4Config, input_dim: int = 64):
        super().__init__()
        self.config = config
        self.special_token = nn.Parameter(torch.randn(1, 1, 1, config.hidden_size))
        self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)
        self.encoder = MiniCPMModel(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, T, P, D]`` latent patches to ``[B, T, hidden]`` embeddings."""
        batch, steps = x.shape[0], x.shape[1]
        x = self.in_proj(x)
        special = self.special_token.expand(batch, steps, 1, -1)
        x = torch.cat([special, x], dim=2)
        x = x.reshape(batch * steps, x.shape[2], x.shape[3])
        outputs = self.encoder(x)
        return outputs[:, 0, :].reshape(batch, steps, -1)


__all__ = ["VoxCPMLocEnc"]
