# SPDX-License-Identifier: Apache-2.0
# Ported from OpenBMB/VoxCPM (Apache-2.0), src/voxcpm/modules/locdit/local_dit_v2.py.
"""VoxCPM2 local DiT: the velocity estimator the flow-matching solver calls."""

from __future__ import annotations

import math

import torch
from torch import nn

from sglang_omni.models.voxcpm2.components.minicpm import MiniCPM4Config, MiniCPMModel


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalPosEmb requires an even dim, got {dim}")
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000) -> torch.Tensor:
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        step = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=x.dtype, device=x.device) * -step)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class VoxCPMLocDiT(nn.Module):
    """Predicts the flow velocity for one latent patch."""

    def __init__(self, config: MiniCPM4Config, in_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.config = config

        hidden = config.hidden_size
        self.in_proj = nn.Linear(in_channels, hidden, bias=True)
        self.cond_proj = nn.Linear(in_channels, hidden, bias=True)
        self.out_proj = nn.Linear(hidden, self.out_channels, bias=True)

        self.time_embeddings = SinusoidalPosEmb(hidden)
        self.time_mlp = TimestepEmbedding(in_channels=hidden, time_embed_dim=hidden)
        self.delta_time_mlp = TimestepEmbedding(
            in_channels=hidden, time_embed_dim=hidden
        )

        self.decoder = MiniCPMModel(config)

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """``x``/``cond`` are ``[N, C, T]``, ``mu`` is ``[N, C]``, ``t``/``dt`` are ``[N]``."""
        x = self.in_proj(x.transpose(1, 2).contiguous())
        cond = self.cond_proj(cond.transpose(1, 2).contiguous())
        prefix = cond.size(1)

        t_emb = self.time_mlp(self.time_embeddings(t).to(x.dtype))
        dt_emb = self.delta_time_mlp(self.time_embeddings(dt).to(x.dtype))
        t_emb = t_emb + dt_emb

        mu = mu.view(x.size(0), -1, x.size(-1))
        hidden = torch.cat([mu, t_emb.unsqueeze(1), cond, x], dim=1)

        hidden = self.decoder(hidden)
        hidden = hidden[:, prefix + mu.size(1) + 1 :, :]
        return self.out_proj(hidden).transpose(1, 2).contiguous()


__all__ = ["VoxCPMLocDiT"]
