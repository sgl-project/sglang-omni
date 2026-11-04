# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time-conditioned DiT velocity prediction for FireRedTTS3 latent patches."""

import torch
from torch import nn

from sglang_omni.models.firered_tts3.modules import (
    Attention,
    FeedForward,
    RMSNorm,
    RotaryEmbedding,
    TimestepEmbedder,
)


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.block: torch.nn.Sequential = torch.nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.Mish(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            x = x * mask
        else:
            pass
        x = x.transpose(1, 2)
        x = self.block(x)
        x = x.transpose(1, 2)
        if mask is not None:
            x = x * mask
        else:
            pass
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class FinalLayer(nn.Module):
    """DiT final layer: time-conditioned AdaLN + linear head."""

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.adaLN_modulation: nn.Sequential = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.norm_final: nn.LayerNorm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-06
        )
        self.linear: nn.Linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(timestep_embedding).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTBlock(nn.Module):
    """DiT block: attention + depthwise conv + FFN, all AdaLN-conditioned."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1: RMSNorm = RMSNorm(hidden_size, eps=1e-06)
        self.attn: Attention = Attention(
            dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=dropout,
        )
        self.norm2: RMSNorm = RMSNorm(hidden_size, eps=1e-06)
        self.conv: ConvBlock = ConvBlock(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=3
        )
        self.norm3: RMSNorm = RMSNorm(hidden_size, eps=1e-06)
        self.mlp: FeedForward = FeedForward(
            dim=hidden_size, mult=mlp_ratio, dropout=dropout, approximate="tanh"
        )
        self.adaLN_modulation: nn.Sequential = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        mask: torch.Tensor | None,
        rope: tuple[torch.Tensor, float | torch.Tensor | None],
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_conv,
            scale_conv,
            gate_conv,
        ) = self.adaLN_modulation(timestep_embedding).chunk(9, dim=-1)
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), mask=mask, rope=rope
        )
        x = x + gate_conv * self.conv(
            modulate(self.norm2(x), shift_conv, scale_conv), mask=mask
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class DiT(nn.Module):
    """Diffusion model with a Transformer backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float = 4.0,
        depth: int = 28,
        num_heads: int = 8,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.in_proj: nn.Linear = nn.Linear(in_channels, hidden_size)
        self.t_embedder: TimestepEmbedder = TimestepEmbedder(hidden_size)
        self.rotary_embed: RotaryEmbedding = RotaryEmbedding(hidden_size // num_heads)
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=0.0)
                for _ in range(depth)
            ]
        )
        self.final_layer: FinalLayer = FinalLayer(hidden_size, self.out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Full attention DiT head."""
        t = self.t_embedder(t.view(-1)).unsqueeze(1)
        x = self.in_proj(x)
        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])
        for block in self.blocks:
            x = block(x, t, mask=None, rope=rope)
        x = self.final_layer(x, t)
        return x
