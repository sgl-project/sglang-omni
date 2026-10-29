# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared acoustic layers, rotary embeddings, and latent patch encoding for FireRedTTS3."""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import arange, cat, nn, stack
from torch.amp import autocast
from torch.nn import Module


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim: int = dim

    def forward(self, x: torch.Tensor, scale: float = 1000) -> torch.Tensor:
        device = x.device
        half_embedding_size = self.dim // 2
        position_embedding = math.log(10000) / (half_embedding_size - 1)
        position_embedding = torch.exp(
            torch.arange(half_embedding_size, device=device).float()
            * -position_embedding
        )
        position_embedding = scale * x.unsqueeze(1) * position_embedding.unsqueeze(0)
        position_embedding = torch.cat(
            (position_embedding.sin(), position_embedding.cos()), dim=-1
        )
        return position_embedding


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256) -> None:
        super().__init__()
        self.time_embed: SinusPositionEmbedding = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep_hidden_states = self.time_embed(timestep)
        timestep_hidden_states = timestep_hidden_states.to(timestep.dtype)
        timestep_embedding = self.time_mlp(timestep_hidden_states)
        return timestep_embedding


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        else:
            pass
        return F.rms_norm(
            x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 4,
        dropout: float = 0.0,
        approximate: Literal["none", "tanh"] = "none",
    ) -> None:
        super().__init__()
        projection_size = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        activation = nn.GELU(approximate=approximate)
        input_projection = nn.Sequential(nn.Linear(dim, projection_size), activation)
        self.ff: nn.Sequential = nn.Sequential(
            input_projection, nn.Dropout(dropout), nn.Linear(projection_size, dim_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.dim: int = dim
        self.heads: int = heads
        self.inner_dim: int = dim_head * heads
        self.dropout: float = dropout
        self.to_q: nn.Linear = nn.Linear(dim, self.inner_dim)
        self.to_k: nn.Linear = nn.Linear(dim, self.inner_dim)
        self.to_v: nn.Linear = nn.Linear(dim, self.inner_dim)
        self.to_out: nn.ModuleList = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

    def project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.to_q(x), self.to_k(x), self.to_v(x))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, float | torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        query, key, value = self.project_qkv(x)
        projection_size = key.shape[-1]
        head_size = projection_size // self.heads
        query = query.view(batch_size, -1, self.heads, head_size).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_size).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_size).transpose(1, 2)
        if rope is not None:
            position_frequencies, position_scale = rope
            query_position_scale, key_position_scale = (
                (position_scale, position_scale ** (-1.0))
                if position_scale is not None
                else (1.0, 1.0)
            )
            query = apply_rotary_pos_emb(
                query, position_frequencies, query_position_scale
            )
            key = apply_rotary_pos_emb(key, position_frequencies, key_position_scale)
        else:
            pass
        x = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_size)
        x = x.to(query.dtype)
        x = self.to_out[0](x)
        x = self.to_out[1](x)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
        else:
            pass
        return x


class PackedAttention(Attention):
    """Reuse attention's post-projection operations without changing its source."""

    def __init__(self, source: Attention) -> None:
        torch.nn.Module.__init__(self)
        self.dim, self.inner_dim = source.dim, source.inner_dim
        self.heads, self.dropout = source.heads, source.dropout
        self.training = source.training
        self.to_q, self.to_k, self.to_v = source.to_q, source.to_k, source.to_v
        self.to_out = source.to_out
        self.to_qkv = torch.nn.Linear(self.dim, 3 * self.inner_dim, device="meta")
        projections = (self.to_q, self.to_k, self.to_v)
        self.to_qkv.weight = torch.nn.Parameter(
            torch.cat([layer.weight.detach() for layer in projections]),
            requires_grad=False,
        )
        self.to_qkv.bias = torch.nn.Parameter(
            torch.cat([layer.bias.detach() for layer in projections]),
            requires_grad=False,
        )
        self.to_qkv.training = source.training

    def project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.to_qkv(x).chunk(3, dim=-1)


class DiTBlock(nn.Module):
    """Patch-encoder block: attention + FFN only (no conv, no AdaLN)."""

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
        self.mlp: FeedForward = FeedForward(
            dim=hidden_size, mult=mlp_ratio, dropout=dropout, approximate="tanh"
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        rope: tuple[torch.Tensor, float | torch.Tensor | None],
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """Patch-encoder output layer: RMSNorm + linear (no AdaLN)."""

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final: RMSNorm = RMSNorm(hidden_size, eps=1e-06)
        self.linear: nn.Linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class PatchEncoder(torch.nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        patch_size: int = 4,
        hidden_size: int = 1024,
        mlp_ratio: int = 3,
        depth: int = 8,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.in_dim: int = in_dim
        self.patch_size: int = patch_size
        self.hidden_size: int = hidden_size
        self.out_dim: int = out_dim
        self.cls_tok: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros(1, 1, hidden_size)
        )
        self.rotary_embed: RotaryEmbedding = RotaryEmbedding(hidden_size // num_heads)
        self.blocks: torch.nn.ModuleList = torch.nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.in_proj: torch.nn.Linear | torch.nn.Identity = (
            torch.nn.Linear(in_dim, hidden_size)
            if in_dim != hidden_size
            else torch.nn.Identity()
        )
        self.out_proj: FinalLayer = FinalLayer(hidden_size, out_dim)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Patch encoder aggregating patch_size latents into one."""
        assert (
            inputs_embeds.shape[1] % self.patch_size == 0
        ), f"inputs_embeds.shape={inputs_embeds.shape} patch_size={self.patch_size}"
        inputs_embeds = self.in_proj(inputs_embeds)
        hidden_states = inputs_embeds.reshape(-1, self.patch_size, self.hidden_size)
        cls_tok = self.cls_tok.expand(hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat([cls_tok, hidden_states], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(hidden_states.shape[1])
        for block in self.blocks:
            hidden_states = block(hidden_states, None, rope)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = hidden_states[:, 0]
        hidden_states = hidden_states.unsqueeze(0)
        return hidden_states


class RotaryEmbedding(Module):
    inv_freq: torch.Tensor

    def __init__(
        self,
        dim: int,
        interpolation_factor: float = 1.0,
        base: float = 10000,
        base_rescale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))
        inv_freq = 1.0 / base ** (arange(0, dim, 2).float() / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        assert interpolation_factor >= 1.0
        self.interpolation_factor: float = interpolation_factor

    def forward_from_seq_len(
        self, position_count: int
    ) -> tuple[torch.Tensor, float | torch.Tensor | None]:
        device = self.inv_freq.device
        t = arange(position_count, device=device)
        return self.forward(t)

    @autocast("cuda", enabled=False)
    def forward(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, float | torch.Tensor | None]:
        if t.ndim == 1:
            t = rearrange(t, "n -> 1 n")
        else:
            pass
        position_frequencies = (
            torch.einsum("b i , j -> b i j", t.type_as(self.inv_freq), self.inv_freq)
            / self.interpolation_factor
        )
        position_frequencies = stack(
            (position_frequencies, position_frequencies), dim=-1
        )
        position_frequencies = rearrange(position_frequencies, "... d r -> ... (d r)")
        return (position_frequencies, 1.0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_pos_emb(
    t: torch.Tensor, position_frequencies: torch.Tensor, scale: float | torch.Tensor = 1
) -> torch.Tensor:
    rotary_size, position_count, original_dtype = (
        position_frequencies.shape[-1],
        t.shape[-2],
        t.dtype,
    )
    position_frequencies = position_frequencies[:, -position_count:, :]
    scale = scale[:, -position_count:, :] if isinstance(scale, torch.Tensor) else scale
    if t.ndim == 4 and position_frequencies.ndim == 3:
        position_frequencies = rearrange(position_frequencies, "b n d -> b 1 n d")
    else:
        pass
    t, t_unrotated = (t[..., :rotary_size], t[..., rotary_size:])
    t = (
        t * position_frequencies.cos() * scale
        + rotate_half(t) * position_frequencies.sin() * scale
    )
    out = cat((t, t_unrotated), dim=-1)
    return out.type(original_dtype)
