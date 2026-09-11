# SPDX-License-Identifier: Apache-2.0
# Ported from OpenBMB/VoxCPM (Apache-2.0), src/voxcpm/modules/minicpm4/.
"""MiniCPM transformer for VoxCPM2's local encoder and local DiT.

Both run full attention over a handful of positions and are re-run from
scratch every AR step, so this port keeps only the cache-free forward pass.
The AR stacks that do need a KV cache run on SGLang instead.
"""

from __future__ import annotations

import math

import torch
from pydantic import BaseModel
from torch import nn


class RopeScalingConfig(BaseModel):
    type: str
    long_factor: list[float]
    short_factor: list[float]
    original_max_position_embeddings: int


class MiniCPM4Config(BaseModel):
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: RopeScalingConfig
    scale_depth: float
    use_mup: bool = True
    kv_channels: int | None = None
    no_rope: bool = False

    @property
    def head_dim(self) -> int:
        return self.kv_channels or self.hidden_size // self.num_attention_heads


def rms_layernorm(
    hidden: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class MiniCPMRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class MiniCPMLongRoPE(nn.Module):
    def __init__(self, config: MiniCPM4Config):
        super().__init__()
        self.dim = config.head_dim
        scaling = config.rope_scaling
        self.short_factor = scaling.short_factor
        self.long_factor = scaling.long_factor
        original = scaling.original_max_position_embeddings

        scale = config.max_position_embeddings / original
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(original))
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._set_cos_sin_cache(
            config.max_position_embeddings, original, self.inv_freq.device
        )

    def _set_cos_sin_cache(
        self, seq_len: int, original: int, device: torch.device
    ) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        factors = self.long_factor if seq_len > original else self.short_factor
        ext_factors = torch.tensor(factors, dtype=torch.float32, device=device)
        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(torch.float32),
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(torch.float32) * self.scaling_factor
        self.sin_cached = emb.sin().to(torch.float32) * self.scaling_factor

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


class MiniCPMAttention(nn.Module):
    def __init__(self, config: MiniCPM4Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value = value.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_emb is not None:
            cos, sin = position_emb
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            is_causal=False,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class MiniCPMMLP(nn.Module):
    def __init__(self, config: MiniCPM4Config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiniCPMDecoderLayer(nn.Module):
    def __init__(self, config: MiniCPM4Config):
        super().__init__()
        self.self_attn = MiniCPMAttention(config)
        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = MiniCPMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MiniCPMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.use_mup = config.use_mup
        self.residual_scale = config.scale_depth / math.sqrt(config.num_hidden_layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        scale = self.residual_scale if self.use_mup else 1.0

        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), position_emb
        )
        hidden_states = residual + hidden_states * scale

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states * scale


class MiniCPMModel(nn.Module):
    """Cache-free bidirectional MiniCPM stack over embeddings."""

    def __init__(self, config: MiniCPM4Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [MiniCPMDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope_emb = None if config.no_rope else MiniCPMLongRoPE(config)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        position_emb = None
        if self.rope_emb is not None:
            position_ids = torch.arange(
                inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device
            )
            position_emb = self.rope_emb(position_ids)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_emb)
        return self.norm(hidden_states)


__all__ = [
    "MiniCPM4Config",
    "MiniCPMModel",
    "RopeScalingConfig",
]
