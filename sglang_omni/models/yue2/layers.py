# SPDX-License-Identifier: Apache-2.0
"""YuE2 attention and modality-specific transformer layers."""

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch import nn
from torch.nn import functional as F

if torch.version.cuda is not None:
    import sgl_kernel
else:
    sgl_kernel = None


class Yue2ModelConfig(BaseModel):
    latent_type: Literal["vae"] = "vae"
    hidden_size: int = Field(default=2048, gt=0)
    num_hidden_layers: int = Field(default=28, gt=0)
    num_attention_heads: int = Field(default=16, gt=0)
    num_key_value_heads: int = Field(default=8, gt=0)
    head_dim: int = Field(default=128, gt=0)
    intermediate_size: int = Field(default=6144, gt=0)
    vocab_size: int = Field(default=184704, gt=0)
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 24576
    latent_dim: int = 64
    max_latent_frames: int = 24576
    timestep_shift: float = 1.0


@dataclass(kw_only=True)
class TokenCache:
    capacity: int
    length: int = 0
    keys: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)

    def append(
        self, layer_index: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        end = self.length + key.shape[2]
        assert end <= self.capacity
        if layer_index == len(self.keys):
            shape = (*key.shape[:2], self.capacity, key.shape[-1])
            self.keys.append(key.new_empty(shape))
            self.values.append(value.new_empty(shape))
        else:
            pass
        self.keys[layer_index][:, :, self.length : end].copy_(key)
        self.values[layer_index][:, :, self.length : end].copy_(value)
        return self.keys[layer_index][:, :, :end], self.values[layer_index][:, :, :end]


class RMSNorm(nn.Module):
    def __init__(self, channels: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.epsilon = epsilon
        self.fused_kernels = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fused_kernels:
            return sgl_kernel.rmsnorm(
                hidden_states.reshape(-1, hidden_states.shape[-1]),
                self.weight,
                self.epsilon,
            ).view_as(hidden_states)
        else:
            pass
        scale = torch.rsqrt(
            hidden_states.float().square().mean(-1, keepdim=True) + self.epsilon
        )
        return hidden_states * scale.to(hidden_states.dtype) * self.weight

    def add(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fused_kernels:
            sgl_kernel.fused_add_rmsnorm(
                hidden_states.view(-1, hidden_states.shape[-1]),
                residual.view(-1, residual.shape[-1]),
                self.weight,
                self.epsilon,
            )
            return hidden_states, residual
        else:
            residual = residual + hidden_states
            return self(residual), residual


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = 1.0 / self.base ** (
            torch.arange(
                0, self.head_dim, 2, device=positions.device, dtype=torch.float32
            )
            / self.head_dim
        )
        angles = positions.float().unsqueeze(-1) * frequencies
        return angles.cos(), angles.sin()


def apply_rotary(
    hidden_states: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    left, right = hidden_states.chunk(2, dim=-1)
    cosine = cosine.unsqueeze(2).to(hidden_states.dtype)
    sine = sine.unsqueeze(2).to(hidden_states.dtype)
    return torch.cat(
        (left * cosine - right * sine, right * cosine + left * sine), dim=-1
    )


class Attention(nn.Module):
    def __init__(self, config: Yue2ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.fused_kernels = False
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.register_buffer("fused_qkv", None, persistent=False)

    def fuse_projections(self) -> None:
        self.fused_qkv = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight)
        )
        sizes = [
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        ]
        for projection, weight in zip(
            (self.q_proj, self.k_proj, self.v_proj), self.fused_qkv.split(sizes)
        ):
            projection.weight = nn.Parameter(weight, requires_grad=False)

    def project_qkv(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, token_count = hidden_states.shape[:2]
        if self.fused_qkv is None:
            query, key, value = (
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            )
        else:
            projected = F.linear(hidden_states, self.fused_qkv)
            if self.fused_kernels:
                sgl_kernel.fused_qk_norm_rope(
                    projected.flatten(0, 1),
                    self.num_heads,
                    self.num_kv_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.q_norm.epsilon,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    self.rope_theta,
                    True,
                    positions.expand(batch_size, token_count).reshape(-1),
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                )
            else:
                pass
            query, key, value = projected.split(
                [
                    self.num_heads * self.head_dim,
                    self.num_kv_heads * self.head_dim,
                    self.num_kv_heads * self.head_dim,
                ],
                dim=-1,
            )
        query = query.view(batch_size, token_count, self.num_heads, self.head_dim)
        key = key.view(batch_size, token_count, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, token_count, self.num_kv_heads, self.head_dim)
        if self.fused_kernels:
            pass
        else:
            query = apply_rotary(self.q_norm(query), cosine, sine)
            key = apply_rotary(self.k_norm(key), cosine, sine)
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=causal,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        return self.o_proj(attended.transpose(1, 2).flatten(2))


class MLP(nn.Module):
    def __init__(self, config: Yue2ModelConfig) -> None:
        super().__init__()
        self.fused_kernels = False
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.register_buffer("fused_gate_up", None, persistent=False)

    def fuse_projections(self) -> None:
        self.fused_gate_up = torch.cat((self.gate_proj.weight, self.up_proj.weight))
        gate, up = self.fused_gate_up.chunk(2)
        self.gate_proj.weight = nn.Parameter(gate, requires_grad=False)
        self.up_proj.weight = nn.Parameter(up, requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fused_kernels:
            assert self.fused_gate_up is not None
            return self.down_proj(
                sgl_kernel.silu_and_mul(F.linear(hidden_states, self.fused_gate_up))
            )
        elif self.fused_gate_up is None:
            gate, up = self.gate_proj(hidden_states), self.up_proj(hidden_states)
        else:
            gate, up = F.linear(hidden_states, self.fused_gate_up).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, config: Yue2ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)
        self.nar_input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.nar_self_attn = Attention(config)
        self.nar_pre_mlp_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.nar_mlp = MLP(config)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(128, device=timestep.device, dtype=torch.float32)
            / 128
        )
        angles = timestep.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class AudioPositionEmbedding(nn.Module):
    def __init__(self, frame_count: int, hidden_size: int) -> None:
        super().__init__()
        embedding = torch.zeros(frame_count, hidden_size)
        positions = torch.arange(frame_count, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_size)
        )
        embedding[:, 0::2] = (positions * frequencies).sin()
        embedding[:, 1::2] = (positions * frequencies).cos()
        self.register_buffer("pe", embedding)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pe[positions]
