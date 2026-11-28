# SPDX-License-Identifier: Apache-2.0
"""Audio8 Fast AR predicts one set of residual codebooks per speech frame."""

from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F

from sglang_omni.models.audio8_tts.hf_config import ArkttsConfig


class CodebookSampler(Protocol):
    def __call__(self, codebook_logits: torch.Tensor, /) -> torch.Tensor: ...


class Audio8RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(hidden_size))
        self.epsilon: float = epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        float_hidden_states = hidden_states.float()
        normalized_hidden_states = float_hidden_states * torch.rsqrt(
            float_hidden_states.square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return normalized_hidden_states.to(hidden_states.dtype) * self.weight


def precompute_rotary_values(
    position_count: int, head_size: int, rope_base: float
) -> torch.Tensor:
    # note (BeiningWu): Audio8 rounds the CPU-computed rotary values to BF16.
    dimensions = torch.arange(0, head_size, 2, device="cpu", dtype=torch.float32)
    frequencies = 1.0 / (rope_base ** (dimensions / head_size))
    positions = torch.arange(position_count, device="cpu")
    angles = torch.outer(positions, frequencies)
    rotary_values = torch.polar(torch.ones_like(angles), angles)
    return torch.view_as_real(rotary_values).to(torch.bfloat16)


def apply_rope(
    hidden_states: torch.Tensor, rotary_values: torch.Tensor
) -> torch.Tensor:
    pairs = hidden_states.float().unflatten(-1, (-1, 2))
    cosine, sine = rotary_values.unbind(dim=-1)
    real, imaginary = pairs.unbind(dim=-1)
    return (
        torch.stack(
            (real * cosine - imaginary * sine, imaginary * cosine + real * sine),
            dim=-1,
        )
        .flatten(-2)
        .to(hidden_states.dtype)
    )


class Audio8FastAttention(nn.Module):
    def __init__(self, config: ArkttsConfig) -> None:
        super().__init__()
        self.attention_head_count: int = config.fast_n_head
        self.key_value_head_count: int = config.fast_n_local_heads
        self.head_size: int = config.fast_head_dim
        self.query_width: int = self.attention_head_count * self.head_size
        self.key_value_width: int = self.key_value_head_count * self.head_size
        self.query_heads_per_group: int = (
            self.attention_head_count // self.key_value_head_count
        )
        self.attention_scale: float = math.sqrt(self.head_size)
        self.wqkv: nn.Linear = nn.Linear(
            config.fast_dim,
            self.query_width + 2 * self.key_value_width,
            bias=config.fast_attention_qkv_bias,
        )
        self.wo: nn.Linear = nn.Linear(
            self.query_width, config.fast_dim, bias=config.fast_attention_o_bias
        )
        self.q_norm: Audio8RMSNorm | None
        self.k_norm: Audio8RMSNorm | None
        if config.fast_attention_qk_norm:
            self.q_norm = Audio8RMSNorm(self.head_size, config.norm_eps)
            self.k_norm = Audio8RMSNorm(self.head_size, config.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_values: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        codebook_position: int,
    ) -> torch.Tensor:
        query, key, value = self.wqkv(hidden_states).split(
            (self.query_width, self.key_value_width, self.key_value_width), dim=-1
        )
        query = query.unflatten(-1, (self.attention_head_count, self.head_size))
        key = key.unflatten(-1, (self.key_value_head_count, self.head_size))
        value = value.unflatten(-1, (self.key_value_head_count, self.head_size))
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        else:
            pass
        query = apply_rope(query, rotary_values).unsqueeze(2)
        key_cache[:, :, codebook_position].copy_(apply_rope(key, rotary_values))
        value_cache[:, :, codebook_position].copy_(value)
        attention_keys = key_cache[:, :, : codebook_position + 1].repeat_interleave(
            self.query_heads_per_group, dim=1
        )
        attention_values = value_cache[:, :, : codebook_position + 1].repeat_interleave(
            self.query_heads_per_group, dim=1
        )
        attention_scores = (
            query @ attention_keys.transpose(-1, -2) / self.attention_scale
        )
        attention_output = torch.softmax(attention_scores, dim=-1) @ attention_values
        return self.wo(attention_output.squeeze(2).flatten(1))


class Audio8FastFeedForward(nn.Module):
    def __init__(self, config: ArkttsConfig) -> None:
        super().__init__()
        self.w1: nn.Linear = nn.Linear(
            config.fast_dim, config.fast_intermediate_size, bias=False
        )
        self.w2: nn.Linear = nn.Linear(
            config.fast_intermediate_size, config.fast_dim, bias=False
        )
        self.w3: nn.Linear = nn.Linear(
            config.fast_dim, config.fast_intermediate_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class Audio8FastBlock(nn.Module):
    def __init__(self, config: ArkttsConfig) -> None:
        super().__init__()
        self.attention: Audio8FastAttention = Audio8FastAttention(config)
        self.feed_forward: Audio8FastFeedForward = Audio8FastFeedForward(config)
        self.attention_norm: Audio8RMSNorm = Audio8RMSNorm(
            config.fast_dim, config.norm_eps
        )
        self.ffn_norm: Audio8RMSNorm = Audio8RMSNorm(config.fast_dim, config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_values: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        codebook_position: int,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states),
            rotary_values,
            key_cache,
            value_cache,
            codebook_position,
        )
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class Audio8FastAR(nn.Module):
    rotary_values: torch.Tensor

    def __init__(self, config: ArkttsConfig) -> None:
        super().__init__()
        assert config.fast_n_head % config.fast_n_local_heads == 0
        self.codebook_count: int = config.num_codebooks
        self.semantic_begin_id: int = config.semantic_begin_id
        self.key_value_head_count: int = config.fast_n_local_heads
        self.head_size: int = config.fast_head_dim
        self.fast_project_in: nn.Linear | nn.Identity
        if config.dim == config.fast_dim:
            self.fast_project_in = nn.Identity()
        else:
            self.fast_project_in = nn.Linear(config.dim, config.fast_dim)
        self.fast_embeddings: nn.Embedding = nn.Embedding(
            config.codebook_size, config.fast_dim
        )
        self.fast_layers: nn.ModuleList = nn.ModuleList(
            Audio8FastBlock(config) for _ in range(config.n_fast_layer)
        )
        self.fast_norm: Audio8RMSNorm = Audio8RMSNorm(config.fast_dim, config.norm_eps)
        self.fast_output: nn.Linear = nn.Linear(
            config.fast_dim, config.codebook_size, bias=False
        )
        rotary_values = precompute_rotary_values(
            self.codebook_count, self.head_size, config.rope_base
        )
        self.register_buffer(
            "rotary_values",
            rotary_values.to(self.fast_embeddings.weight.device),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        semantic_token_ids: torch.Tensor,
        sample_codebook: CodebookSampler,
    ) -> torch.Tensor:
        key_value_cache_shape = (
            len(self.fast_layers),
            hidden_states.shape[0],
            self.key_value_head_count,
            self.codebook_count,
            self.head_size,
        )
        key_cache = hidden_states.new_empty(key_value_cache_shape)
        value_cache = hidden_states.new_empty(key_value_cache_shape)
        generated_codebook_ids = [semantic_token_ids - self.semantic_begin_id]
        for codebook_position, rotary_values in enumerate(self.rotary_values):
            codebook_hidden_states = (
                self.fast_project_in(hidden_states)
                if codebook_position == 0
                else self.fast_embeddings(generated_codebook_ids[-1])
            )
            for layer, layer_key_cache, layer_value_cache in zip(
                self.fast_layers, key_cache, value_cache, strict=True
            ):
                codebook_hidden_states = layer(
                    codebook_hidden_states,
                    rotary_values,
                    layer_key_cache,
                    layer_value_cache,
                    codebook_position,
                )
            if codebook_position > 0:
                codebook_logits = self.fast_output(
                    self.fast_norm(codebook_hidden_states)
                )
                generated_codebook_ids.append(sample_codebook(codebook_logits))
            else:
                pass
        return torch.stack(generated_codebook_ids, dim=-1)
