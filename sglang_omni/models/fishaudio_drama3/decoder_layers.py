# SPDX-License-Identifier: Apache-2.0
"""Attention, rotary embeddings and dense layers for the Drama 3 codebook tower."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from torch import nn

from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniAudioDecoderConfig
from sglang_omni.models.fishaudio_drama3.tower_tiles import (
    NormWeights,
    TowerLayerWeights,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.interface import JointRopeInplaceKernel


class DecoderAttention(Protocol):
    def __call__(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Write this call's key/value rows to tower layer layer_index (0-based, shared
        layers first), then attend.

        query [rows, heads * head_dim] and key/value [rows, kv_heads * head_dim];
        query and key arrive normed and rotated. Return causal GQA attention (scale
        1 / sqrt(head_dim)) over each row's own request, [rows, heads * head_dim],
        before the output gate. The callee owns request/position metadata, which
        must match the tower's positions.
        """


@runtime_checkable
class PagedDecoderAttention(DecoderAttention, Protocol):
    def forward_mode(self) -> ForwardMode: ...

    def cache_slots(self) -> torch.Tensor: ...

    def key_buffer(self, layer_index: int) -> torch.Tensor: ...

    def value_buffer(self, layer_index: int) -> torch.Tensor: ...

    def attend(self, layer_index: int, query: torch.Tensor) -> torch.Tensor: ...


def norm_weights(layer_norm: GemmaRMSNorm) -> NormWeights:
    return NormWeights(weight=layer_norm.weight, eps=layer_norm.variance_epsilon)


class InterleavedRotaryEmbedding(nn.Module):
    def __init__(self, config: FishOmniAudioDecoderConfig, max_position: int) -> None:
        super().__init__()
        self.head_dim: int = config.head_dim
        self.rotary_dim: int = config.rotary_dim
        inverse_frequencies = 1.0 / config.rope_base ** (
            torch.arange(0, config.rotary_dim, 2, dtype=torch.float32)
            / config.rotary_dim
        )
        angles = (
            torch.arange(max_position, dtype=torch.float32).unsqueeze(-1)
            * inverse_frequencies
        )
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((angles.cos(), angles.sin()), dim=-1),
            persistent=False,
        )
        self.kernel: JointRopeInplaceKernel | None = (
            current_platform.get_joint_rope_inplace_kernel()
        )

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_count = positions.shape[0]
        rotary_query = query.view(row_count, -1, self.head_dim)[..., : self.rotary_dim]
        rotary_key = key.view(row_count, -1, self.head_dim)[..., : self.rotary_dim]
        if self.kernel is not None and query.is_cuda:
            self.kernel(
                rotary_query, rotary_key, self.cos_sin_cache, positions, is_neox=False
            )
        else:
            cosine, sine = self.cos_sin_cache[positions].unsqueeze(1).chunk(2, dim=-1)
            for rotary_states in (rotary_query, rotary_key):
                even = rotary_states[..., 0::2].float()
                odd = rotary_states[..., 1::2].float()
                rotary_states.copy_(
                    torch.stack(
                        (even * cosine - odd * sine, odd * cosine + even * sine), dim=-1
                    ).flatten(-2)
                )
        return query, key


class DecoderSelfAttention(nn.Module):
    def __init__(
        self,
        config: FishOmniAudioDecoderConfig,
        rotary_embedding: InterleavedRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.head_count: int = config.n_head
        self.kv_head_count: int = config.n_local_heads
        self.head_dim: int = config.head_dim
        query_width = config.n_head * config.head_dim
        key_value_width = config.n_local_heads * config.head_dim
        # Output rows: each head's [query, gate] pair, then key, then value.
        self.qkv_proj: nn.Linear = nn.Linear(
            config.dim, 2 * query_width + 2 * key_value_width, bias=False
        )
        self.o_proj: nn.Linear = nn.Linear(query_width, config.dim, bias=False)
        self.q_norm: GemmaRMSNorm = GemmaRMSNorm(config.head_dim, config.norm_eps)
        self.k_norm: GemmaRMSNorm = GemmaRMSNorm(config.head_dim, config.norm_eps)
        self.rotary_embedding: InterleavedRotaryEmbedding = rotary_embedding

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention: DecoderAttention,
        layer_index: int,
    ) -> torch.Tensor:
        row_count = hidden_states.shape[0]
        query_width = self.head_count * self.head_dim
        key_value_width = self.kv_head_count * self.head_dim
        query_and_gate, key, value = self.qkv_proj(hidden_states).split(
            [2 * query_width, key_value_width, key_value_width], dim=-1
        )
        query, gate = query_and_gate.view(
            row_count, self.head_count, 2 * self.head_dim
        ).chunk(2, dim=-1)
        query, key = self.rotary_embedding(
            positions,
            self.q_norm(query.reshape(-1, self.head_dim)).view(row_count, query_width),
            self.k_norm(key.reshape(-1, self.head_dim)).view(
                row_count, key_value_width
            ),
        )
        attended = attention(layer_index, query, key, value)
        return self.o_proj(
            attended * torch.sigmoid(gate.reshape(row_count, query_width))
        )


class DecoderMlp(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # Output rows: gate, then up.
        self.gate_up_proj: nn.Linear = nn.Linear(
            hidden_size, 2 * intermediate_size, bias=False
        )
        self.silu_and_mul: SiluAndMul = SiluAndMul()
        self.down_proj: nn.Linear = nn.Linear(
            intermediate_size, hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.silu_and_mul(self.gate_up_proj(hidden_states)))


class DecoderLayer(nn.Module):
    """One tower layer: mix a code embedding into the hidden, then a pre-norm block."""

    def __init__(
        self,
        config: FishOmniAudioDecoderConfig,
        rotary_embedding: InterleavedRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.token_embedding: nn.Embedding = nn.Embedding(config.vocab_size, config.dim)
        self.enorm: GemmaRMSNorm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.hnorm: GemmaRMSNorm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.eh_proj: nn.Linear = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.self_attn: DecoderSelfAttention = DecoderSelfAttention(
            config, rotary_embedding
        )
        self.mlp: DecoderMlp = DecoderMlp(config.dim, config.intermediate_size)
        self.input_layernorm: GemmaRMSNorm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.post_attention_layernorm: GemmaRMSNorm = GemmaRMSNorm(
            config.dim, config.norm_eps
        )
        self.final_layernorm: GemmaRMSNorm = GemmaRMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        code_ids: torch.Tensor,
        label_is_semantic: torch.Tensor,
        shared_input: torch.Tensor | None,
        positions: torch.Tensor,
        attention: DecoderAttention,
        layer_index: int,
    ) -> torch.Tensor:
        # Clamping keeps placeholder codes of non-semantic labels inside the
        # table; the label mask then zeroes those rows.
        embeddings = torch.where(
            label_is_semantic.unsqueeze(-1),
            self.token_embedding(
                code_ids.clamp(0, self.token_embedding.num_embeddings - 1)
            ),
            0.0,
        )
        if shared_input is not None:
            embeddings = embeddings + shared_input.to(embeddings.dtype)
        else:
            pass
        mixed = self.eh_proj(
            torch.cat((self.enorm(embeddings), self.hnorm(hidden_states)), dim=-1)
        )
        feedforward_input, residual = self.post_attention_layernorm(
            self.self_attn(
                self.input_layernorm(mixed), positions, attention, layer_index
            ),
            mixed,
        )
        normed, _ = self.final_layernorm(self.mlp(feedforward_input), residual)
        return normed

    def tower_layer_weights(self) -> TowerLayerWeights:
        attention = self.self_attn
        return TowerLayerWeights(
            token_embedding=self.token_embedding.weight,
            enorm=norm_weights(self.enorm),
            hnorm=norm_weights(self.hnorm),
            eh_proj=self.eh_proj,
            input_norm=norm_weights(self.input_layernorm),
            qkv_proj=attention.qkv_proj,
            q_norm=norm_weights(attention.q_norm),
            k_norm=norm_weights(attention.k_norm),
            o_proj=attention.o_proj,
            post_norm=norm_weights(self.post_attention_layernorm),
            gate_up_proj=self.mlp.gate_up_proj,
        )
