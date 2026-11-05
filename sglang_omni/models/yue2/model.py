# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatible YuE2 autoregressive backbone."""

import torch
from torch import nn

from sglang_omni.models.yue2.layers import (
    AudioPositionEmbedding,
    DecoderLayer,
    RMSNorm,
    RotaryEmbedding,
    TimestepEmbedder,
    TokenCache,
    Yue2ModelConfig,
)


class Backbone(nn.Module):
    def __init__(self, config: Yue2ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)


class Yue2Model(nn.Module):
    def __init__(self, config: Yue2ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Backbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.llm2vae = nn.Linear(config.hidden_size, config.latent_dim)
        self.vae2llm = nn.Linear(config.latent_dim, config.hidden_size)
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.latent_pos_embed = AudioPositionEmbedding(
            config.max_latent_frames, config.hidden_size
        )

    def forward(self, tokens: torch.Tensor, cache: TokenCache) -> torch.Tensor:
        token_count = tokens.shape[1]
        assert cache.length == 0 or token_count == 1
        positions = torch.arange(
            cache.length, cache.length + token_count, device=tokens.device
        )[None]
        cosine, sine = self.model.rotary_emb(positions)
        hidden_states = self.model.embed_tokens(tokens)
        for layer_index, layer in enumerate(self.model.layers):
            query, key, value = layer.self_attn.project_qkv(
                layer.input_layernorm(hidden_states), cosine, sine
            )
            key, value = cache.append(layer_index, key, value)
            hidden_states = hidden_states + layer.self_attn.attend(
                query, key, value, causal=token_count > 1
            )
            hidden_states = hidden_states + layer.mlp(
                layer.post_attention_layernorm(hidden_states)
            )
        cache.length += token_count
        return self.lm_head(self.model.norm(hidden_states[:, -1:]))[:, -1]
