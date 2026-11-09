# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatible YuE2 autoregressive backbone."""

import torch
from torch import nn

from sglang_omni.models.yue2.layers import (
    MLP,
    Attention,
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

    def enable_fused_kernels(self) -> None:
        weight = self.model.embed_tokens.weight
        if weight.device.type != "cuda" or weight.dtype != torch.bfloat16:
            raise ValueError("YuE2 fused kernels require CUDA BF16 weights")
        else:
            for module in self.modules():
                if isinstance(module, (Attention, MLP)):
                    module.fuse_projections()
                    module.fused_kernels = True
                elif isinstance(module, RMSNorm):
                    module.fused_kernels = True
                else:
                    pass

    def forward(self, tokens: torch.Tensor, cache: TokenCache) -> torch.Tensor:
        token_count = tokens.shape[1]
        assert cache.length == 0 or token_count == 1
        positions = torch.arange(
            cache.length,
            cache.length + token_count,
            device=tokens.device,
            dtype=torch.int32,
        )[None]
        cosine, sine = self.model.rotary_emb(positions)
        residual = self.model.embed_tokens(tokens)
        hidden_states = self.model.layers[0].input_layernorm(residual)
        for layer_index, layer in enumerate(self.model.layers):
            query, key, value = layer.self_attn.project_qkv(
                hidden_states, cosine, sine, positions
            )
            key, value = cache.append(layer_index, key, value)
            attended = layer.self_attn.attend(query, key, value, causal=token_count > 1)
            hidden_states, residual = layer.post_attention_layernorm.add(
                attended, residual
            )
            hidden_states = layer.mlp(hidden_states)
            if layer_index + 1 < len(self.model.layers):
                hidden_states, residual = self.model.layers[
                    layer_index + 1
                ].input_layernorm.add(hidden_states, residual)
            else:
                hidden_states, residual = self.model.norm.add(
                    hidden_states[:, -1:], residual[:, -1:]
                )
        cache.length += token_count
        return self.lm_head(hidden_states[:, -1:])[:, -1]
