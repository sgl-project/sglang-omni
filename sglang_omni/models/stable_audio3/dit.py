# SPDX-License-Identifier: Apache-2.0
# Adapted from Stability-AI/stable-audio-3 at commit 790793b89ddceea612be9ce42ff6830d18f6fa57.
# Copyright (c) 2026 Stability AI. The MIT license notice is retained in licenses/.
"""Stable Audio 3 diffusion transformer with adaLN timing and text cross-attention."""

import torch
from torch import nn
from torch.nn import functional as F

from sglang_omni.models.stable_audio3.layers import (
    DifferentialAttention,
    GatedFeedForward,
    RMSNorm,
    RotaryEmbedding,
    expo_fourier_features,
)
from sglang_omni.models.stable_audio3.model_config import DiTOptions

NORM_EPS = 1e-5
TIMESTEP_FEATURES = 256
MIN_ROTARY_DIM = 32
ADALN_CHUNKS = 6


def projection(input_dim: int, output_dim: int, *, bias: bool) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim, bias=bias),
        nn.SiLU(),
        nn.Linear(output_dim, output_dim, bias=bias),
    )


class TransformerBlock(nn.Module):
    def __init__(self, config: DiTOptions) -> None:
        super().__init__()
        dim, head_dim = config.embed_dim, config.embed_dim // config.num_heads
        self.pre_norm = RMSNorm(dim, eps=NORM_EPS, force_fp32=True)
        self.self_attn = DifferentialAttention(
            dim, head_dim=head_dim, context_dim=None, qk_norm="rms"
        )
        self.cross_attend_norm = RMSNorm(dim, eps=NORM_EPS, force_fp32=True)
        self.cross_attn = DifferentialAttention(
            dim, head_dim=head_dim, context_dim=dim, qk_norm="rms"
        )
        self.ff_norm = RMSNorm(dim, eps=NORM_EPS, force_fp32=True)
        self.ff = GatedFeedForward(
            dim, multiplier=config.ff_kwargs.mult, sinusoidal=False
        )
        self.to_scale_shift_gate = nn.Parameter(torch.zeros(ADALN_CHUNKS * dim))
        self.to_local_embed = nn.Sequential(
            nn.Linear(config.local_add_cond_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor,
        global_condition: torch.Tensor,
        rotary_angles: torch.Tensor,
        attention_mask: torch.Tensor,
        local_embedding: torch.Tensor,
    ) -> torch.Tensor:
        scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (
            (self.to_scale_shift_gate + global_condition)
            .unsqueeze(1)
            .chunk(ADALN_CHUNKS, dim=-1)
        )
        attention_output = self.self_attn(
            self.pre_norm(x) * (1 + scale_self) + shift_self,
            rotary_angles=rotary_angles,
            attention_mask=attention_mask,
        )
        x = x + attention_output * torch.sigmoid(1 - gate_self)
        x = x + self.cross_attn(self.cross_attend_norm(x), context=context)
        x = x + local_embedding
        feed_forward_output = self.ff(self.ff_norm(x) * (1 + scale_ff) + shift_ff)
        return x + feed_forward_output * torch.sigmoid(1 - gate_ff)


class ContinuousTransformer(nn.Module):
    def __init__(self, config: DiTOptions) -> None:
        super().__init__()
        dim = config.embed_dim
        self.project_in = nn.Linear(config.io_channels, dim, bias=False)
        self.project_out = nn.Linear(dim, config.io_channels, bias=False)
        self.rotary_pos_emb = RotaryEmbedding(
            max(dim // config.num_heads // 2, MIN_ROTARY_DIM)
        )
        self.memory_tokens = nn.Parameter(torch.zeros(config.num_memory_tokens, dim))
        self.local_condition_dim = config.local_add_cond_dim
        self.global_cond_embedder = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim * ADALN_CHUNKS)
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.depth)
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor,
        global_embedding: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, latent_length, _ = x.shape
        num_memory_tokens = self.memory_tokens.shape[0]
        x = torch.cat(
            [self.memory_tokens.expand(batch_size, -1, -1), self.project_in(x)], dim=1
        )
        rotary_angles = self.rotary_pos_emb(x.shape[1])
        global_condition = self.global_cond_embedder(global_embedding)
        positions = torch.arange(x.shape[1], device=x.device)
        # note (ruilin): padded latents lie past the returned audio, so masking their keys suffices.
        attention_mask = (positions < (valid_lengths + num_memory_tokens)[:, None])[
            :, None, None, :
        ]
        zero_local_condition = x.new_zeros(1, 1, self.local_condition_dim)
        for layer in self.layers:
            local_embedding = layer.to_local_embed(zero_local_condition).expand(
                batch_size, latent_length, -1
            )
            x = layer(
                x,
                context=context,
                global_condition=global_condition,
                rotary_angles=rotary_angles,
                attention_mask=attention_mask,
                local_embedding=F.pad(local_embedding, (0, 0, num_memory_tokens, 0)),
            )
        return self.project_out(x[:, num_memory_tokens:])


class DiffusionTransformer(nn.Module):
    """Predicts the flow velocity of latents; the inpainting inputs are zero for text-to-audio."""

    def __init__(self, config: DiTOptions) -> None:
        super().__init__()
        dim = config.embed_dim
        self.to_timestep_embed = projection(TIMESTEP_FEATURES, dim, bias=True)
        self.to_cond_embed = projection(config.cond_token_dim, dim, bias=False)
        self.to_global_embed = projection(config.global_cond_dim, dim, bias=False)
        self.transformer = ContinuousTransformer(config)
        self.preprocess_conv = nn.Conv1d(
            config.io_channels, config.io_channels, 1, bias=False
        )
        self.postprocess_conv = nn.Conv1d(
            config.io_channels, config.io_channels, 1, bias=False
        )

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        condition_tokens: torch.Tensor,
        seconds_embedding: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> torch.Tensor:
        timestep_embedding = self.to_timestep_embed(
            expo_fourier_features(timesteps, TIMESTEP_FEATURES)
        )
        hidden_states = self.transformer(
            (self.preprocess_conv(latents) + latents).transpose(1, 2),
            context=self.to_cond_embed(condition_tokens),
            global_embedding=self.to_global_embed(seconds_embedding)
            + timestep_embedding,
            valid_lengths=valid_lengths,
        ).transpose(1, 2)
        return self.postprocess_conv(hidden_states) + hidden_states
