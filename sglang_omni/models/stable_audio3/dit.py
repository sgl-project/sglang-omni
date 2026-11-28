# SPDX-License-Identifier: Apache-2.0
# Adapted from Stability-AI/stable-audio-3 at commit 790793b89ddceea612be9ce42ff6830d18f6fa57.
# Copyright (c) 2026 Stability AI. The MIT license notice is retained in licenses/.
"""Stable Audio 3 diffusion transformer with adaLN timing and text cross-attention."""

from dataclasses import dataclass

import torch
from torch import nn

from sglang_omni.models.stable_audio3.layers import (
    AttentionBackend,
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


@dataclass(frozen=True, kw_only=True)
class SamplingConditions:
    """What stays fixed across one batch's sampling steps, computed once per batch."""

    seconds_embedding: torch.Tensor
    context_keys_values: list[tuple[torch.Tensor, torch.Tensor]]
    local_embeddings: list[torch.Tensor]
    rotary_angles: torch.Tensor
    key_lengths: torch.Tensor


def projection(input_dim: int, output_dim: int, *, bias: bool) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim, bias=bias),
        nn.SiLU(),
        nn.Linear(output_dim, output_dim, bias=bias),
    )


class TransformerBlock(nn.Module):
    def __init__(self, config: DiTOptions, *, backend: AttentionBackend) -> None:
        super().__init__()
        dim, head_dim = config.embed_dim, config.embed_dim // config.num_heads
        self.pre_norm = RMSNorm(dim, eps=NORM_EPS, force_fp32=True)
        self.self_attn = DifferentialAttention(
            dim, head_dim=head_dim, context_dim=None, qk_norm="rms", backend=backend
        )
        self.cross_attend_norm = RMSNorm(dim, eps=NORM_EPS, force_fp32=True)
        self.cross_attn = DifferentialAttention(
            dim, head_dim=head_dim, context_dim=dim, qk_norm="rms", backend=backend
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
        hidden_states: torch.Tensor,
        *,
        context_keys_values: tuple[torch.Tensor, torch.Tensor],
        adaln_modulation: torch.Tensor,
        rotary_angles: torch.Tensor,
        key_lengths: torch.Tensor,
        local_embedding: torch.Tensor,
        num_memory_tokens: int,
    ) -> torch.Tensor:
        scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (
            (self.to_scale_shift_gate + adaln_modulation)
            .unsqueeze(1)
            .chunk(ADALN_CHUNKS, dim=-1)
        )
        attention_output = self.self_attn(
            self.pre_norm(hidden_states) * (1 + scale_self) + shift_self,
            rotary_angles=rotary_angles,
            key_lengths=key_lengths,
        )
        hidden_states = hidden_states + attention_output * torch.sigmoid(1 - gate_self)
        hidden_states = hidden_states + self.cross_attn(
            self.cross_attend_norm(hidden_states),
            context_keys_values=context_keys_values,
        )
        # note (ruilin): the local condition reaches latent tokens only, never the memory tokens; a mask rather than a slice keeps the token count symbolic when compiled.
        is_latent = (
            torch.arange(hidden_states.shape[1], device=hidden_states.device)
            >= num_memory_tokens
        )
        hidden_states = hidden_states + local_embedding * is_latent[:, None].to(
            hidden_states.dtype
        )
        feed_forward_output = self.ff(
            self.ff_norm(hidden_states) * (1 + scale_ff) + shift_ff
        )
        return hidden_states + feed_forward_output * torch.sigmoid(1 - gate_ff)


class ContinuousTransformer(nn.Module):
    def __init__(self, config: DiTOptions, *, backend: AttentionBackend) -> None:
        super().__init__()
        dim = config.embed_dim
        self.project_in = nn.Linear(config.io_channels, dim, bias=False)
        self.project_out = nn.Linear(dim, config.io_channels, bias=False)
        self.rotary_pos_emb = RotaryEmbedding(
            max(dim // config.num_heads // 2, MIN_ROTARY_DIM)
        )
        self.memory_tokens = nn.Parameter(torch.zeros(config.num_memory_tokens, dim))
        self.local_condition_dim: int = config.local_add_cond_dim
        self.global_cond_embedder = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim * ADALN_CHUNKS)
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config, backend=backend) for _ in range(config.depth)
        )

    def forward(
        self,
        latent_tokens: torch.Tensor,
        *,
        global_embedding: torch.Tensor,
        conditions: SamplingConditions,
    ) -> torch.Tensor:
        batch_size = latent_tokens.shape[0]
        num_memory_tokens = self.memory_tokens.shape[0]
        hidden_states = torch.cat(
            [
                self.memory_tokens.expand(batch_size, -1, -1),
                self.project_in(latent_tokens),
            ],
            dim=1,
        )
        adaln_modulation = self.global_cond_embedder(global_embedding)
        for layer, context_keys_values, local_embedding in zip(
            self.layers,
            conditions.context_keys_values,
            conditions.local_embeddings,
            strict=True,
        ):
            hidden_states = layer(
                hidden_states,
                context_keys_values=context_keys_values,
                adaln_modulation=adaln_modulation,
                rotary_angles=conditions.rotary_angles,
                key_lengths=conditions.key_lengths,
                local_embedding=local_embedding,
                num_memory_tokens=num_memory_tokens,
            )
        return self.project_out(hidden_states[:, num_memory_tokens:])


class DiffusionTransformer(nn.Module):
    """Predicts the flow velocity of latents; the inpainting inputs are zero for text-to-audio."""

    def __init__(self, config: DiTOptions, *, backend: AttentionBackend) -> None:
        super().__init__()
        dim = config.embed_dim
        self.to_timestep_embed = projection(TIMESTEP_FEATURES, dim, bias=True)
        self.to_cond_embed = projection(config.cond_token_dim, dim, bias=False)
        self.to_global_embed = projection(config.global_cond_dim, dim, bias=False)
        self.transformer = ContinuousTransformer(config, backend=backend)
        self.preprocess_conv = nn.Conv1d(
            config.io_channels, config.io_channels, 1, bias=False
        )
        self.postprocess_conv = nn.Conv1d(
            config.io_channels, config.io_channels, 1, bias=False
        )

    def prepare(
        self,
        condition_tokens: torch.Tensor,
        seconds_embedding: torch.Tensor,
        *,
        valid_lengths: torch.Tensor,
        latent_length: int,
    ) -> SamplingConditions:
        """Embeds the conditions and the cross-attention keys and values once for every step; padded latents lie past the returned audio, so masking their keys suffices."""
        transformer = self.transformer
        context = self.to_cond_embed(condition_tokens)
        zero_local_condition = context.new_zeros(1, 1, transformer.local_condition_dim)
        num_memory_tokens = transformer.memory_tokens.shape[0]
        return SamplingConditions(
            seconds_embedding=self.to_global_embed(seconds_embedding),
            context_keys_values=[
                layer.cross_attn.context_keys_values(context)
                for layer in transformer.layers
            ],
            local_embeddings=[
                layer.to_local_embed(zero_local_condition).flatten()
                for layer in transformer.layers
            ],
            rotary_angles=transformer.rotary_pos_emb(latent_length + num_memory_tokens),
            key_lengths=(valid_lengths + num_memory_tokens).to(torch.int32),
        )

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        conditions: SamplingConditions,
    ) -> torch.Tensor:
        timestep_embedding = self.to_timestep_embed(
            expo_fourier_features(timesteps, TIMESTEP_FEATURES)
        )
        hidden_states = self.transformer(
            (self.preprocess_conv(latents) + latents).transpose(1, 2),
            global_embedding=conditions.seconds_embedding + timestep_embedding,
            conditions=conditions,
        ).transpose(1, 2)
        return self.postprocess_conv(hidden_states) + hidden_states
