# SPDX-License-Identifier: Apache-2.0
"""The causal audio encoder of the ELLSA 2 generator: convolutional subsampling, then a causal stack of Llama layers."""

from __future__ import annotations

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from sglang_omni.models.ellsa2.fbank import NUM_MEL_BINS

# The checkpoint's encoder: a 3x3 convolution, then one with stride 2 halving the frames, 128 channels each; 22 layers, 1280 wide, 16 heads, a GELU MLP four times as wide.
ENCODER_WIDTH = 1280
ENCODER_LAYERS = 22
ENCODER_HEADS = 16
SUBSAMPLING_CHANNELS = 128
SUBSAMPLING_KERNEL = 3
SUBSAMPLING_STRIDE = 2
# Positions the encoder's rotary table covers: 2048 frames after subsampling, about 41 s of audio, beyond the 12 s history.
MAX_POSITIONS = 2048
FLASH_ATTENTION = "flash_attention_2"


def encoder_config(attn_implementation: str) -> LlamaConfig:
    return LlamaConfig(
        hidden_size=ENCODER_WIDTH,
        intermediate_size=4 * ENCODER_WIDTH,
        num_hidden_layers=ENCODER_LAYERS,
        vocab_size=1,
        num_attention_heads=ENCODER_HEADS,
        max_position_embeddings=MAX_POSITIONS,
        hidden_act="gelu",
        rms_norm_eps=1e-6,
        attn_implementation=attn_implementation,
    )


class Subsampling(nn.Module):
    """[batch, frames, NUM_MEL_BINS] → [batch, (frames - 1) // 2, width]."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, SUBSAMPLING_CHANNELS, SUBSAMPLING_KERNEL, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(
                SUBSAMPLING_CHANNELS,
                SUBSAMPLING_CHANNELS,
                SUBSAMPLING_KERNEL,
                stride=SUBSAMPLING_STRIDE,
            ),
            nn.GELU(),
        )
        self.out = nn.Linear(
            SUBSAMPLING_CHANNELS * ((NUM_MEL_BINS - 1) // SUBSAMPLING_STRIDE),
            width,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(features.unsqueeze(1))
        batch, channels, frames, bins = hidden.size()
        return self.out(
            hidden.transpose(1, 2).contiguous().view(batch, frames, channels * bins)
        )


class CausalLlamaStack(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.takes_padding_mask = (
            config._attn_implementation == FLASH_ATTENTION
        )  # noqa: leading-underscore  # upstream spelling
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, index)
                for index in range(config.num_hidden_layers)
            ]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden, position_ids)
        # Note (Junnan Li): The reference hands flash-attention a padding mask, which runs its varlen kernel; that kernel rounds differently from the dense one, and other backends read a 2-D mask as non-causal.
        padding_mask = (
            torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            if self.takes_padding_mask
            else None
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=padding_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        return self.norm(hidden)


class CausalAudioEncoder(nn.Module):
    """Log-mel features [batch, frames, NUM_MEL_BINS] → [batch, (frames - 1) // 2, hidden], each output reading only earlier frames."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.encoder_embed = Subsampling(config.hidden_size)
        self.encoder = CausalLlamaStack(config)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.encoder_embed(features))


__all__ = ["FLASH_ATTENTION", "CausalAudioEncoder", "encoder_config"]
