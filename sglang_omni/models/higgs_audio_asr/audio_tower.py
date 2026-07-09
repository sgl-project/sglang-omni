# SPDX-License-Identifier: Apache-2.0
"""Higgs-Audio audio tower + feature projector (transformers-5 native).

Tower = Whisper encoder (conv1 -> conv2(stride 2) -> learned positions ->
``WhisperEncoderLayer`` stack) followed by ``AvgPool1d(2)`` over time and
a final LayerNorm — 25 embeddings/s. Projector = depthwise
``Conv1d(stride 2)`` temporal downsample + 2-layer ReLU MLP into LLM
space — 12.5 embeddings/s. Parameter paths mirror the checkpoint's
``audio_tower.*`` / ``audio_encoder_proj.*`` so weights load by name.

Reimplemented (rather than running the checkpoint's remote code) because
the shipped modeling code targets the transformers-4 layer API:
``encoder_layer(...)[0]`` strips the batch dim on transformers 5.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

from .configuration_higgs_audio_asr import HiggsAudio3Config


class HiggsAudioTower(nn.Module):
    def __init__(self, config: HiggsAudio3Config):
        super().__init__()
        enc = config.audio_encoder_config
        embed_dim = enc.d_model
        whisper_cfg = WhisperConfig(
            d_model=embed_dim,
            encoder_layers=enc.encoder_layers,
            encoder_attention_heads=enc.encoder_attention_heads,
            encoder_ffn_dim=enc.encoder_ffn_dim,
            num_mel_bins=enc.num_mel_bins,
            max_source_positions=enc.max_source_positions,
            activation_function="gelu",
            dropout=0.0,
            activation_dropout=0.0,
            attention_dropout=0.0,
        )
        whisper_cfg._attn_implementation = "sdpa"

        self.conv1 = nn.Conv1d(enc.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(enc.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(whisper_cfg) for _ in range(enc.encoder_layers)]
        )
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.layer_norm = nn.LayerNorm(embed_dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """(batch, num_mel_bins, T_mel) -> (batch, ~T_mel/4, d_model)."""
        hidden = F.gelu(self.conv1(input_features))
        hidden = F.gelu(self.conv2(hidden))
        hidden = hidden.permute(0, 2, 1)  # (B, T, D)
        hidden = hidden + self.embed_positions.weight[: hidden.shape[1]]
        for layer in self.layers:
            hidden = layer(hidden, None)
        hidden = self.avg_pooler(hidden.permute(0, 2, 1)).permute(0, 2, 1)
        return self.layer_norm(hidden)


class HiggsAudioFeatureProjector(nn.Module):
    """"mlp" projector with stride-2 temporal downsample (the v3-stt
    config; the "linear" variant is not supported here)."""

    def __init__(self, config: HiggsAudio3Config):
        super().__init__()
        audio_dim = config.audio_encoder_config.d_model
        llm_dim = config.text_config.hidden_size
        assert config.projector_temporal_downsample == 2, (
            f"Only stride-2 temporal downsample is supported; "
            f"got {config.projector_temporal_downsample}."
        )
        self.temporal = nn.Conv1d(
            audio_dim, audio_dim, 3, 2, padding=1, groups=audio_dim, bias=True
        )
        self.linear1 = nn.Linear(audio_dim, 2048, bias=True)
        self.linear2 = nn.Linear(2048, llm_dim, bias=True)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """(B, T, audio_dim) -> (B, ceil(T/2), llm_dim)."""
        x = self.temporal(audio_features.permute(0, 2, 1)).permute(0, 2, 1)
        return self.linear2(F.relu(self.linear1(x)))
