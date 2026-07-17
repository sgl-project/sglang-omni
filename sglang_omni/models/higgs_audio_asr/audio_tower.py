# SPDX-License-Identifier: Apache-2.0
"""Higgs-Audio audio tower + feature projector.

note (zhudian): reimplemented natively rather than running the checkpoint's
remote code because the shipped modeling code targets the transformers-4 layer
API (``encoder_layer(...)[0]``), which strips the batch dim on transformers 5.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

from .configuration_higgs_audio_asr import HiggsAudio3Config


def _conv2_valid_lengths(mel_lengths: torch.Tensor) -> torch.Tensor:
    """Valid post-``conv2`` frame count (k=3, s=2, p=1) for each sample."""
    return (mel_lengths - 1) // 2 + 1


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

    def _key_padding_bias(
        self, mel_lengths: torch.Tensor, seq_len: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Additive ``(B, 1, 1, seq_len)`` attention bias masking padded keys."""
        valid = _conv2_valid_lengths(mel_lengths)  # (B,)
        positions = torch.arange(seq_len, device=mel_lengths.device)
        pad = positions[None, :] >= valid[:, None]  # (B, seq_len)
        bias = torch.zeros(
            mel_lengths.shape[0], 1, 1, seq_len, dtype=dtype, device=mel_lengths.device
        )
        return bias.masked_fill(pad[:, None, None, :], torch.finfo(dtype).min)

    def forward(
        self,
        input_features: torch.Tensor,
        mel_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, num_mel_bins, T_mel) -> (B, ~T_mel/4, d_model)``.

        note (zhudian): ``mel_lengths`` (valid mel frames per sample) is
        required whenever the batch mixes clip lengths — Whisper attention is
        bidirectional, so without masking a short/partial chunk attends to the
        right-padding shared with longer chunks and slicing the output
        afterwards cannot recover the correct embeddings.
        """
        hidden = F.gelu(self.conv1(input_features))
        hidden = F.gelu(self.conv2(hidden))
        hidden = hidden.permute(0, 2, 1)  # (B, T, D)
        hidden = hidden + self.embed_positions.weight[: hidden.shape[1]]

        attention_mask = None
        if mel_lengths is not None:
            attention_mask = self._key_padding_bias(
                mel_lengths, hidden.shape[1], hidden.dtype
            )
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        hidden = self.avg_pooler(hidden.permute(0, 2, 1)).permute(0, 2, 1)
        return self.layer_norm(hidden)


class HiggsAudioFeatureProjector(nn.Module):
    """Stride-2 temporal-downsample conv + 2-layer ReLU MLP into LLM space."""

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
        x = self.temporal(audio_features.permute(0, 2, 1)).permute(0, 2, 1)
        return self.linear2(F.relu(self.linear1(x)))
