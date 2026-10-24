# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
"""Whisper features and convolutional projection for Ming 3 audio."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from pydantic import BaseModel
from torch import nn
from whisper.model import ResidualAttentionBlock

from sglang_omni.models.weight_loader import load_weights_by_prefix


class AudioEncoderConfig(BaseModel):
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    downsampler_kernel_size_list: list[int]
    downsampler_stride_list: list[int]
    whisper_chunk_size: int = 3000

    def output_length(self, mel_frames: int) -> int:
        total = 0
        for offset in range(0, mel_frames, self.whisper_chunk_size):
            length = (min(self.whisper_chunk_size, mel_frames - offset) + 1) // 2
            for kernel, stride in zip(
                self.downsampler_kernel_size_list,
                self.downsampler_stride_list,
                strict=True,
            ):
                length = (length + 2 * (kernel // 2) - kernel) // stride + 1
            total += length
        return total


class Float32LayerNorm(nn.LayerNorm):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return functional.layer_norm(
            hidden.float(),
            self.normalized_shape,
            self.weight.float(),
            self.bias.float(),
            self.eps,
        ).to(hidden.dtype)


class WhisperEncoder(nn.Module):
    def __init__(
        self, config: AudioEncoderConfig, *, should_upcast_layer_norm: bool
    ) -> None:
        super().__init__()
        self.conv1: nn.Conv1d = nn.Conv1d(
            config.n_mels, config.n_audio_state, 3, padding=1
        )
        self.conv2: nn.Conv1d = nn.Conv1d(
            config.n_audio_state, config.n_audio_state, 3, stride=2, padding=1
        )
        self.positional_embedding: nn.Parameter = nn.Parameter(
            torch.zeros(config.n_audio_ctx, config.n_audio_state)
        )
        self.blocks: nn.ModuleList = nn.ModuleList(
            ResidualAttentionBlock(config.n_audio_state, config.n_audio_head)
            for _ in range(config.n_audio_layer)
        )
        # note (Codex): Whisper's own LayerNorm casts BF16 inputs to float32
        # but keeps BF16 parameters, which fails; each checkpoint picks a fix.
        if should_upcast_layer_norm:
            layer_norm_type = Float32LayerNorm
        else:
            layer_norm_type = nn.LayerNorm
        for block in self.blocks:
            block.attn_ln = layer_norm_type(config.n_audio_state)
            block.mlp_ln = layer_norm_type(config.n_audio_state)
        self.ln_post: nn.LayerNorm = layer_norm_type(config.n_audio_state)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        hidden = functional.gelu(self.conv1(mel.transpose(1, 2)))
        hidden = functional.gelu(self.conv2(hidden)).transpose(1, 2)
        hidden = hidden + self.positional_embedding[: hidden.shape[1]]
        for block in self.blocks:
            hidden = block(hidden)
        return self.ln_post(hidden)


class Transpose(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.transpose(-1, -2)


class Ming3AudioEncoder(nn.Module):
    def __init__(
        self,
        config: AudioEncoderConfig,
        hidden_size: int,
        *,
        should_upcast_layer_norm: bool,
        should_batch_chunks: bool,
    ) -> None:
        super().__init__()
        self.config: AudioEncoderConfig = config
        self.should_batch_chunks: bool = should_batch_chunks
        self.audio: WhisperEncoder = WhisperEncoder(
            config, should_upcast_layer_norm=should_upcast_layer_norm
        )
        projection_layers: list[nn.Module] = []
        channels = config.n_audio_state
        for kernel, stride in zip(
            config.downsampler_kernel_size_list,
            config.downsampler_stride_list,
            strict=True,
        ):
            projection_layers.extend(
                [
                    nn.Conv1d(channels, hidden_size, kernel, stride, kernel // 2),
                    nn.GELU(),
                ]
            )
            channels = hidden_size
        projection_layers.extend([Transpose(), nn.Linear(hidden_size, hidden_size)])
        self.linear_proj_audio: nn.Sequential = nn.Sequential(*projection_layers)

    def load_checkpoint(self, model_path: str) -> None:
        self.audio.load_state_dict(
            load_weights_by_prefix(model_path, prefix="model.audio.")
        )
        self.linear_proj_audio.load_state_dict(
            load_weights_by_prefix(model_path, prefix="linear_proj_audio.")
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        chunks = [
            chunk.to(self.audio.conv1.weight)
            for mel in features
            for chunk in mel.split(self.config.whisper_chunk_size)
        ]
        if self.should_batch_chunks:
            lengths = [self.config.output_length(chunk.shape[0]) for chunk in chunks]
            padded_mels = nn.utils.rnn.pad_sequence(chunks, batch_first=True)
            projected = self.linear_proj_audio(self.audio(padded_mels).transpose(1, 2))
            embeddings = [row[:length] for row, length in zip(projected, lengths)]
        else:
            # note (Codex): Whisper attention has no padding mask, so unbatched
            # chunks keep every chunk's output independent of its neighbors.
            embeddings = [
                self.linear_proj_audio(self.audio(chunk[None]).transpose(1, 2))[0]
                for chunk in chunks
            ]
        return functional.normalize(torch.cat(embeddings), dim=-1)
