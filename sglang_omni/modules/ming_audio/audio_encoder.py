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
        hidden = self.convolve(mel)
        hidden = hidden + self.positional_embedding[: hidden.shape[1]]
        for block in self.blocks:
            hidden = block(hidden)
        return self.ln_post(hidden)

    def convolve(self, mel: torch.Tensor) -> torch.Tensor:
        hidden = functional.gelu(self.conv1(mel.transpose(1, 2)))
        return functional.gelu(self.conv2(hidden)).transpose(1, 2)

    def attend(
        self,
        block: ResidualAttentionBlock,
        hidden: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> torch.Tensor:
        normed = block.attn_ln(hidden)
        query = block.attn.query(normed)
        key = block.attn.key(normed)
        value = block.attn.value(normed)
        batch_size, time_steps, state_size = query.shape
        head_count = block.attn.n_head
        head_size = state_size // head_count

        def split_heads(states: torch.Tensor) -> torch.Tensor:
            return states.view(batch_size, time_steps, head_count, head_size).permute(
                0, 2, 1, 3
            )

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        positions = torch.arange(time_steps, device=hidden.device)
        valid_positions = positions.unsqueeze(0) < valid_frames.unsqueeze(1)
        padding_mask = torch.zeros(
            batch_size, 1, 1, time_steps, dtype=query.dtype, device=hidden.device
        )
        padding_mask = padding_mask.masked_fill(
            ~valid_positions[:, None, None, :], torch.finfo(query.dtype).min
        )
        attended = functional.scaled_dot_product_attention(
            query, key, value, attn_mask=padding_mask
        )
        attended = attended.permute(0, 2, 1, 3).reshape(
            batch_size, time_steps, state_size
        )
        hidden = hidden + block.attn.out(attended)
        return hidden + block.mlp(block.mlp_ln(hidden))


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

    def embed_mels(self, mels: list[torch.Tensor]) -> list[torch.Tensor]:
        weight = self.audio.conv1.weight
        frame_counts = [int(mel.shape[0]) for mel in mels]
        if len(set(frame_counts)) == 1:
            stacked = torch.stack(
                [mel.to(device=weight.device, dtype=weight.dtype) for mel in mels]
            )
            encoded = self.audio(stacked)
            conv_counts = [encoded.shape[1]] * len(mels)
        else:
            convolved = [
                self.audio.convolve(
                    mel.to(device=weight.device, dtype=weight.dtype)[None]
                )[0]
                for mel in mels
            ]
            conv_counts = [int(row.shape[0]) for row in convolved]
            time_steps = max(conv_counts)
            hidden = torch.zeros(
                len(mels),
                time_steps,
                convolved[0].shape[-1],
                device=weight.device,
                dtype=weight.dtype,
            )
            for index, row in enumerate(convolved):
                hidden[index, : row.shape[0]] = (
                    row + self.audio.positional_embedding[: row.shape[0]]
                )
            valid_frames = torch.tensor(conv_counts, device=weight.device)
            for block in self.audio.blocks:
                hidden = self.audio.attend(block, hidden, valid_frames)
            encoded = self.audio.ln_post(hidden)
        embeddings: list[torch.Tensor] = []
        for index, frames in enumerate(frame_counts):
            projected = self.linear_proj_audio(
                encoded[index, : conv_counts[index]].transpose(0, 1).unsqueeze(0)
            )
            token_count = self.config.output_length(frames)
            embeddings.append(functional.normalize(projected[0, :token_count], dim=-1))
        return embeddings
