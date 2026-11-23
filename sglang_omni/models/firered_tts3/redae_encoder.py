# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encode reference audio into RedAE conditioning latents."""

import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3Model

MAX_DOWNSAMPLE_PATCHES_PER_BATCH = 32768


class Qwen3ClsDownsample(torch.nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        downsample_rate: int = 2,
        hidden_size: int = 896,
        intermediate_size: int = 896 * 4,
        num_hidden_layers: int = 4,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
    ) -> None:
        super().__init__()
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.downsample_rate: int = downsample_rate
        self.qwen3_config: Qwen3Config = Qwen3Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        self.qwen3: Qwen3Model = Qwen3Model(self.qwen3_config)
        self.cls_tok: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones(1, 1, hidden_size)
        )
        self.in_proj: torch.nn.Linear | torch.nn.Identity = (
            torch.nn.Linear(in_dim, hidden_size)
            if in_dim != hidden_size
            else torch.nn.Identity()
        )
        self.out_proj: torch.nn.Linear | torch.nn.Identity = (
            torch.nn.Linear(hidden_size, out_dim)
            if out_dim != hidden_size
            else torch.nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Downsample latents as BERT [CLS] token."""
        assert hidden_states.shape[1] % self.downsample_rate == 0, hidden_states.shape
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.downsample_rate, self.in_dim)
        hidden_states = self.in_proj(hidden_states)
        cls_tok = self.cls_tok.expand(hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat([hidden_states, cls_tok], dim=1)
        output_embeddings = []
        for chunk_hidden_states in torch.split(
            hidden_states, MAX_DOWNSAMPLE_PATCHES_PER_BATCH, dim=0
        ):
            chunk_output = self.qwen3(inputs_embeds=chunk_hidden_states)
            chunk_embeddings = chunk_output.last_hidden_state[:, -1]
            output_embeddings.append(chunk_embeddings)
        output_embeddings = torch.cat(output_embeddings, dim=0)
        output_embeddings = self.out_proj(output_embeddings)
        output_embeddings = output_embeddings.reshape(batch_size, -1, self.out_dim)
        return output_embeddings


class RedAEAudioEncoder(torch.nn.Module):

    def __init__(
        self,
        out_dim: int = 1024,
        audio_patch_size: int = 480,
        audio_sample_rate: int = 24000,
        hidden_size: int = 896,
        intermediate_size: int = 896 * 4,
        num_hidden_layers: int = 24,
        max_position_embeddings: int = 32768,
        max_window_layers: int = 0,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        sliding_window: int = 64,
        use_sliding_window: bool = True,
        extra_downsample_rate: int = 2,
        downsample_num_hidden_layers: int = 4,
    ) -> None:
        super().__init__()
        self.audio_patch_size: int = audio_patch_size
        self.audio_sample_rate: int = audio_sample_rate
        self.extra_downsample_rate: int = extra_downsample_rate
        self.out_dim: int = out_dim
        self.in_proj: nn.Sequential = nn.Sequential(
            nn.Linear(self.audio_patch_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.qwen3_config: Qwen3Config = Qwen3Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            max_window_layers=max_window_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            sliding_window=sliding_window,
            use_sliding_window=use_sliding_window,
        )
        self.qwen3: Qwen3Model = Qwen3Model(self.qwen3_config)
        if self.extra_downsample_rate > 1:
            self.downsample: Qwen3ClsDownsample = Qwen3ClsDownsample(
                in_dim=hidden_size,
                out_dim=hidden_size,
                downsample_rate=extra_downsample_rate,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_hidden_layers=downsample_num_hidden_layers,
                max_position_embeddings=max_position_embeddings,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )
        else:
            pass
        self.out_proj: nn.Linear = nn.Linear(hidden_size, out_dim)

    @property
    def downsample_rate(self) -> int:
        return int(self.audio_patch_size * self.extra_downsample_rate)

    @property
    def sample_rate(self) -> int:
        return self.audio_sample_rate

    @property
    def hidden_size(self) -> int:
        return self.out_dim

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        assert audio.shape[1] % self.downsample_rate == 0, audio.shape
        hidden_states = audio.unfold(
            dimension=1, size=self.audio_patch_size, step=self.audio_patch_size
        )
        hidden_states = self.in_proj(hidden_states)
        model_output = self.qwen3(inputs_embeds=hidden_states, attention_mask=None)
        hidden_states = model_output.last_hidden_state
        if self.extra_downsample_rate > 1:
            hidden_states = self.downsample(hidden_states)
        else:
            pass
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
