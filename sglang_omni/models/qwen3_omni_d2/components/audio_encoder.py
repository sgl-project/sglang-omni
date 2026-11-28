# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni D2 audio encoder: causal log-mel and the sliding-window audio tower for one waveform stream."""

import torch
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)

MEL_FRAMES_PER_TOKEN = 8
CONV_WINDOW_MEL_FRAMES = 104
ATTENTION_WINDOW_TOKENS = 104
LOG_MEL_FLOOR = 1e-10
LOG_MEL_DYNAMIC_RANGE = 8.0


class CausalLogMel:
    """Whisper log-mel with an end-aligned window and a running peak, so emitted frames never change."""

    def __init__(
        self, mel_filters: torch.Tensor, fft_window_samples: int, hop_samples: int
    ) -> None:
        self.mel_filters = mel_filters
        self.fft_window_samples = fft_window_samples
        self.hop_samples = hop_samples
        self.window = torch.hann_window(fft_window_samples)
        self.history = torch.zeros(fft_window_samples - hop_samples)
        self.pending = torch.zeros(0)
        self.running_peak = torch.tensor(float("-inf"))

    def push(self, waveform: torch.Tensor) -> torch.Tensor:
        """Consume float samples and return the newly finalized [mel, frames] features."""
        available = torch.cat((self.pending, waveform))
        frame_count = int(available.numel()) // self.hop_samples
        complete = available[: frame_count * self.hop_samples]
        self.pending = available[frame_count * self.hop_samples :]
        if frame_count == 0:
            return torch.zeros(int(self.mel_filters.shape[1]), 0)
        else:
            stft_input = torch.cat((self.history, complete))
        self.history = stft_input[-self.history.numel() :]
        spectrum = torch.stft(
            stft_input,
            n_fft=self.fft_window_samples,
            hop_length=self.hop_samples,
            window=self.window,
            center=False,
            return_complex=True,
        )
        mel = self.mel_filters.T @ spectrum.abs().square()
        log_mel = mel.clamp_min(LOG_MEL_FLOOR).log10()
        running_peak = torch.cummax(
            torch.cat((self.running_peak.view(1), log_mel.amax(dim=0))), dim=0
        ).values[1:]
        self.running_peak = running_peak[-1]
        log_mel = torch.maximum(log_mel, running_peak - LOG_MEL_DYNAMIC_RANGE)
        return (log_mel + 4.0) / 4.0


class Qwen3OmniD2AudioEncoder:
    """Encode each 80 ms frame as one token from the latest 104 mel frames and 104 attention tokens.

    History fills with the tower's native attention width; once full, a CUDA
    graph replays the same bounded window.
    """

    def __init__(
        self,
        audio_tower: Qwen3OmniMoeAudioEncoder,
        mel_filters: torch.Tensor,
        fft_window_samples: int,
        hop_samples: int,
    ) -> None:
        self.audio_tower = audio_tower
        self.mel_filters = mel_filters
        self.fft_window_samples = fft_window_samples
        self.hop_samples = hop_samples
        self.past_token_limit = ATTENTION_WINDOW_TOKENS - 1
        device = audio_tower.conv2d1.weight.device
        dtype = audio_tower.conv2d1.weight.dtype
        self.window = torch.zeros(
            (audio_tower.num_mel_bins, CONV_WINDOW_MEL_FRAMES),
            device=device,
            dtype=dtype,
        )
        attention = audio_tower.layers[0].self_attn
        cache_shape = (
            1,
            attention.num_heads,
            self.past_token_limit,
            attention.head_dim,
        )
        self.keys = [
            torch.zeros(cache_shape, device=device, dtype=dtype)
            for _ in audio_tower.layers
        ]
        self.values = [torch.zeros_like(key) for key in self.keys]
        slots = CONV_WINDOW_MEL_FRAMES // MEL_FRAMES_PER_TOKEN
        self.positional_row = audio_tower.positional_embedding.positional_embedding[
            slots - 1 : slots
        ]
        self.graph = torch.cuda.CUDAGraph()
        self.reset()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.forward(self.past_token_limit)
        self.reset()
        torch.cuda.synchronize(device)
        with (
            torch.autocast("cuda", dtype=torch.bfloat16),
            torch.cuda.graph(self.graph, capture_error_mode="thread_local"),
        ):
            self.graph_output = self.forward(self.past_token_limit)
        self.reset()

    def reset(self) -> None:
        self.window.zero_()
        for cache in (*self.keys, *self.values):
            cache.zero_()
        self.past_tokens = 0
        self.log_mel = CausalLogMel(
            self.mel_filters, self.fft_window_samples, self.hop_samples
        )

    def forward(self, past_tokens: int) -> torch.Tensor:
        """Run the window through the frontend and each layer, keeping the latest attention tokens."""
        tower = self.audio_tower
        hidden_states = self.window[None, None]
        for convolution in (tower.conv2d1, tower.conv2d2, tower.conv2d3):
            hidden_states = F.gelu(convolution(hidden_states))
        _, channels, frequencies, slots = hidden_states.shape
        tokens = tower.conv_out(
            hidden_states.permute(0, 3, 1, 2).reshape(slots, channels * frequencies)
        )
        hidden_states = tokens[-1:] + self.positional_row
        kept_tokens = min(past_tokens + 1, self.past_token_limit)
        for layer, keys, values in zip(
            tower.layers, self.keys, self.values, strict=True
        ):
            attention = layer.self_attn
            residual = hidden_states
            normalized = layer.self_attn_layer_norm(hidden_states)
            query, key, value = (
                projection(normalized)
                .reshape(1, attention.num_heads, -1)
                .transpose(0, 1)[None]
                for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
            )
            key = torch.cat((keys[..., :past_tokens, :], key), dim=-2)
            value = torch.cat((values[..., :past_tokens, :], value), dim=-2)
            attended = F.scaled_dot_product_attention(query, key, value)
            keys[..., :kept_tokens, :].copy_(key[..., -kept_tokens:, :])
            values[..., :kept_tokens, :].copy_(value[..., -kept_tokens:, :])
            hidden_states = residual + attention.out_proj(
                attended[0].transpose(0, 1).reshape(1, -1)
            )
            residual = hidden_states
            hidden_states = layer.fc2(
                layer.activation_fn(layer.fc1(layer.final_layer_norm(hidden_states)))
            )
            hidden_states = residual + hidden_states
        hidden_states = tower.ln_post(hidden_states)
        return tower.proj2(tower.act(tower.proj1(hidden_states)))

    def push(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode one 80 ms frame of 16 kHz float samples into [1, 2048] features."""
        mel = self.log_mel.push(waveform)
        assert mel.shape[1] == MEL_FRAMES_PER_TOKEN
        self.window[:, :-MEL_FRAMES_PER_TOKEN].copy_(
            self.window[:, MEL_FRAMES_PER_TOKEN:].clone()
        )
        self.window[:, -MEL_FRAMES_PER_TOKEN:].copy_(mel)
        if self.past_tokens < self.past_token_limit:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = self.forward(self.past_tokens)
            self.past_tokens += 1
            return output
        else:
            self.graph.replay()
            return self.graph_output.clone()
