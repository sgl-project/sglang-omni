# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RedAE decoding with incremental attention caches and fp32 overlap-add synthesis."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from transformers import DynamicCache, Qwen3Config, Qwen3Model
from transformers.cache_utils import DynamicSlidingWindowLayer

ENVELOPE_FLOOR = 1e-8
NOLA_ENVELOPE_FLOOR = 1e-11
MAX_SPECTRAL_MAGNITUDE = 1e2
WINDOW_SCALARS: dict[torch.device, torch.Tensor] = {}


class ISTFT(nn.Module):
    """Overlap-add synthesis with custom same padding or native center padding."""

    # note (kermit): Same padding trims the zero-envelope edges rejected by native ISTFT.
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        padding: Literal["center", "same"] = "same",
    ) -> None:
        super().__init__()
        assert padding in ["center", "same"], "Padding must be 'center' or 'same'."
        self.padding: Literal["center", "same"] = padding
        self.n_fft: int = n_fft
        self.hop_length: int = hop_length
        self.win_length: int = win_length
        self.window: torch.Tensor
        self.register_buffer("window", torch.hann_window(win_length))

    def overlap_add(
        self, windowed_frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_samples = (
            windowed_frames.shape[-1] - 1
        ) * self.hop_length + self.win_length
        waveform = torch.nn.functional.fold(
            windowed_frames,
            output_size=(1, output_samples),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, :]
        window_squared = (
            self.window.square()
            .expand(1, windowed_frames.shape[-1], -1)
            .transpose(1, 2)
        )
        window_envelope = torch.nn.functional.fold(
            window_squared,
            output_size=(1, output_samples),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()
        return waveform, window_envelope

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.padding == "center":
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            padding_samples = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")
        assert spec.dim() == 3, "Expected a 3D tensor as input"
        frames = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        frames = frames * self.window[None, :, None]
        waveform, envelope = self.overlap_add(frames)
        waveform = waveform[:, padding_samples:-padding_samples]
        envelope = envelope[padding_samples:-padding_samples]
        assert (envelope > NOLA_ENVELOPE_FLOOR).all()
        return waveform / envelope

    def forward_stream(
        self,
        spec: torch.Tensor,
        cache: torch.Tensor | None = None,
        last_chunk: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.padding == "same", "Padding must be same."
        assert (
            self.win_length % self.hop_length == 0
        ), f"{self.win_length} {self.hop_length}"
        padding_samples = (self.win_length - self.hop_length) // 2
        frames = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        frames = frames * self.window[None, :, None]
        if cache is not None:
            frames = torch.cat([cache, frames], dim=-1)
        else:
            pass
        cached_frame_count = self.win_length // self.hop_length - 1
        new_cache = frames[..., -cached_frame_count:]
        waveform, envelope = self.overlap_add(frames)
        # note (kermit): Streaming edges can have zero envelopes before the padding is trimmed.
        waveform = waveform / envelope.clamp_min(ENVELOPE_FLOOR)
        if cache is None:
            waveform = waveform[:, padding_samples:]
        else:
            waveform = waveform[:, (self.win_length - self.hop_length) :]
        if last_chunk:
            waveform = waveform[:, :-padding_samples]
        else:
            waveform = waveform[:, : -(self.win_length - self.hop_length)]
        return waveform, new_cache

    def flush_stream(self, cache: torch.Tensor) -> torch.Tensor:
        """Emit the terminal overlap tail without a zero-frame inverse FFT."""
        padding_samples = (self.win_length - self.hop_length) // 2
        waveform, envelope = self.overlap_add(cache)
        waveform = waveform / envelope.clamp_min(ENVELOPE_FLOOR)
        waveform = waveform[:, (self.win_length - self.hop_length) :]
        return waveform[:, :-padding_samples]


class ISTFTHead(nn.Module):
    """Predict complex spectra and synthesize full or incremental audio."""

    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: Literal["center", "same"] = "same",
    ) -> None:
        super().__init__()
        self.hop_length: int = hop_length
        self.out: nn.Linear = nn.Linear(dim, n_fft + 2)
        self.istft: ISTFT = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )

    def predict_spectrum(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # note (kermit): Phase and overlap-add stay fp32 to avoid accumulated waveform distortion.
        predicted_spectrum = self.out(hidden_states)
        predicted_spectrum = predicted_spectrum.transpose(1, 2)
        magnitude, phase = predicted_spectrum.chunk(2, dim=1)
        magnitude = torch.exp(magnitude)
        magnitude = torch.clip(magnitude, max=MAX_SPECTRAL_MAGNITUDE)
        phase_cosine = torch.cos(phase)
        phase_sine = torch.sin(phase)
        return magnitude * (phase_cosine + 1j * phase_sine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.istft(self.predict_spectrum(x))

    def forward_stream(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        last_chunk: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.istft.forward_stream(self.predict_spectrum(x), cache, last_chunk)


@dataclass
class RedAEDecoderState:
    """Per-request attention cache, windowed overlap frames, and decoded frame count."""

    qwen3_cache: DynamicCache | None = None
    istft_cache: torch.Tensor | None = None
    frames_seen: int = 0


def build_initialized_cache(
    config: Qwen3Config, device: torch.device, dtype: torch.dtype = torch.float32
) -> DynamicCache:
    """Initialize cache layers without synchronizing each sliding-window scalar."""
    cache = DynamicCache(config=config)
    window_scalar = WINDOW_SCALARS.get(device)
    if window_scalar is None:
        window_scalar = torch.tensor(
            config.sliding_window, dtype=torch.long, device=device
        )
        WINDOW_SCALARS[device] = window_scalar
    else:
        pass
    empty = torch.zeros(0, dtype=dtype, device=device)
    for layer in cache.layers:
        layer.dtype, layer.device = (dtype, device)
        layer.keys = empty
        layer.values = empty
        layer.is_initialized = True
        if isinstance(layer, DynamicSlidingWindowLayer):
            layer._sliding_window_tensor = window_scalar  # noqa: SLF001  # leading-underscore: Transformers cache attribute
        else:
            pass
    return cache


class RedAEAudioDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        upsample_rate: int = 2,
        audio_patch_size: int = 480,
        audio_sample_rate: int = 24000,
        hidden_size: int = 896,
        intermediate_size: int = 896 * 4,
        num_hidden_layers: int = 18,
        max_position_embeddings: int = 32768,
        max_window_layers: int = 0,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        sliding_window: int = 64,
        use_sliding_window: bool = True,
    ) -> None:
        super().__init__()
        self.upsample_rate: int = upsample_rate
        self.audio_patch_size: int = audio_patch_size
        self.audio_sample_rate: int = audio_sample_rate
        self.in_proj: nn.Linear = nn.Linear(in_dim, upsample_rate * hidden_size)
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
        self.istft_head: ISTFTHead = ISTFTHead(
            dim=hidden_size,
            n_fft=audio_patch_size * 4,
            hop_length=audio_patch_size,
            padding="same",
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Decode a full segment for independent streaming comparisons."""
        hidden_states = self.in_proj(xs)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], -1, self.qwen3_config.hidden_size
        )
        model_output = self.qwen3(inputs_embeds=hidden_states, attention_mask=None)
        return self.istft_head(model_output.last_hidden_state)

    def forward_stream(
        self,
        xs: torch.Tensor,
        cache: RedAEDecoderState | None = None,
        last_chunk: bool = False,
    ) -> tuple[torch.Tensor, RedAEDecoderState]:
        """Decode one chunk while retaining its attention and overlap state."""
        if cache is None:
            cache = RedAEDecoderState(
                build_initialized_cache(
                    self.qwen3_config,
                    device=xs.device,
                    dtype=self.in_proj.weight.dtype,
                )
            )
        else:
            pass
        hidden_states = self.in_proj(xs)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], -1, self.qwen3_config.hidden_size
        )
        model_output = self.qwen3.forward(
            inputs_embeds=hidden_states,
            use_cache=True,
            past_key_values=cache.qwen3_cache,
        )
        hidden_states = model_output.last_hidden_state
        waveform, istft_cache = self.istft_head.forward_stream(
            hidden_states, cache.istft_cache, last_chunk
        )
        new_cache = RedAEDecoderState(model_output.past_key_values, istft_cache)
        new_cache.frames_seen = cache.frames_seen + int(hidden_states.shape[1])
        return waveform, new_cache

    def flush_stream(self, cache: RedAEDecoderState) -> torch.Tensor:
        """Emit the terminal overlap tail after the final decoded chunk."""
        if cache.istft_cache is None:
            raise RuntimeError("RedAE flush_stream requires a decoded chunk first")
        else:
            pass
        return self.istft_head.istft.flush_stream(cache.istft_cache)

    def merge_states(self, states: list[RedAEDecoderState]) -> RedAEDecoderState:
        """Merge equal-progress request states into one batched state."""
        merged_cache = build_initialized_cache(
            self.qwen3_config,
            device=states[0].qwen3_cache.layers[0].keys.device,
            dtype=states[0].qwen3_cache.layers[0].keys.dtype,
        )
        for layer_index, layer in enumerate(merged_cache.layers):
            first = states[0].qwen3_cache.layers[layer_index]
            layer.keys = torch.cat(
                [state.qwen3_cache.layers[layer_index].keys for state in states], dim=0
            )
            layer.values = torch.cat(
                [state.qwen3_cache.layers[layer_index].values for state in states],
                dim=0,
            )
            if isinstance(layer, DynamicSlidingWindowLayer):
                layer.cumulative_length = first.cumulative_length
            else:
                pass
        merged = RedAEDecoderState(merged_cache)
        merged.istft_cache = torch.cat([state.istft_cache for state in states], dim=0)
        merged.frames_seen = states[0].frames_seen
        return merged

    def split_state(
        self, state: RedAEDecoderState, batch_size: int
    ) -> list[RedAEDecoderState]:
        """Split a batched state back into per-request states."""
        states: list[RedAEDecoderState] = []
        for row in range(batch_size):
            row_cache = DynamicCache(config=self.qwen3_config)
            for layer_index, layer in enumerate(state.qwen3_cache.layers):
                row_layer = copy.copy(layer)
                row_layer.keys = layer.keys[row : row + 1]
                row_layer.values = layer.values[row : row + 1]
                row_cache.layers[layer_index] = row_layer
            row_state = RedAEDecoderState(row_cache)
            row_state.istft_cache = state.istft_cache[row : row + 1]
            row_state.frames_seen = state.frames_seen
            states.append(row_state)
        return states


def rematerialize_istft_window(
    decoder: RedAEAudioDecoder, device: torch.device | str
) -> None:
    """Recompute the iSTFT Hann window after a meta-device materialization."""
    istft = decoder.istft_head.istft
    istft.window = torch.hann_window(istft.win_length, device=device)
