# SPDX-License-Identifier: Apache-2.0
"""Native AudioVAE, including request-local interpolation and ISTFT overlap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen2 import ModelArgs, Qwen2Model


@dataclass
class AudioKVCache:
    keys: mx.array | None = None
    values: mx.array | None = None
    offset: int = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        self.offset += keys.shape[2]
        if self.keys is not None:
            keys = mx.concatenate((self.keys, keys), axis=2)
            values = mx.concatenate((self.values, values), axis=2)
        self.keys, self.values = keys, values
        return keys, values


class AudioQwen2Model(Qwen2Model):
    def __init__(self, config: dict[str, Any]) -> None:
        from transformers import Qwen2Config

        resolved = Qwen2Config(**config)
        args = resolved.to_dict()
        args["rope_theta"] = resolved.rope_parameters["rope_theta"]
        args["rope_scaling"] = resolved.rope_parameters
        super().__init__(ModelArgs.from_dict(args))
        self._windows = [
            resolved.sliding_window if kind == "sliding_attention" else None
            for kind in resolved.layer_types
        ]

    def __call__(
        self, inputs_embeds: mx.array, cache: list[AudioKVCache] | None = None
    ) -> mx.array:
        x = inputs_embeds
        for index, layer in enumerate(self.layers):
            entry = None if cache is None else cache[index]
            past = 0 if entry is None or entry.keys is None else entry.keys.shape[2]
            query = mx.arange(past, past + x.shape[1])[:, None]
            key = mx.arange(past + x.shape[1])[None, :]
            mask = query >= key
            window = self._windows[index]
            if window is not None:
                mask = mask & (query - key < window)
            x = layer(x, mask=mask, cache=entry)
            if entry is not None and window is not None:
                entry.keys = entry.keys[:, :, -window:]
                entry.values = entry.values[:, :, -window:]
        return self.norm(x)

    def make_cache(self) -> list[AudioKVCache]:
        return [AudioKVCache() for _ in self.layers]


@dataclass
class UpsampleState:
    pending: mx.array
    left: mx.array | None = None


class StreamingLinearUpsample(nn.Module):
    def __init__(self, scale_factor: int) -> None:
        super().__init__()
        self.scale_factor = scale_factor

    def interpolate(self, x: mx.array) -> mx.array:
        positions = (mx.arange(x.shape[1] * self.scale_factor) + 0.5) / self.scale_factor - 0.5
        positions = mx.clip(positions, 0, x.shape[1] - 1)
        left = mx.floor(positions).astype(mx.int32)
        right = mx.minimum(left + 1, x.shape[1] - 1)
        weight = (positions - left).astype(x.dtype)[None, :, None]
        return x[:, left] * (1 - weight) + x[:, right] * weight

    def __call__(
        self, x: mx.array, state: UpsampleState | None, *, last_chunk: bool
    ) -> tuple[mx.array | None, UpsampleState | None]:
        if state is None:
            if last_chunk:
                return self.interpolate(x), None
            return None, UpsampleState(x)
        parts = [state.pending, x[:, :1]]
        start = 0
        if state.left is not None:
            parts.insert(0, state.left)
            start = self.scale_factor
        previous = self.interpolate(mx.concatenate(parts, axis=1))
        previous = previous[:, start:start + state.pending.shape[1] * self.scale_factor]
        left = state.pending[:, -1:]
        if last_chunk:
            tail = self.interpolate(mx.concatenate((left, x), axis=1))[:, self.scale_factor:]
            return mx.concatenate((previous, tail), axis=1), None
        return previous, UpsampleState(x, left)


class ISTFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = 0.5 - 0.5 * mx.cos(2 * mx.pi * mx.arange(n_fft) / n_fft)

    def __call__(
        self,
        spectrum: mx.array,
        *,
        overlap: tuple[mx.array, mx.array] | None = None,
        streaming: bool = False,
        last_chunk: bool = False,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        batch, frames, _ = spectrum.shape
        window = self.window.astype(mx.float32)
        inverse = mx.fft.irfft(spectrum, n=self.n_fft, axis=-1) * window
        size = (frames - 1) * self.hop_length + self.n_fft
        indices = mx.arange(frames)[:, None] * self.hop_length + mx.arange(self.n_fft)[None]
        audio = mx.zeros((batch, size), dtype=mx.float32)
        audio = audio.at[:, indices.reshape(-1)].add(inverse.reshape(batch, -1))
        envelope = mx.zeros((1, size), dtype=mx.float32)
        window_frames = mx.broadcast_to(window * window, (frames, self.n_fft))
        envelope = envelope.at[:, indices.reshape(-1)].add(window_frames.reshape(1, -1))
        buffer_len = self.n_fft - self.hop_length
        pad = buffer_len // 2
        if not streaming:
            return audio[:, pad:-pad] / envelope[:, pad:-pad], None
        if overlap is None:
            audio, envelope = audio[:, pad:], envelope[:, pad:]
        else:
            audio[:, :buffer_len] += overlap[0]
            envelope[:, :buffer_len] += overlap[1]
        next_overlap = (audio[:, -buffer_len:], envelope[:, -buffer_len:])
        end = -pad if last_chunk else -buffer_len
        return audio[:, :end] / envelope[:, :end], None if last_chunk else next_overlap


class ISTFTHead(nn.Module):
    def __init__(self, hidden_size: int, hop_length: int) -> None:
        super().__init__()
        self.out = nn.Linear(hidden_size, hop_length * 4 + 2)
        self.istft = ISTFT(hop_length * 4, hop_length)

    def __call__(
        self,
        x: mx.array,
        *,
        overlap: tuple[mx.array, mx.array] | None = None,
        streaming: bool = False,
        last_chunk: bool = False,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        mag, phase = mx.split(self.out(x).astype(mx.float32), 2, axis=-1)
        mag = mx.minimum(mx.exp(mag), 100)
        spectrum = mag * (mx.cos(phase) + 1j * mx.sin(phase))
        return self.istft(spectrum, overlap=overlap, streaming=streaming, last_chunk=last_chunk)


class Encoder(nn.Module):
    def __init__(self, config: dict[str, Any], patch_size: int) -> None:
        super().__init__()
        backbone = config["backbone"]
        hidden = backbone["hidden_size"]
        self.encoder = AudioQwen2Model(backbone)
        self.input_dim = config["input_dim"]
        self.hop_size = config.get("hop_size", 320)
        self.patch_size = patch_size
        self.fc1 = nn.Linear(self.input_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, config["latent_dim"] * 2)
        self.norm = nn.LayerNorm(hidden)
        if patch_size != -1:
            aggregator = dict(backbone, num_hidden_layers=4)
            if aggregator.get("layer_types") is not None:
                aggregator["layer_types"] = aggregator["layer_types"][:4]
            self.aggregator = AudioQwen2Model(aggregator)
            self.cls_embed = mx.zeros((1, 1, hidden))

    def __call__(self, waveform: mx.array) -> mx.array:
        batch, length = waveform.shape
        count = (length + self.hop_size - 1) // self.hop_size
        needed = (count - 1) * self.hop_size + self.input_dim
        waveform = mx.pad(waveform, ((0, 0), (0, max(0, needed - length))))
        indices = mx.arange(count)[:, None] * self.hop_size + mx.arange(self.input_dim)[None]
        frames = waveform[:, indices]
        x = self.encoder(self.fc2(self.fc1(frames.astype(self.fc1.weight.dtype))))
        if self.patch_size != -1:
            padding = (-x.shape[1]) % self.patch_size
            x = mx.pad(x, ((0, 0), (0, padding), (0, 0)))
            hidden = x.shape[-1]
            x = x.reshape(-1, self.patch_size, hidden)
            cls = mx.broadcast_to(self.cls_embed, (x.shape[0], 1, hidden))
            x = mx.concatenate((x, cls.astype(x.dtype)), axis=1).reshape(batch, -1, hidden)
            x = self.aggregator(x).reshape(batch, -1, self.patch_size + 1, hidden)[:, :, -1]
        return self.fc3(x)


@dataclass
class AudioDecoderState:
    cache: list[AudioKVCache]
    upsample: UpsampleState | None = None
    overlap: tuple[mx.array, mx.array] | None = None


class Decoder(nn.Module):
    def __init__(self, config: dict[str, Any], patch_size: int) -> None:
        super().__init__()
        hidden = config["backbone"]["hidden_size"]
        self.decoder = AudioQwen2Model(config["backbone"])
        self.fc1 = nn.Linear(config["latent_dim"], hidden)
        self.head = ISTFTHead(hidden, config["output_dim"])
        self.patch_size = patch_size
        if patch_size != -1:
            self.upsampling = StreamingLinearUpsample(patch_size)

    def __call__(
        self, latent: mx.array, *, state: AudioDecoderState | None = None,
        streaming: bool = False, last_chunk: bool = True,
    ) -> tuple[mx.array, AudioDecoderState | None]:
        if streaming and state is None:
            state = AudioDecoderState(self.decoder.make_cache())
        x = self.fc1(latent.astype(self.fc1.weight.dtype))
        if self.patch_size != -1:
            if streaming:
                x, state.upsample = self.upsampling(x, state.upsample, last_chunk=last_chunk)
                if x is None:
                    return mx.zeros((latent.shape[0], 0)), state
            else:
                x = self.upsampling.interpolate(x)
        x = self.decoder(x, cache=state.cache if streaming else None)
        audio, overlap = self.head(
            x, overlap=state.overlap if streaming else None,
            streaming=streaming, last_chunk=last_chunk,
        )
        if streaming:
            state.overlap = overlap
        return audio, None if last_chunk else state


class AudioVAE(nn.Module):
    def __init__(
        self, config: dict[str, Any], *, component: Literal["encoder", "decoder"]
    ) -> None:
        super().__init__()
        self.config = config
        if config["sample_rate"] != 44100 or config.get("semantic_module_kwargs") is not None:
            raise ValueError("Ming AudioVAE requires 44.1 kHz audio without semantic modules")
        if component == "encoder":
            self.encoder = Encoder(config["enc_kwargs"], config["patch_size"])
        elif component == "decoder":
            self.decoder = Decoder(config["dec_kwargs"], config["patch_size"])
        else:
            raise ValueError("AudioVAE component must be encoder or decoder")

    def encode_latent(self, waveform: mx.array, *, noise: mx.array | None = None) -> mx.array:
        mean, scale = mx.split(self.encoder(waveform), 2, axis=-1)
        std = nn.softplus(scale) + 1e-4
        if noise is None:
            noise = mx.random.normal(mean.shape, dtype=mean.dtype)
        return mean + std * noise
