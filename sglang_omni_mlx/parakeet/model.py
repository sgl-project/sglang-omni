# SPDX-License-Identifier: Apache-2.0
"""Parakeet in plain MLX: FastConformer encoder, CTC head or RNN-T/TDT transducer, and loading.

Module and parameter names follow the checkpoint, so its model.safetensors
loads directly; only convolution weights change axis order, from the
checkpoint's channels-first layout to MLX's channels-last one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sglang_omni_mlx.checkpoint import load_weights, read_weights
from sglang_omni_mlx.transcription import CancelCheck, TranscriptionCancelled

ARCHITECTURES = ("ParakeetForCTC", "ParakeetForRNNT", "ParakeetForTDT")
# Finite instead of -inf so fully padded query rows stay finite; a NaN
# there would leak into valid frames through the depthwise convolution.
ATTENTION_MASK_VALUE = -1e9


@dataclass(frozen=True)
class EncoderConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_act: str
    attention_bias: bool
    convolution_bias: bool
    conv_kernel_size: int
    subsampling_factor: int
    subsampling_conv_channels: int
    subsampling_conv_kernel_size: int
    subsampling_conv_stride: int
    num_mel_bins: int
    scale_input: bool

    @classmethod
    def from_dict(cls, config: Mapping[str, object]) -> EncoderConfig:
        defaults = {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "intermediate_size": 4096,
            "hidden_act": "silu",
            "attention_bias": True,
            "convolution_bias": True,
            "conv_kernel_size": 9,
            "subsampling_factor": 8,
            "subsampling_conv_channels": 256,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "num_mel_bins": 80,
            "scale_input": True,
        }
        return cls(
            **{name: config.get(name, value) for name, value in defaults.items()}
        )


@dataclass(frozen=True)
class ParakeetConfig:
    architecture: str
    encoder: EncoderConfig
    vocab_size: int
    pad_token_id: int | None
    blank_token_id: int | None
    decoder_hidden_size: int
    num_decoder_layers: int
    hidden_act: str
    max_symbols_per_step: int
    durations: tuple[int, ...]

    @property
    def is_ctc(self) -> bool:
        return self.architecture == "ParakeetForCTC"

    @property
    def is_tdt(self) -> bool:
        return self.architecture == "ParakeetForTDT"

    @classmethod
    def from_dict(cls, config: Mapping[str, object]) -> ParakeetConfig:
        architectures = list(config.get("architectures") or ())
        matches = [a for a in architectures if a in ARCHITECTURES]
        if not matches:
            raise ValueError(
                f"Parakeet supports {list(ARCHITECTURES)} checkpoints, "
                f"got {architectures}"
            )
        else:
            pass
        architecture = matches[0]
        durations = tuple(config.get("durations") or ())
        if architecture == "ParakeetForTDT" and not durations:
            raise ValueError("Parakeet TDT config is missing its durations")
        else:
            pass
        return cls(
            architecture=architecture,
            encoder=EncoderConfig.from_dict(config["encoder_config"]),
            vocab_size=int(config["vocab_size"]),
            pad_token_id=config.get("pad_token_id"),
            blank_token_id=config.get("blank_token_id"),
            decoder_hidden_size=int(config.get("decoder_hidden_size", 640)),
            num_decoder_layers=int(config.get("num_decoder_layers", 2)),
            hidden_act=str(config.get("hidden_act", "relu")),
            max_symbols_per_step=int(config.get("max_symbols_per_step", 10)),
            durations=durations,
        )


def activation(name: str):
    if name == "silu":
        return nn.silu
    elif name == "relu":
        return nn.relu
    else:
        raise ValueError(f"Unsupported Parakeet activation {name!r}")


def subsampled_lengths(lengths: mx.array, config: EncoderConfig) -> mx.array:
    """Frame count after the strided subsampling convolutions."""
    kernel = config.subsampling_conv_kernel_size
    padding = (kernel - 1) // 2
    for _ in range(int(math.log2(config.subsampling_factor))):
        lengths = (lengths + 2 * padding - kernel) // config.subsampling_conv_stride + 1
    return lengths


def relative_position_embeddings(length: int, hidden_size: int) -> mx.array:
    """Interleaved sin/cos embeddings for relative offsets length-1 .. -(length-1)."""
    inv_freq = 1.0 / (
        10000.0 ** (mx.arange(0, hidden_size, 2, dtype=mx.float32) / hidden_size)
    )
    positions = mx.arange(length - 1, -length, -1, dtype=mx.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    embeddings = mx.stack([mx.sin(freqs), mx.cos(freqs)], axis=-1)
    return embeddings.reshape(1, 2 * length - 1, hidden_size)


def relative_shift(scores: mx.array) -> mx.array:
    """Transformer-XL shift that aligns each query with its relative offsets."""
    batch, heads, queries, positions = scores.shape
    scores = mx.pad(scores, [(0, 0), (0, 0), (0, 0), (1, 0)])
    scores = scores.reshape(batch, heads, positions + 1, queries)
    return scores[:, :, 1:].reshape(batch, heads, queries, positions)


class FeedForward(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.linear1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.attention_bias
        )
        self.linear2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.attention_bias
        )
        self.activation = activation(config.hidden_act)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.linear2(self.activation(self.linear1(hidden)))


class ConvolutionModule(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        channels = config.hidden_size
        kernel = config.conv_kernel_size
        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, 1, bias=config.convolution_bias
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel,
            padding=(kernel - 1) // 2,
            groups=channels,
            bias=config.convolution_bias,
        )
        self.norm = nn.BatchNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, 1, bias=config.convolution_bias
        )
        self.activation = activation(config.hidden_act)

    def __call__(self, hidden: mx.array, valid: mx.array | None) -> mx.array:
        hidden = self.pointwise_conv1(hidden)
        value, gate = mx.split(hidden, 2, axis=-1)
        hidden = value * mx.sigmoid(gate)
        if valid is not None:
            hidden = mx.where(valid[:, :, None], hidden, 0.0)
        else:
            pass
        hidden = self.activation(self.norm(self.depthwise_conv(hidden)))
        return self.pointwise_conv2(hidden)


class RelPositionAttention(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = hidden // self.num_heads
        self.scale = self.head_dim**-0.5
        bias = config.attention_bias
        self.q_proj = nn.Linear(hidden, hidden, bias=bias)
        self.k_proj = nn.Linear(hidden, hidden, bias=bias)
        self.v_proj = nn.Linear(hidden, hidden, bias=bias)
        self.o_proj = nn.Linear(hidden, hidden, bias=bias)
        self.relative_k_proj = nn.Linear(hidden, hidden, bias=False)
        self.bias_u = mx.zeros((self.num_heads, self.head_dim))
        self.bias_v = mx.zeros((self.num_heads, self.head_dim))

    def heads(self, hidden: mx.array) -> mx.array:
        batch, length, _ = hidden.shape
        return hidden.reshape(batch, length, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

    def __call__(
        self, hidden: mx.array, positions: mx.array, mask: mx.array | None
    ) -> mx.array:
        batch, length, _ = hidden.shape
        query = self.heads(self.q_proj(hidden))
        key = self.heads(self.k_proj(hidden))
        value = self.heads(self.v_proj(hidden))
        relative_key = (
            self.relative_k_proj(positions)
            .reshape(1, -1, self.num_heads, self.head_dim)
            .transpose(0, 2, 3, 1)
        )
        position_scores = (query + self.bias_v[None, :, None, :]) @ relative_key
        position_scores = relative_shift(position_scores)[..., :length] * self.scale
        if mask is not None:
            position_scores = position_scores + mask
        else:
            pass
        output = mx.fast.scaled_dot_product_attention(
            query + self.bias_u[None, :, None, :],
            key,
            value,
            scale=self.scale,
            mask=position_scores.astype(query.dtype),
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class ConformerBlock(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        hidden = config.hidden_size
        self.feed_forward1 = FeedForward(config)
        self.self_attn = RelPositionAttention(config)
        self.conv = ConvolutionModule(config)
        self.feed_forward2 = FeedForward(config)
        self.norm_feed_forward1 = nn.LayerNorm(hidden)
        self.norm_self_att = nn.LayerNorm(hidden)
        self.norm_conv = nn.LayerNorm(hidden)
        self.norm_feed_forward2 = nn.LayerNorm(hidden)
        self.norm_out = nn.LayerNorm(hidden)

    def __call__(
        self,
        hidden: mx.array,
        positions: mx.array,
        mask: mx.array | None,
        valid: mx.array | None,
    ) -> mx.array:
        hidden = hidden + 0.5 * self.feed_forward1(self.norm_feed_forward1(hidden))
        hidden = hidden + self.self_attn(self.norm_self_att(hidden), positions, mask)
        hidden = hidden + self.conv(self.norm_conv(hidden), valid)
        hidden = hidden + 0.5 * self.feed_forward2(self.norm_feed_forward2(hidden))
        return self.norm_out(hidden)


class SubsamplingConv2D(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        kernel = config.subsampling_conv_kernel_size
        stride = config.subsampling_conv_stride
        channels = config.subsampling_conv_channels
        padding = (kernel - 1) // 2
        num_layers = int(math.log2(config.subsampling_factor))
        # ReLU modules occupy slots 1, 4 and 7, as in the checkpoint's numbering.
        # Each entry pairs a layer with whether it downsamples time.
        layers: list[tuple[nn.Module, bool]] = [
            (nn.Conv2d(1, channels, kernel, stride=stride, padding=padding), True),
            (nn.ReLU(), False),
        ]
        for _ in range(num_layers - 1):
            layers.append(
                (
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel,
                        stride=stride,
                        padding=padding,
                        groups=channels,
                    ),
                    True,
                )
            )
            layers.append((nn.Conv2d(channels, channels, 1), False))
            layers.append((nn.ReLU(), False))
        self.layers = [layer for layer, _ in layers]
        self.strided = tuple(strided for _, strided in layers)
        self.config = config
        out_bins = config.num_mel_bins // (stride**num_layers)
        self.linear = nn.Linear(channels * out_bins, config.hidden_size)

    def __call__(
        self, features: mx.array, lengths: mx.array | None
    ) -> tuple[mx.array, mx.array | None]:
        kernel = self.config.subsampling_conv_kernel_size
        stride = self.config.subsampling_conv_stride
        padding = (kernel - 1) // 2
        hidden = features[..., None]
        for layer, strided in zip(self.layers, self.strided):
            hidden = layer(hidden)
            if lengths is not None and isinstance(layer, nn.Conv2d):
                if strided:
                    lengths = (lengths + 2 * padding - kernel) // stride + 1
                else:
                    pass
                frames = mx.arange(hidden.shape[1])[None, :] < lengths[:, None]
                hidden = hidden * frames[:, :, None, None].astype(hidden.dtype)
            else:
                pass
        batch, length, bins, channels = hidden.shape
        hidden = hidden.transpose(0, 1, 3, 2).reshape(batch, length, channels * bins)
        return self.linear(hidden), lengths


class ParakeetEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.input_scale = math.sqrt(config.hidden_size) if config.scale_input else 1.0
        self.subsampling = SubsamplingConv2D(config)
        self.layers = [ConformerBlock(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self, features: mx.array, lengths: mx.array | None
    ) -> tuple[mx.array, mx.array]:
        """Encode [batch, frames, mels] features; lengths None means unpadded."""
        hidden, _ = self.subsampling(features, lengths)
        hidden = hidden * self.input_scale
        batch, length, _ = hidden.shape
        positions = relative_position_embeddings(
            length, self.config.hidden_size
        ).astype(hidden.dtype)
        if lengths is None:
            out_lengths = mx.full((batch,), length, dtype=mx.int32)
            mask = valid = None
        else:
            out_lengths = subsampled_lengths(lengths, self.config)
            valid = mx.arange(length)[None, :] < out_lengths[:, None]
            pair = valid[:, None, :, None] & valid[:, None, None, :]
            mask = mx.where(pair, 0.0, ATTENTION_MASK_VALUE).astype(hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, positions, mask, valid)
        return hidden, out_lengths


class LSTM(nn.Module):
    """Stacked unidirectional LSTM with the checkpoint's parameter names and gate order."""

    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        for layer in range(num_layers):
            setattr(
                self, f"weight_ih_l{layer}", mx.zeros((4 * hidden_size, hidden_size))
            )
            setattr(
                self, f"weight_hh_l{layer}", mx.zeros((4 * hidden_size, hidden_size))
            )
            setattr(self, f"bias_ih_l{layer}", mx.zeros((4 * hidden_size,)))
            setattr(self, f"bias_hh_l{layer}", mx.zeros((4 * hidden_size,)))

    def step(
        self, inputs: mx.array, hidden: mx.array, cell: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """One time step; hidden and cell are [layers, batch, size]."""
        new_hidden, new_cell = [], []
        for layer in range(self.num_layers):
            gates = (
                inputs @ self[f"weight_ih_l{layer}"].T
                + self[f"bias_ih_l{layer}"]
                + hidden[layer] @ self[f"weight_hh_l{layer}"].T
                + self[f"bias_hh_l{layer}"]
            )
            input_gate, forget_gate, candidate, output_gate = mx.split(
                gates, 4, axis=-1
            )
            layer_cell = mx.sigmoid(forget_gate) * cell[layer] + mx.sigmoid(
                input_gate
            ) * mx.tanh(candidate)
            inputs = mx.sigmoid(output_gate) * mx.tanh(layer_cell)
            new_hidden.append(inputs)
            new_cell.append(layer_cell)
        return inputs, mx.stack(new_hidden), mx.stack(new_cell)


class PredictionNetwork(nn.Module):
    def __init__(self, config: ParakeetConfig):
        super().__init__()
        size = config.decoder_hidden_size
        self.embedding = nn.Embedding(config.vocab_size, size)
        self.lstm = LSTM(size, config.num_decoder_layers)
        self.decoder_projector = nn.Linear(size, size)

    def __call__(
        self, tokens: mx.array, hidden: mx.array, cell: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        output, hidden, cell = self.lstm.step(self.embedding(tokens), hidden, cell)
        return self.decoder_projector(output), hidden, cell


class JointNetwork(nn.Module):
    def __init__(self, config: ParakeetConfig):
        super().__init__()
        width = config.vocab_size + len(config.durations)
        self.head = nn.Linear(config.decoder_hidden_size, width)
        self.activation = activation(config.hidden_act)

    def __call__(self, encoder_frame: mx.array, decoder_output: mx.array) -> mx.array:
        return self.head(self.activation(encoder_frame + decoder_output))


class ParakeetModel(nn.Module):
    def __init__(self, config: ParakeetConfig):
        super().__init__()
        self.config = config
        self.encoder = ParakeetEncoder(config.encoder)
        hidden = config.encoder.hidden_size
        if config.is_ctc:
            self.ctc_head = nn.Conv1d(hidden, config.vocab_size, 1)
        else:
            self.encoder_projector = nn.Linear(hidden, config.decoder_hidden_size)
            self.decoder = PredictionNetwork(config)
            self.joint = JointNetwork(config)

    def greedy_decode(
        self, features: mx.array, lengths: mx.array | None, cancel: CancelCheck
    ) -> list[list[int]]:
        """Token ids per utterance: raw CTC frames, or emitted transducer tokens."""
        hidden, out_lengths = self.encoder(features, lengths)
        if self.config.is_ctc:
            frame_ids = mx.argmax(self.ctc_head(hidden), axis=-1)
            mx.eval(frame_ids, out_lengths)
            ids, valid = np.array(frame_ids), np.array(out_lengths)
            return [ids[row, : valid[row]].tolist() for row in range(ids.shape[0])]
        else:
            return self.transducer_decode(
                self.encoder_projector(hidden), out_lengths, cancel
            )

    def transducer_decode(
        self, encoder_frames: mx.array, lengths: mx.array, cancel: CancelCheck
    ) -> list[list[int]]:
        """Batched greedy RNN-T / TDT decoding in lockstep across the batch.

        Each step scores the current frame of every unfinished utterance, emits
        non-blank tokens, advances the decoder state only where a token was
        emitted, and moves each frame pointer: by one on blank for RNN-T, by
        the predicted duration for TDT. max_symbols_per_step consecutive
        emissions on one frame force a one-frame advance.
        """
        config = self.config
        batch, frames, _ = encoder_frames.shape
        blank = int(config.blank_token_id)
        vocab = config.vocab_size
        durations = np.array(config.durations, dtype=np.int64)
        layers, size = config.num_decoder_layers, config.decoder_hidden_size
        dtype = encoder_frames.dtype
        hidden = mx.zeros((layers, batch, size), dtype=dtype)
        cell = mx.zeros((layers, batch, size), dtype=dtype)
        decoder_output, hidden, cell = self.decoder(
            mx.full((batch,), blank, dtype=mx.int32), hidden, cell
        )
        valid_lengths = np.array(lengths, dtype=np.int64)
        frame_index = np.zeros(batch, dtype=np.int64)
        symbols_on_frame = np.zeros(batch, dtype=np.int64)
        emitted: list[list[int]] = [[] for _ in range(batch)]
        rows = mx.arange(batch)
        while True:
            if cancel.is_set():
                raise TranscriptionCancelled()
            else:
                pass
            active = frame_index < valid_lengths
            if not active.any():
                break
            else:
                pass
            current = mx.array(np.minimum(frame_index, frames - 1))
            logits = self.joint(encoder_frames[rows, current], decoder_output)
            tokens = mx.argmax(logits[:, :vocab], axis=-1)
            if config.is_tdt:
                duration_ids = mx.argmax(logits[:, vocab:], axis=-1)
                mx.eval(tokens, duration_ids)
                step_durations = durations[np.array(duration_ids)]
            else:
                mx.eval(tokens)
                step_durations = None
            token_ids = np.array(tokens).astype(np.int64)
            is_blank = token_ids == blank
            emit = active & ~is_blank
            if emit.any():
                for row in np.flatnonzero(emit):
                    emitted[row].append(int(token_ids[row]))
                new_output, new_hidden, new_cell = self.decoder(tokens, hidden, cell)
                keep = mx.array(emit)
                decoder_output = mx.where(keep[:, None], new_output, decoder_output)
                hidden = mx.where(keep[None, :, None], new_hidden, hidden)
                cell = mx.where(keep[None, :, None], new_cell, cell)
            else:
                pass
            if step_durations is None:
                advance = is_blank.astype(np.int64)
            else:
                advance = np.where(is_blank & (step_durations == 0), 1, step_durations)
            symbols_on_frame = np.where(advance > 0, 0, symbols_on_frame + 1)
            forced = symbols_on_frame >= config.max_symbols_per_step
            advance = np.where(forced, 1, advance)
            symbols_on_frame = np.where(forced, 0, symbols_on_frame)
            frame_index = np.where(active, frame_index + advance, frame_index)
        return emitted


def checkpoint_weights(weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
    """Checkpoint weights in this module tree's layout."""
    converted: dict[str, mx.array] = {}
    for name, weight in weights.items():
        if name.endswith("num_batches_tracked"):
            continue
        elif weight.ndim == 4:
            # Conv2d: [out, in, height, width] to [out, height, width, in].
            converted[name] = weight.transpose(0, 2, 3, 1)
        elif weight.ndim == 3:
            # Conv1d: [out, in, kernel] to [out, kernel, in].
            converted[name] = weight.transpose(0, 2, 1)
        else:
            converted[name] = weight
    return converted


def load_parakeet(model_directory: Path, dtype: mx.Dtype) -> ParakeetModel:
    """Build the model from a Parakeet checkpoint directory, cast to dtype."""
    config = ParakeetConfig.from_dict(
        json.loads((model_directory / "config.json").read_text())
    )
    model = ParakeetModel(config)
    load_weights(model, checkpoint_weights(read_weights(model_directory)), None)
    model.set_dtype(dtype)
    model.eval()
    mx.eval(model.parameters())
    return model
