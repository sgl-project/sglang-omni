# SPDX-License-Identifier: MIT
# Derived from mlx-audio Qwen3-TTS (Copyright 2025 Prince Canuma and contributors).
"""ECAPA-TDNN speaker encoder: reference mel spectrogram -> x-vector.

Used by Qwen3-TTS Base voice cloning to condition the talker on a speaker
embedding alongside the in-context reference codes.

Tensors move in ``[batch, channels, time]`` (NCL) throughout, because that is
how ECAPA-TDNN is described and how the checkpoint is laid out; MLX ``Conv1d``
wants NLC, so each convolution transposes in and out.
"""

from __future__ import annotations

from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from .config import SpeakerEncoderConfig


def reflect_pad_time(x: mx.array, padding: int) -> mx.array:
    """Reflect-pad the time axis of an NLC tensor without repeating the edge."""
    if padding <= 0:
        return x
    left = x[:, 1 : padding + 1, :][:, ::-1, :]
    right = x[:, -(padding + 1) : -1, :][:, ::-1, :]
    return mx.concatenate([left, x, right], axis=1)


class TimeDelayNetBlock(nn.Module):
    """Dilated 1-D convolution with reflect "same" padding and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 1))
        x = reflect_pad_time(x, self.pad)
        out = self.conv(x)
        return nn.relu(mx.transpose(out, (0, 2, 1)))


class Res2NetBlock(nn.Module):
    """Res2Net multi-scale block: channel groups chained through TDNN blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.blocks = [
            TimeDelayNetBlock(
                in_channels // scale,
                out_channels // scale,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            for _ in range(scale - 1)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        chunks = mx.split(x, self.scale, axis=1)
        outputs = []
        carried = None
        for index, chunk in enumerate(chunks):
            if index == 0:
                carried = chunk
            elif index == 1:
                carried = self.blocks[index - 1](chunk)
            else:
                carried = self.blocks[index - 1](chunk + carried)
            outputs.append(carried)
        return mx.concatenate(outputs, axis=1)


class SqueezeExcitationBlock(nn.Module):
    """Channel attention from the time-averaged signal."""

    def __init__(self, in_channels: int, se_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, se_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(se_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        pooled = mx.mean(x, axis=2, keepdims=True)
        gate = mx.transpose(pooled, (0, 2, 1))
        gate = nn.relu(self.conv1(gate))
        gate = mx.sigmoid(self.conv2(gate))
        return x * mx.transpose(gate, (0, 2, 1))


class SqueezeExcitationRes2NetBlock(nn.Module):
    """TDNN -> Res2Net -> TDNN -> SE with a residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res2net_scale: int = 8,
        se_channels: int = 128,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TimeDelayNetBlock(in_channels, out_channels, 1, 1)
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TimeDelayNetBlock(out_channels, out_channels, 1, 1)
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)
        return x + residual


class AttentiveStatisticsPooling(nn.Module):
    """Pool over time into attention-weighted mean and standard deviation."""

    def __init__(self, channels: int, attention_channels: int = 128) -> None:
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.conv = nn.Conv1d(attention_channels, channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        batch, channels, length = x.shape

        mean = mx.mean(x, axis=2, keepdims=True)
        std = mx.sqrt(mx.var(x, axis=2, keepdims=True) + self.eps)
        attention = mx.concatenate(
            [
                x,
                mx.broadcast_to(mean, (batch, channels, length)),
                mx.broadcast_to(std, (batch, channels, length)),
            ],
            axis=1,
        )

        attention = mx.tanh(self.tdnn(attention))
        attention = mx.transpose(attention, (0, 2, 1))
        attention = self.conv(attention)
        attention = mx.transpose(attention, (0, 2, 1))
        attention = mx.softmax(attention, axis=2)

        mean = mx.sum(attention * x, axis=2, keepdims=True)
        var = mx.sum(attention * (x - mean) ** 2, axis=2, keepdims=True)
        std = mx.sqrt(mx.clip(var, self.eps, None))
        return mx.concatenate([mean, std], axis=1)


class Qwen3TTSSpeakerEncoder(nn.Module):
    """ECAPA-TDNN x-vector encoder."""

    def __init__(self, config: SpeakerEncoderConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.enc_channels

        self.blocks = [
            TimeDelayNetBlock(
                config.mel_dim,
                channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        ]
        for index in range(1, len(channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    channels[index - 1],
                    channels[index],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[index],
                    dilation=config.enc_dilations[index],
                )
            )

        # Multi-layer feature aggregation consumes the concatenated SE-Res2Net
        # outputs, so the declared final width must equal their sum (the
        # released config is [512, 512, 512, 512, 1536]).
        aggregated = sum(channels[1:-1])
        if aggregated != channels[-1]:
            raise ValueError(
                "Speaker encoder enc_channels[-1] must equal the sum of the "
                f"SE-Res2Net widths: got {channels[-1]} for {channels[1:-1]} "
                f"(sum {aggregated})"
            )
        self.mfa = TimeDelayNetBlock(
            channels[-1],
            channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=config.enc_attention_channels,
        )
        self.fc = nn.Conv1d(channels[-1] * 2, config.enc_dim, kernel_size=1)

    def __call__(self, mels: mx.array) -> mx.array:
        """``mels`` is ``[batch, frames, mel_dim]``; returns ``[batch, enc_dim]``."""
        x = mx.transpose(mels, (0, 2, 1))

        hidden_states = []
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)

        # Multi-layer feature aggregation over the SE-Res2Net outputs only.
        x = self.mfa(mx.concatenate(hidden_states[1:], axis=1))
        x = self.asp(x)

        x = mx.transpose(x, (0, 2, 1))
        x = self.fc(x)
        x = mx.transpose(x, (0, 2, 1))
        return mx.squeeze(x, axis=-1)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Keep the ``speaker_encoder.`` subtree, unprefixed.

        Conv layout is fixed separately by
        :func:`~sglang_omni.models.qwen3_tts.mlx.weights.align_conv_weights`,
        which derives the permutation from each module's type instead of
        guessing from dimension sizes.
        """
        prefix = "speaker_encoder."
        if not any(key.startswith(prefix) for key in weights):
            return dict(weights)
        return {
            key[len(prefix) :]: value
            for key, value in weights.items()
            if key.startswith(prefix)
        }
