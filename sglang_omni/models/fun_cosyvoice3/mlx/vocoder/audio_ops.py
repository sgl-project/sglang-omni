# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Prince Canuma and contributors.
# Derived from Blaizzy/mlx-audio CosyVoice3 PR #861 (commit 5272f213f8cc).
# Based on FunAudioLLM/CosyVoice (Apache-2.0, Copyright 2024-2025 Alibaba Inc).
# Modified for the non-streaming sglang-omni vocoder contract.

"""Small MLX audio primitives used by the native HiFT decoder."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def hann_window_periodic(size: int) -> mx.array:
    """Create the periodic Hann window used by ``torch.stft``."""
    return mx.array(
        [0.5 * (1 - math.cos(2 * math.pi * index / size)) for index in range(size)]
    )


class Snake(nn.Module):
    """Snake activation with the parameter layout used by the checkpoint."""

    def __init__(self, channels: int, alpha_logscale: bool = False) -> None:
        super().__init__()
        self.alpha = mx.ones(channels)
        self.alpha_logscale = alpha_logscale

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha.reshape(1, -1, 1)
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
        abs_alpha = mx.abs(alpha)
        clamped = mx.sign(alpha) * mx.maximum(abs_alpha, 1e-4)
        clamped = mx.where(abs_alpha < 1e-9, 1e-4, clamped)
        return x + mx.sin(x * alpha) ** 2 / clamped


def stft(
    x: mx.array,
    n_fft: int,
    hop_length: int,
    window: mx.array,
) -> tuple[mx.array, mx.array]:
    """Return real and imaginary STFT components with Torch-style centering."""
    batch_size = x.shape[0]
    pad_length = n_fft // 2
    left_pad = x[:, 1 : pad_length + 1][:, ::-1]
    right_pad = x[:, -(pad_length + 1) : -1][:, ::-1]
    padded = mx.concatenate([left_pad, x, right_pad], axis=1)

    frame_count = (padded.shape[1] - n_fft) // hop_length + 1
    starts = mx.arange(frame_count) * hop_length
    indices = starts[:, None] + mx.arange(n_fft)[None, :]
    frames = mx.take(padded, indices.flatten(), axis=1).reshape(
        batch_size, frame_count, n_fft
    )
    frames = mx.swapaxes(frames, 1, 2) * window.reshape(1, -1, 1)
    spectrum = mx.fft.fft(frames, axis=1)
    positive = spectrum[:, : n_fft // 2 + 1, :]
    return mx.real(positive), mx.imag(positive)


def istft(
    magnitude: mx.array,
    phase: mx.array,
    n_fft: int,
    hop_length: int,
    window: mx.array,
) -> mx.array:
    """Reconstruct a centered waveform with vectorized overlap-add."""
    magnitude = mx.clip(magnitude, a_min=None, a_max=1e2)
    real = magnitude * mx.cos(phase)
    imag = magnitude * mx.sin(phase)

    batch_size, _, frame_count = real.shape
    real_mirror = real[:, 1:-1, :][:, ::-1, :]
    imag_mirror = imag[:, 1:-1, :][:, ::-1, :]
    spectrum = mx.concatenate([real, real_mirror], axis=1) + 1j * mx.concatenate(
        [imag, -imag_mirror], axis=1
    )
    frames = mx.real(mx.fft.ifft(spectrum, axis=1))
    frames = frames * window.reshape(1, -1, 1)

    output_length = (frame_count - 1) * hop_length + n_fft
    offsets = mx.arange(frame_count) * hop_length
    indices = (offsets[:, None] + mx.arange(n_fft)[None, :]).flatten()

    window_sum = mx.zeros(output_length)
    window_sum = window_sum.at[indices].add(mx.tile(window**2, frame_count))
    window_sum = mx.maximum(window_sum, 1e-8)

    updates = mx.swapaxes(frames, 1, 2).reshape(batch_size, -1)
    batch_indices = mx.repeat(mx.arange(batch_size), frame_count * n_fft)
    flat_indices = mx.tile(indices, batch_size)
    linear_indices = batch_indices * output_length + flat_indices
    output = mx.zeros(batch_size * output_length)
    output = output.at[linear_indices].add(updates.flatten())
    output = output.reshape(batch_size, output_length) / window_sum
    pad_length = n_fft // 2
    return output[:, pad_length:-pad_length]
