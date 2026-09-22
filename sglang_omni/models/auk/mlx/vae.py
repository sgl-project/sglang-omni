# SPDX-License-Identifier: MIT AND Apache-2.0
# Copyright (C) 2026 Tencent. All rights reserved.
# Copyright (c) 2022 NVIDIA CORPORATION.
# Derived from Tencent-Hunyuan/AuK; see LICENSE for the MIT permission notice.
"""Native MLX AuK audio VAE encoder, posterior sampling, and BigVGAN decoder."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from sglang_omni.models.auk.hf_config import AuKVAEConfig
from sglang_omni.models.auk.mlx.vae_layers import (
    Activation1d,
    AMPBlock1,
    Conv1d,
    ConvTranspose1d,
    Encoder,
)


class BigVGANFlowVAE(nn.Module):
    """Inference VAE; the posterior regularization flow is training-only."""

    def __init__(self, h: AuKVAEConfig) -> None:
        super().__init__()
        if not h.use_vae:
            raise ValueError("AuK posterior encoding requires use_vae=True")
        else:
            pass
        self.h = h
        self.hop_size = h.hop_size
        self.global_mean = mx.zeros(h.latent_dim, dtype=mx.float32)
        self.global_log_std = mx.ones(h.latent_dim, dtype=mx.float32)
        self.audio_encoder = Encoder(
            h.latent_dim, h.downsample_channels, h.downsample_rates
        )
        self.conv_pre = Conv1d(h.latent_dim, h.upsample_initial_channel, 7)
        self.ups = [
            [
                ConvTranspose1d(
                    h.upsample_initial_channel // 2**i,
                    h.upsample_initial_channel // 2 ** (i + 1),
                    kernel,
                    rate,
                    h.causal,
                )
            ]
            for i, (rate, kernel) in enumerate(
                zip(h.upsample_rates, h.upsample_kernel_sizes)
            )
        ]
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.resblocks = [
            AMPBlock1(
                h.upsample_initial_channel // 2 ** (i + 1),
                kernel,
                dilation,
                h.causal,
                h.act_causal,
                h.snake_logscale,
            )
            for i in range(len(self.ups))
            for kernel, dilation in zip(
                h.resblock_kernel_sizes, h.resblock_dilation_sizes
            )
        ]
        output_channels = h.upsample_initial_channel // 2 ** len(self.ups)
        self.activation_post = Activation1d(
            output_channels, h.act_causal, h.snake_logscale
        )
        self.conv_post = Conv1d(output_channels, 1, 7, causal=h.causal, bias=False)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Fold Torch weight normalization and convert channel-first kernels."""
        converted = {}
        for name, value in weights.items():
            if name.startswith("flow.") or name.endswith(".weight_g"):
                continue
            elif name.endswith(".weight_v"):
                prefix = name.removesuffix("_v")
                value = value.astype(mx.float32)
                magnitude = weights[f"{name.removesuffix('_v')}_g"].astype(mx.float32)
                value = value * (
                    magnitude
                    / mx.sqrt(mx.sum(value * value, axis=(1, 2), keepdims=True))
                )
                name = prefix
            else:
                pass
            if name.endswith(".weight") and value.ndim == 3:
                axes = (1, 2, 0) if name.startswith("ups.") else (0, 2, 1)
                value = value.transpose(axes)
            elif name.endswith(".filter"):
                value = value.transpose(0, 2, 1)
            else:
                pass
            converted[name] = value.astype(mx.float32)
        return converted

    def encode(
        self,
        sample: mx.array,
        sample_lengths: mx.array | None = None,
        *,
        noise: mx.array | None = None,
        key: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Encode [B, 1, samples] into normalized [B, frames, channels].

        Optional posterior noise uses [B, channels, frames], matching Torch.
        """
        if sample.ndim != 3 or sample.shape[1] != 1:
            raise ValueError("VAE input must have shape [batch, 1, samples]")
        else:
            pass
        stats = self.audio_encoder(sample.astype(mx.float32).transpose(0, 2, 1))
        mean, log_std = mx.split(stats, 2, axis=-1)
        if noise is None:
            posterior_noise = mx.random.normal(mean.shape, key=key)
        elif noise.shape != (mean.shape[0], mean.shape[2], mean.shape[1]):
            raise ValueError(
                "Posterior noise must have shape [batch, channels, frames]"
            )
        else:
            posterior_noise = noise.astype(mx.float32).transpose(0, 2, 1)
        latents = mean + posterior_noise * mx.exp(log_std)
        latents = (latents - self.global_mean) / mx.sqrt(self.global_log_std)
        if sample_lengths is None:
            sample_lengths = mx.full((sample.shape[0],), sample.shape[-1], mx.int32)
        else:
            pass
        lengths = mx.minimum(sample_lengths // self.hop_size, latents.shape[1])
        return latents, lengths

    def denormalize(self, latents: mx.array) -> mx.array:
        return (
            latents.astype(mx.float32) * mx.sqrt(self.global_log_std) + self.global_mean
        )

    def inference_from_latents(self, x: mx.array) -> mx.array:
        """Decode [B, channels, frames] into a clipped [B, 1, samples] waveform."""
        if x.ndim != 3 or x.shape[1] != self.h.latent_dim:
            raise ValueError(
                f"VAE latents must have shape [batch, {self.h.latent_dim}, frames]"
            )
        else:
            pass
        x = self.conv_pre(x.astype(mx.float32).transpose(0, 2, 1))
        for i, upsample in enumerate(self.ups):
            for layer in upsample:
                x = layer(x)
            start = i * self.num_kernels
            combined = self.resblocks[start](x)
            for block in self.resblocks[start + 1 : start + self.num_kernels]:
                combined = combined + block(x)
            x = combined / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return mx.clip(x, -1.0, 1.0).transpose(0, 2, 1)

    def decode(self, latents: mx.array) -> mx.array:
        """Decode normalized [B, frames, channels] into [B, samples]."""
        return self.inference_from_latents(
            self.denormalize(latents).transpose(0, 2, 1)
        )[:, 0]
