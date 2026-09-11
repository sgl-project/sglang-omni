# SPDX-License-Identifier: Apache-2.0
# Ported from OpenBMB/VoxCPM (Apache-2.0), src/voxcpm/modules/audiovae/audio_vae_v2.py.
"""AudioVAE V2: causal DAC-style encoder and sample-rate-conditioned decoder."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Conv1d):
    def __init__(
        self, *args: Any, padding: int = 0, output_padding: int = 0, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.causal_padding = padding * 2 - output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.causal_padding, 0)))


class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(
        self, *args: Any, padding: int = 0, output_padding: int = 0, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.causal_trim = padding * 2 - output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)[..., : -self.causal_trim]


def WNCausalConv1d(*args: Any, **kwargs: Any) -> nn.Module:
    return weight_norm(CausalConv1d(*args, **kwargs))


def WNCausalTransposeConv1d(*args: Any, **kwargs: Any) -> nn.Module:
    return weight_norm(CausalTransposeConv1d(*args, **kwargs))


# note (Xinhao Tan): do not drop @torch.jit.script without measuring first -
# upstream reports it makes the VAE 1.4x faster.
@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


class CausalResidualUnit(nn.Module):
    def __init__(
        self, dim: int = 16, dilation: int = 1, kernel: int = 7, groups: int = 1
    ):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(dim),
            WNCausalConv1d(
                dim,
                dim,
                kernel_size=kernel,
                dilation=dilation,
                padding=((7 - 1) * dilation) // 2,
                groups=groups,
            ),
            Snake1d(dim),
            WNCausalConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CausalEncoderBlock(nn.Module):
    def __init__(
        self,
        output_dim: int = 16,
        input_dim: int | None = None,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            CausalResidualUnit(input_dim, dilation=1, groups=groups),
            CausalResidualUnit(input_dim, dilation=3, groups=groups),
            CausalResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            WNCausalConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CausalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        latent_dim: int = 32,
        strides: list[int] | None = None,
        depthwise: bool = False,
    ):
        super().__init__()
        strides = strides or [2, 4, 8, 8]
        blocks: list[nn.Module] = [WNCausalConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            blocks.append(
                CausalEncoderBlock(
                    output_dim=d_model,
                    stride=stride,
                    groups=d_model // 2 if depthwise else 1,
                )
            )
        self.block = nn.Sequential(*blocks)
        self.fc_mu = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        # note (Xinhao Tan): do not delete fc_logvar. Inference never calls it,
        # but audiovae.pth carries its weights and loading is strict, so
        # removing it fails the checkpoint load.
        self.fc_logvar = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        self.enc_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_mu(self.block(x))


class CausalDecoderBlock(nn.Module):
    def __init__(
        self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, groups: int = 1
    ):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNCausalTransposeConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            CausalResidualUnit(output_dim, dilation=1, groups=groups),
            CausalResidualUnit(output_dim, dilation=3, groups=groups),
            CausalResidualUnit(output_dim, dilation=9, groups=groups),
        )
        self.input_channels = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SampleRateConditionLayer(nn.Module):
    """Per-block FiLM conditioning that selects the decoder's output sample rate."""

    def __init__(self, input_dim: int, sr_bin_buckets: int):
        super().__init__()
        self.scale_embed = nn.Embedding(sr_bin_buckets, input_dim)
        self.bias_embed = nn.Embedding(sr_bin_buckets, input_dim)

    def forward(self, x: torch.Tensor, sr_bin: torch.Tensor) -> torch.Tensor:
        scale = self.scale_embed(sr_bin).unsqueeze(-1)
        bias = self.bias_embed(sr_bin).unsqueeze(-1)
        return x * scale + bias


class CausalDecoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: list[int],
        sr_bin_boundaries: list[int],
        depthwise: bool = False,
        d_out: int = 1,
    ):
        super().__init__()
        if depthwise:
            layers: list[nn.Module] = [
                WNCausalConv1d(
                    input_channel,
                    input_channel,
                    kernel_size=7,
                    padding=3,
                    groups=input_channel,
                ),
                WNCausalConv1d(input_channel, channels, kernel_size=1),
            ]
        else:
            layers = [WNCausalConv1d(input_channel, channels, kernel_size=7, padding=3)]

        output_dim = channels
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers.append(
                CausalDecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    groups=output_dim if depthwise else 1,
                )
            )

        layers += [
            Snake1d(output_dim),
            WNCausalConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.ModuleList(layers)
        self.register_buffer(
            "sr_bin_boundaries", torch.tensor(sr_bin_boundaries, dtype=torch.int32)
        )
        sr_bin_buckets = len(sr_bin_boundaries) + 1
        self.sr_cond_model = nn.ModuleList(
            [
                (
                    SampleRateConditionLayer(layer.input_channels, sr_bin_buckets)
                    if isinstance(layer, CausalDecoderBlock)
                    else None
                )
                for layer in self.model
            ]
        )

    def forward(self, x: torch.Tensor, sample_rate: torch.Tensor) -> torch.Tensor:
        sr_bin = torch.bucketize(sample_rate, self.sr_bin_boundaries)
        for layer, cond_layer in zip(self.model, self.sr_cond_model):
            if cond_layer is not None:
                x = cond_layer(x, sr_bin)
            x = layer(x)
        return x


class AudioVAEConfig(BaseModel):
    encoder_dim: int = 128
    encoder_rates: list[int] = [2, 5, 8, 8]
    latent_dim: int = 64
    decoder_dim: int = 2048
    decoder_rates: list[int] = [8, 6, 5, 2, 2, 2]
    depthwise: bool = True
    sample_rate: int = 16000
    out_sample_rate: int = 48000
    sr_bin_boundaries: list[int] = [20000, 30000, 40000]


class AudioVAE(nn.Module):
    """Encodes 16 kHz waveforms to latents and decodes latents to 48 kHz audio."""

    def __init__(self, config: AudioVAEConfig | None = None):
        super().__init__()
        config = config or AudioVAEConfig()
        self.config = config
        self.latent_dim = config.latent_dim
        self.sample_rate = config.sample_rate
        self.out_sample_rate = config.out_sample_rate
        self.hop_length = math.prod(config.encoder_rates)
        self.decode_chunk_size = math.prod(config.decoder_rates)

        self.encoder = CausalEncoder(
            config.encoder_dim,
            config.latent_dim,
            config.encoder_rates,
            depthwise=config.depthwise,
        )
        self.decoder = CausalDecoder(
            config.latent_dim,
            config.decoder_dim,
            config.decoder_rates,
            sr_bin_boundaries=config.sr_bin_boundaries,
            depthwise=config.depthwise,
        )

    @torch.inference_mode()
    def encode(
        self, audio: torch.Tensor, sample_rate: int | None = None
    ) -> torch.Tensor:
        """Encode ``[B, 1, T]`` (or ``[B, T]``) audio into ``[B, latent_dim, T']``."""
        if sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError(
                f"AudioVAE expects {self.sample_rate} Hz input, got {sample_rate}"
            )
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        length = audio.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        return self.encoder(F.pad(audio, (0, right_pad)))

    @torch.inference_mode()
    def decode(
        self, latents: torch.Tensor, sample_rate: int | None = None
    ) -> torch.Tensor:
        """Decode ``[B, latent_dim, T']`` latents into ``[B, 1, T]`` audio."""
        target = self.out_sample_rate if sample_rate is None else sample_rate
        sr_cond = torch.tensor(
            [target], device=latents.device, dtype=torch.int32
        ).expand(latents.shape[0])
        return self.decoder(latents, sr_cond)


__all__ = ["AudioVAE", "AudioVAEConfig"]
