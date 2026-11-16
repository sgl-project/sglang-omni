# SPDX-License-Identifier: Apache-2.0
# The Oobleck decoder and SnakeBeta equations derive from stable-audio-tools commit
# a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45 and BigVGAN, as identified by YuE2's
# third-party notices.
# Oobleck / SnakeBeta: Copyright (c) 2023 Stability AI; Copyright (c) 2022 NVIDIA CORPORATION.
# MIT license notices are retained in licenses/.
"""YuE2 FP32 stereo decoder with bounded, overlapping latent tiles."""

import math
from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch import nn
from torch.nn.utils import weight_norm

from sglang_omni.models.yue2.request import CancellationCheck, check_cancelled


class DecoderConfig(BaseModel):
    out_channels: int = 2
    channels: int = 64
    latent_dim: int = 64
    c_mults: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    strides: list[int] = Field(default_factory=lambda: [2, 2, 4, 4, 5, 6])
    use_snake: bool = True
    snake_type: Literal["vanilla"] = "vanilla"
    antialias_activation: Literal[False] = False
    use_nearest_upsample: Literal[False] = False
    use_filter: Literal[False] = False
    final_tanh: bool = False


class VaeConfig(BaseModel):
    decoder_config: DecoderConfig
    sample_rate: Literal[48000] = 48000
    latent_dim: Literal[64] = 64
    downsampling_ratio: int = 1920
    audio_channels: Literal[2] = 2


class SnakeBeta(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha[None, :, None].exp()
        beta = self.beta[None, :, None].exp()
        return waveform + (1.0 / (beta + 1e-9)) * torch.sin(waveform * alpha).square()


def activation(channels: int, use_snake: bool) -> nn.Module:
    return SnakeBeta(channels) if use_snake else nn.ELU()


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int, use_snake: bool) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            activation(channels, use_snake),
            weight_norm(
                nn.Conv1d(
                    channels, channels, 7, dilation=dilation, padding=3 * dilation
                )
            ),
            activation(channels, use_snake),
            weight_norm(nn.Conv1d(channels, channels, 1)),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.layers(waveform) + waveform


class DecoderBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, stride: int, use_snake: bool
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            activation(input_channels, use_snake),
            weight_norm(
                nn.ConvTranspose1d(
                    input_channels,
                    output_channels,
                    2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                )
            ),
            ResidualUnit(output_channels, 1, use_snake),
            ResidualUnit(output_channels, 3, use_snake),
            ResidualUnit(output_channels, 9, use_snake),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.layers(waveform)


class OobleckDecoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        if len(config.c_mults) != len(config.strides):
            raise ValueError(
                "VAE channel multipliers and strides must have equal lengths"
            )
        else:
            pass
        channels = [
            config.channels,
            *(multiple * config.channels for multiple in config.c_mults),
        ]
        layers: list[nn.Module] = [
            weight_norm(nn.Conv1d(config.latent_dim, channels[-1], 7, padding=3))
        ]
        for index in range(len(config.strides) - 1, -1, -1):
            layers.append(
                DecoderBlock(
                    channels[index + 1],
                    channels[index],
                    config.strides[index],
                    config.use_snake,
                )
            )
        layers.extend(
            [
                activation(channels[0], config.use_snake),
                weight_norm(
                    nn.Conv1d(
                        channels[0], config.out_channels, 7, padding=3, bias=False
                    )
                ),
                nn.Tanh() if config.final_tanh else nn.Identity(),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.layers(latent)


def output_length(module: nn.Module, length: int) -> int:
    if isinstance(module, (OobleckDecoder, DecoderBlock)):
        return output_length(module.layers, length)
    elif isinstance(module, nn.Sequential):
        for child in module:
            length = output_length(child, length)
        return length
    elif isinstance(module, nn.ConvTranspose1d):
        return (
            (length - 1) * module.stride[0]
            - 2 * module.padding[0]
            + module.dilation[0] * (module.kernel_size[0] - 1)
            + module.output_padding[0]
            + 1
        )
    elif isinstance(module, nn.Conv1d):
        return (
            length
            + 2 * module.padding[0]
            - module.dilation[0] * (module.kernel_size[0] - 1)
            - 1
        ) // module.stride[0] + 1
    elif isinstance(module, (ResidualUnit, SnakeBeta, nn.ELU, nn.Identity, nn.Tanh)):
        return length
    else:
        raise TypeError(f"Unknown VAE length rule for {type(module).__name__}")


def input_support(module: nn.Module, low: int, high: int) -> tuple[int, int]:
    if isinstance(module, (OobleckDecoder, DecoderBlock)):
        return input_support(module.layers, low, high)
    elif isinstance(module, nn.Sequential):
        for child in reversed(list(module)):
            low, high = input_support(child, low, high)
        return low, high
    elif isinstance(module, ResidualUnit):
        residual_low, residual_high = input_support(module.layers, low, high)
        return min(low, residual_low), max(high, residual_high)
    elif isinstance(module, nn.ConvTranspose1d):
        return (
            -(
                -(
                    low
                    + module.padding[0]
                    - module.dilation[0] * (module.kernel_size[0] - 1)
                )
                // module.stride[0]
            ),
            (high + module.padding[0]) // module.stride[0],
        )
    elif isinstance(module, nn.Conv1d):
        return low * module.stride[0] - module.padding[0], high * module.stride[
            0
        ] - module.padding[0] + module.dilation[0] * (module.kernel_size[0] - 1)
    elif isinstance(module, (SnakeBeta, nn.ELU, nn.Identity, nn.Tanh)):
        return low, high
    else:
        raise TypeError(f"Unknown VAE support rule for {type(module).__name__}")


class Yue2Vae(nn.Module):
    def __init__(self, config: VaeConfig) -> None:
        super().__init__()
        self.config = config
        if (
            math.prod(config.decoder_config.strides) != config.downsampling_ratio
            or config.decoder_config.latent_dim != config.latent_dim
            or config.decoder_config.out_channels != config.audio_channels
        ):
            raise ValueError("VAE decoder shape disagrees with its configuration")
        else:
            self.decoder = OobleckDecoder(config.decoder_config)

    def required_halo(self, core_frames: int) -> int:
        low, high = input_support(
            self.decoder, 0, core_frames * self.config.downsampling_ratio - 1
        )
        return max(0, -low, high - core_frames + 1)

    @torch.inference_mode()
    def decode_tiled(
        self,
        latents: torch.Tensor,
        core_frames: int,
        halo_frames: int,
        cancelled: CancellationCheck,
    ) -> torch.Tensor:
        ratio = self.config.downsampling_ratio
        if core_frames < 1 or halo_frames < self.required_halo(core_frames):
            raise ValueError(
                "VAE tile needs a positive core and sufficient receptive-field halo"
            )
        else:
            pass
        frame_count = latents.shape[-1]
        sample_count = output_length(self.decoder, frame_count)
        audio = torch.empty(
            (latents.shape[0], self.config.audio_channels, sample_count),
            dtype=torch.float32,
            device="cpu",
        )
        device = next(self.parameters()).device
        for start in range(0, frame_count, core_frames):
            check_cancelled(cancelled)
            end = min(frame_count, start + core_frames)
            left, right = max(0, start - halo_frames), min(
                frame_count, end + halo_frames
            )
            with torch.autocast(device_type=device.type, enabled=False):
                tile = self.decoder(
                    latents[..., left:right].to(device=device, dtype=torch.float32)
                )
            output_start, output_end = start * ratio, min(end * ratio, sample_count)
            crop_start = (start - left) * ratio
            audio[..., output_start:output_end].copy_(
                tile[..., crop_start : crop_start + output_end - output_start].cpu()
            )
        if not torch.isfinite(audio).all():
            raise FloatingPointError("YuE2 VAE produced non-finite audio")
        else:
            return audio.clamp(-1, 1)
