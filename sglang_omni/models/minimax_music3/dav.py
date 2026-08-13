# SPDX-License-Identifier: Apache-2.0
"""Decoder-only DAC VAE used by MiniMax Music 3."""

from __future__ import annotations

import math
from types import MethodType
from typing import Any

import torch
from torch import Tensor, nn


def snake(x: Tensor, alpha: Tensor) -> Tensor:
    shape = x.shape
    flat = x.reshape(shape[0], shape[1], -1)
    flat = flat + (alpha + 1e-9).reciprocal() * torch.sin(alpha * flat).pow(2)
    return flat.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        return snake(x, self.alpha)


def _wn_conv(*args: Any, **kwargs: Any) -> nn.Module:
    return nn.utils.weight_norm(nn.Conv1d(*args, **kwargs))


def _wn_conv_transpose(*args: Any, **kwargs: Any) -> nn.Module:
    return nn.utils.weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def remove_weight_norm(module: nn.Module) -> int:
    """Fold fixed inference weight normalization into convolution weights."""
    removed = 0
    for child in module.modules():
        try:
            nn.utils.remove_weight_norm(child)
        except (ValueError, RuntimeError):
            continue
        removed += 1
    return removed


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        pad = (7 - 1) * dilation // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            _wn_conv(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            _wn_conv(dim, dim, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.block(x)
        if y.shape[-1] != x.shape[-1]:
            pad = (x.shape[-1] - y.shape[-1]) // 2
            x = x[..., pad : x.shape[-1] - pad]
        return x + y


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            _wn_conv_transpose(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, 1),
            ResidualUnit(output_dim, 3),
            ResidualUnit(output_dim, 9),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = [_wn_conv(1024, 1536, kernel_size=7, padding=3)]
        rates = (8, 8, 4, 2)
        channels = 1536
        output_dim = channels
        for index, stride in enumerate(rates):
            input_dim = channels // (2**index)
            output_dim = channels // (2 ** (index + 1))
            layers.append(DecoderBlock(input_dim, output_dim, stride))
        layers.extend(
            (
                Snake1d(output_dim),
                _wn_conv(output_dim, 1, kernel_size=7, padding=3),
                nn.Tanh(),
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class MiniMaxMusic3DAV(nn.Module):
    """Decoder wrapper: [B,128,T] -> [B,2,T*512] at 44.1 kHz."""

    def __init__(self) -> None:
        super().__init__()
        self.dec_in_proj = nn.Conv1d(64, 1024, kernel_size=1)
        self.decoder = Decoder()

    def enable_compiled_decoder(self, *, warmup_mel_length: int) -> None:
        """Compile the upsampling stack, which runs on a crippled conv path."""
        self.decoder.forward = MethodType(
            torch.compile(type(self.decoder).forward, dynamic=True), self.decoder
        )
        parameter = next(self.parameters())
        with torch.inference_mode():
            self(
                torch.zeros(
                    (1, 128, warmup_mel_length),
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )

    @torch.inference_mode()
    def forward(self, latent: Tensor) -> Tensor:
        bsz, _, frames = latent.shape
        folded = latent.reshape(bsz * 2, 64, frames)
        wave = self.decoder(self.dec_in_proj(folded))
        return wave.reshape(bsz, 2, -1)


_REQUIRED_DECODER_PREFIXES = ("dec_in_proj.", "decoder.")


def select_decoder_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in state.items()
        if key.startswith(_REQUIRED_DECODER_PREFIXES)
    }


__all__ = ["MiniMaxMusic3DAV", "remove_weight_norm", "select_decoder_state"]
