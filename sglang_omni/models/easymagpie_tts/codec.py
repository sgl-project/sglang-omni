# SPDX-License-Identifier: Apache-2.0
"""Causal FSQ codec that turns stacked EasyMagpie codes into a waveform."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

CODEC_SUBDIR = "codec_native"

# One [batch, channels, history] tensor per causal layer, in
# ResNetDecoder.causal_layers order. Rows of different requests stack on dim 0.
CodecStreamState = list[torch.Tensor]
StreamRunner = Callable[[nn.Module, torch.Tensor], torch.Tensor]


class EasyMagpieCodecConfig:
    def __init__(
        self,
        *,
        input_dim: int = 40,
        input_filters: int = 768,
        hidden_filters: int = 1536,
        num_hidden_layers: int = 6,
        pre_upsample_rates: list[int] | None = None,
        pre_upsample_filters: list[int] | None = None,
        resblock_upsample_rates: list[int] | None = None,
        resblock_upsample_filters: list[int] | None = None,
        kernel_size: int = 3,
        resblock_kernel_size: int = 7,
        num_codebooks: int = 8,
        codebook_size: int = 1024,
        num_levels_per_group: list[int] | None = None,
        frame_stacking_factor: int = 2,
        output_sample_rate: int = 22050,
        **_: object,
    ) -> None:
        self.input_dim = input_dim
        self.input_filters = input_filters
        self.hidden_filters = hidden_filters
        self.num_hidden_layers = num_hidden_layers
        self.pre_upsample_rates = list(pre_upsample_rates or [2])
        self.pre_upsample_filters = list(pre_upsample_filters or [768])
        self.resblock_upsample_rates = list(resblock_upsample_rates or [9, 7, 7])
        self.resblock_upsample_filters = list(
            resblock_upsample_filters or [384, 128, 32]
        )
        self.kernel_size = kernel_size
        self.resblock_kernel_size = resblock_kernel_size
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.num_levels_per_group = list(num_levels_per_group or [4] * 5)
        self.frame_stacking_factor = frame_stacking_factor
        self.output_sample_rate = output_sample_rate
        if self.input_dim != self.num_codebooks * len(self.num_levels_per_group):
            raise ValueError("codec input_dim does not match the FSQ configuration")
        else:
            pass

    @property
    def num_stacked_codebooks(self) -> int:
        return self.num_codebooks * self.frame_stacking_factor

    @property
    def samples_per_stacked_frame(self) -> int:
        upsample = math.prod(self.pre_upsample_rates) * math.prod(
            self.resblock_upsample_rates
        )
        return self.frame_stacking_factor * upsample


class FiniteScalarDequantizer(nn.Module):
    def __init__(self, num_groups: int, levels_per_group: list[int]) -> None:
        super().__init__()
        bases = torch.cumprod(torch.tensor([1, *levels_per_group[:-1]]), dim=0)
        self.num_groups = num_groups
        self.register_buffer("levels", torch.tensor(levels_per_group), persistent=False)
        self.register_buffer("bases", bases, persistent=False)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        digits = (
            torch.div(indices.unsqueeze(-1), self.bases, rounding_mode="floor")
            % self.levels
        )
        scale = torch.div(self.levels, 2, rounding_mode="floor")
        return ((digits - scale) / scale).flatten(start_dim=-2)


class HalfSnake(nn.Module):
    """Snake on the first half of the channels, leaky ReLU on the rest."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.snake_channels = channels // 2
        self.alpha = nn.Parameter(torch.ones(1, self.snake_channels, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        snake = inputs[:, : self.snake_channels]
        activated = snake + torch.sin(self.alpha * snake).square() / (self.alpha + 1e-9)
        return torch.cat(
            (activated, F.leaky_relu(inputs[:, self.snake_channels :])), dim=1
        )


class CausalConv1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, activate: bool
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.history = kernel_size - 1
        self.activation = HalfSnake(out_channels) if activate else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(F.pad(inputs, (self.history, 0))))

    def empty_history(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            (batch, self.conv.in_channels, self.history),
            device=device,
            dtype=self.conv.weight.dtype,
        )

    def stream(
        self, inputs: torch.Tensor, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joined = torch.cat((history, inputs), dim=-1)
        return self.activation(self.conv(joined)), joined[..., -self.history :]


class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            groups=out_channels,
        )
        self.activation = HalfSnake(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(inputs)[..., : -self.stride])

    def empty_history(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            (batch, self.conv.in_channels, 1),
            device=device,
            dtype=self.conv.weight.dtype,
        )

    def stream(
        self, inputs: torch.Tensor, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The kernel spans two input frames, so each output window needs the
        # previous chunk's last frame.
        joined = torch.cat((history, inputs), dim=-1)
        outputs = self.conv(joined)[..., self.stride : -self.stride]
        return self.activation(outputs), joined[..., -1:]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, filters: int, kernel_size: int) -> None:
        super().__init__()
        self.input_conv = CausalConv1d(channels, filters, kernel_size, activate=True)
        self.skip_conv = CausalConv1d(filters, channels, kernel_size, activate=False)
        self.output_activation = HalfSnake(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_activation(inputs + self.skip_conv(self.input_conv(inputs)))

    def stream(self, inputs: torch.Tensor, run: StreamRunner) -> torch.Tensor:
        return self.output_activation(
            inputs + run(self.skip_conv, run(self.input_conv, inputs))
        )


class ResNetDecoder(nn.Module):
    def __init__(self, config: EasyMagpieCodecConfig) -> None:
        super().__init__()
        self.pre_conv = CausalConv1d(
            config.input_dim, config.input_filters, config.kernel_size, activate=False
        )
        channels = config.input_filters
        self.pre_resblocks = nn.ModuleList()
        self.pre_up_sample_layers = nn.ModuleList()
        for rate, filters in zip(
            config.pre_upsample_rates, config.pre_upsample_filters
        ):
            self.pre_resblocks.append(
                ResidualBlock(channels, 2 * channels, config.kernel_size)
            )
            self.pre_up_sample_layers.append(
                CausalConvTranspose1d(channels, filters, rate)
            )
            channels = filters
        self.conv_layers = nn.ModuleList(
            ResidualBlock(channels, config.hidden_filters, config.kernel_size)
            for _ in range(config.num_hidden_layers)
        )
        self.resblock_up_sample_layers = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for rate, filters in zip(
            config.resblock_upsample_rates, config.resblock_upsample_filters
        ):
            self.resblock_up_sample_layers.append(
                CausalConvTranspose1d(channels, filters, rate)
            )
            self.resblocks.append(
                ResidualBlock(filters, 2 * filters, config.resblock_kernel_size)
            )
            channels = filters
        self.post_conv = CausalConv1d(
            channels, 1, config.resblock_kernel_size, activate=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.pre_conv(inputs)
        for block, upsample in zip(self.pre_resblocks, self.pre_up_sample_layers):
            hidden = upsample(block(hidden))
        for block in self.conv_layers:
            hidden = block(hidden)
        for upsample, block in zip(self.resblock_up_sample_layers, self.resblocks):
            hidden = block(upsample(hidden))
        return self.post_conv(hidden).squeeze(1).clamp(-1.0, 1.0)

    def causal_layers(self) -> list[CausalConv1d | CausalConvTranspose1d]:
        """Every layer with history, in the order ``stream`` visits them."""
        layers: list[CausalConv1d | CausalConvTranspose1d] = [self.pre_conv]
        for block, upsample in zip(self.pre_resblocks, self.pre_up_sample_layers):
            layers += [block.input_conv, block.skip_conv, upsample]
        for block in self.conv_layers:
            layers += [block.input_conv, block.skip_conv]
        for upsample, block in zip(self.resblock_up_sample_layers, self.resblocks):
            layers += [upsample, block.input_conv, block.skip_conv]
        layers.append(self.post_conv)
        return layers

    def stream(
        self, inputs: torch.Tensor, state: CodecStreamState
    ) -> tuple[torch.Tensor, CodecStreamState]:
        histories = iter(state)
        next_state: CodecStreamState = []

        def run(layer: nn.Module, values: torch.Tensor) -> torch.Tensor:
            values, history = layer.stream(values, next(histories))
            next_state.append(history)
            return values

        hidden = run(self.pre_conv, inputs)
        for block, upsample in zip(self.pre_resblocks, self.pre_up_sample_layers):
            hidden = run(upsample, block.stream(hidden, run))
        for block in self.conv_layers:
            hidden = block.stream(hidden, run)
        for upsample, block in zip(self.resblock_up_sample_layers, self.resblocks):
            hidden = block.stream(run(upsample, hidden), run)
        return run(self.post_conv, hidden).squeeze(1).clamp(-1.0, 1.0), next_state


class EasyMagpieCodec(nn.Module):
    """Decode [batch, frames, stacked codebooks] codes into [batch, samples] audio."""

    def __init__(self, config: EasyMagpieCodecConfig) -> None:
        super().__init__()
        self.config = config
        self.dequantizer = FiniteScalarDequantizer(
            config.num_codebooks, config.num_levels_per_group
        )
        self.audio_decoder = ResNetDecoder(config)

    def codes_to_latent(self, codes: torch.Tensor) -> torch.Tensor:
        config = self.config
        batch, frames, stacked = codes.shape
        if stacked != config.num_stacked_codebooks:
            raise ValueError(
                f"expected [batch, frames, {config.num_stacked_codebooks}] codes, "
                f"got {tuple(codes.shape)}"
            )
        else:
            pass
        # Each row stacks frame_stacking_factor codec frames codebook-major.
        unstacked = (
            codes.unflatten(-1, (config.num_codebooks, config.frame_stacking_factor))
            .transpose(-2, -1)
            .reshape(batch, frames * config.frame_stacking_factor, config.num_codebooks)
        )
        latent = self.dequantizer(unstacked.clamp(0, config.codebook_size - 1))
        return latent.transpose(1, 2).contiguous()

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.audio_decoder(self.codes_to_latent(codes))

    def empty_stream_state(self, batch: int) -> CodecStreamState:
        device = self.dequantizer.levels.device
        return [
            layer.empty_history(batch, device)
            for layer in self.audio_decoder.causal_layers()
        ]

    @torch.inference_mode()
    def stream(
        self, codes: torch.Tensor, state: CodecStreamState
    ) -> tuple[torch.Tensor, CodecStreamState]:
        """Decode the next [batch, frames, stacked] codes after ``state``.

        Chunked decoding reproduces the whole-utterance decode because every
        layer is causal and ``state`` carries each layer's input history.
        """
        return self.audio_decoder.stream(self.codes_to_latent(codes), state)

    @torch.inference_mode()
    def decode_batch(self, codes: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode ragged code matrices in one padded forward.

        Every convolution is causal, so right padding cannot reach the
        samples of shorter items and trimming recovers the single decode.
        """
        device = self.dequantizer.levels.device
        longest = max(item.shape[0] for item in codes)
        padded = torch.zeros(
            (len(codes), longest, self.config.num_stacked_codebooks),
            device=device,
            dtype=torch.long,
        )
        for row, item in enumerate(codes):
            padded[row, : item.shape[0]] = item.to(device)
        audio = self(padded)
        per_frame = self.config.samples_per_stacked_frame
        return [
            audio[row, : item.shape[0] * per_frame] for row, item in enumerate(codes)
        ]


def load_codec(checkpoint_dir: str, device: str) -> EasyMagpieCodec:
    """Load the float32 codec stored beside the talker checkpoint."""
    codec_dir = Path(checkpoint_dir) / CODEC_SUBDIR
    config = EasyMagpieCodecConfig(
        **json.loads((codec_dir / "config.json").read_text())
    )
    codec = EasyMagpieCodec(config).to(device=device, dtype=torch.float32).eval()
    weights = load_file(str(codec_dir / "model.safetensors"), device=device)
    codec.load_state_dict(weights, strict=True)
    return codec


__all__ = [
    "CodecStreamState",
    "EasyMagpieCodec",
    "EasyMagpieCodecConfig",
    "load_codec",
]
