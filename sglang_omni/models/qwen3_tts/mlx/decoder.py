# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Prince Canuma and contributors.
# Adapted from MLX-Audio 0.4.6 Qwen3-TTS speech_tokenizer.py and config.py.
#
# MIT License
#
# Copyright (c) 2024 Prince Canuma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Native MLX speech decoding and converted checkpoint loading."""

import math
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from pydantic import BaseModel, ConfigDict

from sglang_omni.models.qwen3_tts.mlx.decoder_transformer import (
    DecoderTransformer,
    Qwen3TTSMlxDecoderConfig,
)

DECODE_CHUNK_FRAMES = 300
DECODE_LEFT_CONTEXT_FRAMES = 25
DECODER_KERNEL_SIZE = 7
CODEBOOK_EPSILON = 1e-5
SNAKE_EPSILON = 1e-9
RESIDUAL_KERNEL_SIZE = 7
RESIDUAL_DILATIONS = (1, 3, 9)
# note (Codex): Phase expansion wastes a 32-row GEMM tile for short inputs.
TRANSPOSE_PHASE_MIN_ROWS = 32


class Qwen3TTSMlxTokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decoder_config: Qwen3TTSMlxDecoderConfig
    output_sample_rate: int
    decode_upsample_rate: int


class CausalConv1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.padding: int = (kernel_size - 1) * dilation
        self.conv: nn.Conv1d = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            dilation=dilation,
            groups=groups,
        )

    def __call__(self, hidden: mx.array, *, padding: int | None = None) -> mx.array:
        hidden = mx.conv_general(
            hidden,
            self.conv.weight,
            padding=((self.padding if padding is None else padding,), (0,)),
            kernel_dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        return hidden + self.conv.bias


class CausalTransposeConv1d(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, kernel_size: int, stride: int
    ) -> None:
        super().__init__()
        self.conv: nn.ConvTranspose1d = nn.ConvTranspose1d(
            input_channels, output_channels, kernel_size, stride=stride
        )
        self._phase_weight: (  # noqa: leading-underscore - Non-parameter cache.
            tuple[mx.array, mx.array] | None
        ) = None

    def phase_weight(self, dtype: mx.Dtype) -> mx.array:
        cached = self._phase_weight  # noqa: leading-underscore - Non-parameter cache.
        if (
            cached is not None
            and cached[0] is self.conv.weight
            and cached[1].dtype == dtype
        ):
            return cached[1]
        else:
            output_channels, kernel_size, input_channels = self.conv.weight.shape
            phase_kernel_size = kernel_size // self.conv.stride
            weight = (
                self.conv.weight.reshape(
                    output_channels, phase_kernel_size, self.conv.stride, input_channels
                )
                .transpose(2, 0, 1, 3)[:, :, ::-1, :]
                .reshape(
                    self.conv.stride * output_channels,
                    phase_kernel_size,
                    input_channels,
                )
                .astype(dtype)
            )
            self._phase_weight = (
                self.conv.weight,
                weight,
            )  # noqa: leading-underscore - Non-parameter cache.
            return weight

    def __call__(self, hidden: mx.array) -> mx.array:
        output_channels, kernel_size, _ = self.conv.weight.shape
        stride = self.conv.stride
        if hidden.shape[0] * hidden.shape[1] < TRANSPOSE_PHASE_MIN_ROWS:
            hidden = self.conv(hidden)
            trim_right = kernel_size - stride
            return hidden[:, :-trim_right, :] if trim_right > 0 else hidden
        else:
            # note (Codex): Each phase uses a short convolution without inserted zeros.
            hidden = mx.conv_general(
                hidden,
                self.phase_weight(mx.result_type(hidden.dtype, self.conv.weight.dtype)),
                padding=((kernel_size // stride - 1,), (0,)),
            )
            hidden = hidden.reshape(
                hidden.shape[0], hidden.shape[1] * stride, output_channels
            )
            return hidden + self.conv.bias


@partial(mx.compile, shapeless=True)
def snake_beta_activation(
    hidden: mx.array, alpha: mx.array, inverse_beta: mx.array
) -> mx.array:
    """Fuse waveform-sized operations to avoid intermediate allocations."""
    return hidden + inverse_beta * mx.power(mx.sin(hidden * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha: mx.array = mx.zeros((channels,))
        self.beta: mx.array = mx.zeros((channels,))

    def __call__(self, hidden: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha)
        inverse_beta = 1.0 / (mx.exp(self.beta) + SNAKE_EPSILON)
        return snake_beta_activation(hidden, alpha, inverse_beta)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dwconv: CausalConv1d = CausalConv1d(
            channels, channels, RESIDUAL_KERNEL_SIZE, groups=channels
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1: nn.Linear = nn.Linear(channels, 4 * channels)
        self.pwconv2: nn.Linear = nn.Linear(4 * channels, channels)
        self.gamma: mx.array = mx.ones((channels,)) * 1e-6

    def __call__(self, hidden: mx.array) -> mx.array:
        residual = hidden
        hidden = self.dwconv(hidden)
        hidden = self.norm(hidden)
        hidden = self.pwconv2(nn.gelu(self.pwconv1(hidden)))
        return residual + hidden * self.gamma


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self, quantizer_count: int, dimension: int, channels: int, codebook_size: int
    ) -> None:
        super().__init__()
        self.dimension: int = dimension
        self.codebooks: list[nn.Embedding] = [
            nn.Embedding(codebook_size, dimension) for _ in range(quantizer_count)
        ]
        self.output_proj: nn.Conv1d = nn.Conv1d(dimension, channels, 1, bias=False)

    def decode(self, codes: mx.array) -> mx.array:
        quantized = mx.zeros((codes.shape[0], self.dimension, codes.shape[2]))
        for i in range(codes.shape[1]):
            quantized = quantized + self.codebooks[i](codes[:, i]).transpose(0, 2, 1)
        return self.output_proj(quantized.transpose(0, 2, 1)).transpose(0, 2, 1)


class SplitResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        quantizer_count: int,
        semantic_quantizer_count: int,
        dimension: int,
        channels: int,
        codebook_size: int,
    ) -> None:
        super().__init__()
        self.semantic_quantizer_count: int = semantic_quantizer_count
        self.rvq_first: ResidualVectorQuantizer = ResidualVectorQuantizer(
            semantic_quantizer_count, dimension, channels, codebook_size
        )
        self.rvq_rest: ResidualVectorQuantizer = ResidualVectorQuantizer(
            quantizer_count - semantic_quantizer_count,
            dimension,
            channels,
            codebook_size,
        )

    def decode(self, codes: mx.array) -> mx.array:
        quantized = self.rvq_first.decode(codes[:, : self.semantic_quantizer_count])
        if codes.shape[1] > self.semantic_quantizer_count:
            return quantized + self.rvq_rest.decode(
                codes[:, self.semantic_quantizer_count :]
            )
        else:
            return quantized


class DecoderResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.act1: SnakeBeta = SnakeBeta(channels)
        self.conv1: CausalConv1d = CausalConv1d(
            channels, channels, RESIDUAL_KERNEL_SIZE, dilation=dilation
        )
        self.act2: SnakeBeta = SnakeBeta(channels)
        self.conv2: CausalConv1d = CausalConv1d(channels, channels, 1)

    def __call__(self, hidden: mx.array) -> mx.array:
        residual = hidden
        hidden = self.conv1(self.act1(hidden))
        return self.conv2(self.act2(hidden)) + residual


class DecoderBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, upsample_rate: int
    ) -> None:
        super().__init__()
        self.block: list[SnakeBeta | CausalTransposeConv1d | DecoderResidualUnit] = [
            SnakeBeta(input_channels),
            CausalTransposeConv1d(
                input_channels, output_channels, 2 * upsample_rate, upsample_rate
            ),
            *[
                DecoderResidualUnit(output_channels, dilation)
                for dilation in RESIDUAL_DILATIONS
            ],
        ]

    def __call__(self, hidden: mx.array) -> mx.array:
        for layer in self.block:
            hidden = layer(hidden)
        return hidden


class Qwen3TTSMlxSpeechDecoder(nn.Module):
    """Decode batch-major codec frames to waveforms and valid sample counts."""

    def __init__(self, tokenizer_config: Qwen3TTSMlxTokenizerConfig) -> None:
        super().__init__()
        config = tokenizer_config.decoder_config
        self.num_quantizers: int = config.num_quantizers
        self.output_sample_rate: int = tokenizer_config.output_sample_rate
        self.decode_upsample_rate: int = tokenizer_config.decode_upsample_rate
        self.total_upsample: int = math.prod(
            config.upsample_rates + config.upsampling_ratios
        )
        self.pre_transformer: DecoderTransformer = DecoderTransformer(config)
        self.quantizer: SplitResidualVectorQuantizer = SplitResidualVectorQuantizer(
            quantizer_count=config.num_quantizers,
            semantic_quantizer_count=config.num_semantic_quantizers,
            dimension=config.codebook_dim // 2,
            channels=config.codebook_dim,
            codebook_size=config.codebook_size,
        )
        self.pre_conv: CausalConv1d = CausalConv1d(
            config.codebook_dim, config.latent_dim, 3
        )
        self.upsample: list[list[CausalTransposeConv1d | ConvNeXtBlock]] = [
            [
                CausalTransposeConv1d(
                    config.latent_dim, config.latent_dim, factor, factor
                ),
                ConvNeXtBlock(config.latent_dim),
            ]
            for factor in config.upsampling_ratios
        ]
        output_channels = config.decoder_dim // (2 ** len(config.upsample_rates))
        self.decoder: list[CausalConv1d | DecoderBlock | SnakeBeta] = [
            CausalConv1d(config.latent_dim, config.decoder_dim, DECODER_KERNEL_SIZE),
            *[
                DecoderBlock(
                    config.decoder_dim // (2**i),
                    config.decoder_dim // (2 ** (i + 1)),
                    upsample_rate,
                )
                for i, upsample_rate in enumerate(config.upsample_rates)
            ],
            SnakeBeta(output_channels),
            CausalConv1d(output_channels, 1, DECODER_KERNEL_SIZE),
        ]

    def __call__(self, codes: mx.array) -> mx.array:
        hidden = self.quantizer.decode(codes).transpose(0, 2, 1)
        hidden = self.pre_transformer(self.pre_conv(hidden))
        for upsample_layers in self.upsample:
            for layer in upsample_layers:
                hidden = layer(hidden)
        for decoder_layer in self.decoder:
            hidden = decoder_layer(hidden)
        return mx.clip(hidden.transpose(0, 2, 1), -1.0, 1.0)

    def validate_codes(self, codes: mx.array) -> None:
        """Require non-empty batch-major codec frames with all quantizers."""
        if codes.ndim != 3 or codes.shape[1] == 0:
            raise ValueError(
                "Qwen3-TTS MLX decoder requires non-empty batch-major codec frames"
            )
        elif codes.shape[2] != self.num_quantizers:
            raise ValueError(
                f"Expected {self.num_quantizers} quantizers, got {codes.shape[2]}"
            )
        else:
            pass

    def decode(self, codes: mx.array) -> tuple[mx.array, mx.array]:
        """Accept codes shaped as batch, frames, quantizers."""
        self.validate_codes(codes)
        waveforms: list[mx.array] = []
        for start_frame in range(0, codes.shape[1], DECODE_CHUNK_FRAMES):
            end_frame = min(start_frame + DECODE_CHUNK_FRAMES, codes.shape[1])
            context_frames = min(start_frame, DECODE_LEFT_CONTEXT_FRAMES)
            chunk_codes = codes[:, start_frame - context_frames : end_frame]
            chunk_waveform = self(chunk_codes.transpose(0, 2, 1)).squeeze(1)
            waveforms.append(chunk_waveform[:, context_frames * self.total_upsample :])
        waveform = mx.concatenate(waveforms, axis=-1)
        lengths = (codes[..., 0] > 0).sum(axis=1) * self.decode_upsample_rate
        return waveform, lengths


def load_qwen3_tts_mlx_decoder(model_dir: Path) -> Qwen3TTSMlxSpeechDecoder:
    """Load the converted speech decoder without allocating the reference encoder."""
    tokenizer_dir = model_dir / "speech_tokenizer"
    config = Qwen3TTSMlxTokenizerConfig.model_validate_json(
        (tokenizer_dir / "config.json").read_text(encoding="utf-8")
    )
    decoder = Qwen3TTSMlxSpeechDecoder(config)
    expected_weights = dict(tree_flatten(decoder.parameters()))
    weights = mx.load(str(tokenizer_dir / "model.safetensors"))
    decoder_weights: dict[str, mx.array] = {}
    for name, weight in weights.items():
        if (
            not name.startswith("decoder.")
            or name.endswith("._codebook.cluster_usage")
            or name
            in (
                "decoder.quantizer.rvq_first.input_proj.weight",
                "decoder.quantizer.rvq_rest.input_proj.weight",
            )
        ):
            continue
        else:
            pass
        if name.endswith("._codebook.embedding_sum"):
            base_name = name.removesuffix("._codebook.embedding_sum")
            cluster_usage = weights[f"{base_name}._codebook.cluster_usage"]
            weight = weight / mx.clip(cluster_usage[:, None], CODEBOOK_EPSILON, None)
            name = f"{base_name}.codebook.embed.weight"
        else:
            pass
        name = (
            name.removeprefix("decoder.")
            .replace(".vq.layers.", ".codebooks.")
            .replace(".codebook.embed.weight", ".weight")
        )
        if (
            name in expected_weights
            and weight.ndim == 3
            and weight.shape != expected_weights[name].shape
        ):
            is_transpose_conv = (
                "upsample" in name and ".0.conv.weight" in name
            ) or "block.1.conv.weight" in name
            weight = (
                weight.transpose(1, 2, 0)
                if is_transpose_conv
                else weight.swapaxes(-1, -2)
            )
        else:
            pass
        decoder_weights[name] = weight
    decoder.load_weights(list(decoder_weights.items()), strict=True)
    mx.eval(decoder.parameters())
    decoder.eval()
    return decoder
