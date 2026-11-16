# SPDX-License-Identifier: Apache-2.0
"""The release's causal RVQ-VAE motion tokenizers (upper body, hands, FLAME) and their block-by-block decoder.

Every convolution pads only on the left, so a decoded frame depends on
earlier codes alone; decoding keeps a bounded left context and no right
context, which is what lets each 0.8 s block decode as soon as it exists.
"""

from __future__ import annotations

import torch
from torch import nn

from sglang_omni.models.ellsa2.formats import FLAME_FEATURES

# The release tokenizers' shared shape (checkpoint `args`): 2048 codes of width 512, two stride-2 stages (4 frames per code), six residual blocks with dilation growing by 3.
CODEBOOK_SIZE = 2048
CODE_WIDTH = 512
WIDTH = 512
DEPTH = 6
DILATION_GROWTH = 3
DOWNSAMPLE_STAGES = 2
STRIDE = 2
FRAMES_PER_CODE = STRIDE**DOWNSAMPLE_STAGES
EXPRESSION_FEATURES = 100
EYE_FEATURES = 4
JAW = slice(103, 104)
EYE_POSE = slice(104, 108)


def causal_conv(
    channels_in: int,
    channels_out: int,
    kernel: int,
    pad: int,
    stride: int = 1,
    dilation: int = 1,
) -> list[nn.Module]:
    return [
        nn.ReplicationPad1d((pad, 0)),
        nn.Conv1d(channels_in, channels_out, kernel, stride, 0, dilation),
    ]


class ResidualBlock(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.conv1 = nn.Sequential(
            *causal_conv(WIDTH, WIDTH, 3, 2 * dilation, dilation=dilation)
        )
        self.conv2 = nn.Conv1d(WIDTH, WIDTH, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.activation2(self.conv1(self.activation1(x)))) + x


def residual_stack(reverse: bool) -> nn.Sequential:
    blocks = [ResidualBlock(DILATION_GROWTH**depth) for depth in range(DEPTH)]
    return nn.Sequential(*(blocks[::-1] if reverse else blocks))


class ResidualStack(nn.Module):
    """Holds the blocks under `model`, the checkpoint's key layout."""

    def __init__(self, reverse: bool) -> None:
        super().__init__()
        self.model = residual_stack(reverse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Encoder(nn.Module):
    """[batch, features, frames] → [batch, CODE_WIDTH, frames / FRAMES_PER_CODE], each latent ending with its stride-sized frame block."""

    def __init__(self, features: int) -> None:
        super().__init__()
        stages = [
            nn.Sequential(
                *causal_conv(WIDTH, WIDTH, 2 * STRIDE, STRIDE, stride=STRIDE),
                ResidualStack(reverse=True),
            )
            for _ in range(DOWNSAMPLE_STAGES)
        ]
        self.model = nn.Sequential(
            *causal_conv(features, WIDTH, 3, 2),
            nn.ReLU(),
            *stages,
            *causal_conv(WIDTH, CODE_WIDTH, 3, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decoder(nn.Module):
    """[batch, CODE_WIDTH, steps] → [batch, steps * FRAMES_PER_CODE, features]."""

    def __init__(self, features: int) -> None:
        super().__init__()
        stages = [
            nn.Sequential(
                ResidualStack(reverse=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
                *causal_conv(WIDTH, WIDTH, 3, 2),
            )
            for _ in range(DOWNSAMPLE_STAGES)
        ]
        self.model = nn.Sequential(
            *causal_conv(CODE_WIDTH, WIDTH, 3, 2),
            nn.ReLU(),
            *stages,
            *causal_conv(WIDTH, WIDTH, 3, 2),
            nn.ReLU(),
            *causal_conv(WIDTH, features, 3, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).permute(0, 2, 1)


class Codebook(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.codebook = nn.Buffer(torch.zeros(CODEBOOK_SIZE, CODE_WIDTH))

    def nearest(self, x: torch.Tensor) -> torch.Tensor:
        """Rows [n, CODE_WIDTH] → the index of each row's nearest code."""
        book = self.codebook.t()
        distance = (
            torch.sum(x**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, book)
            + torch.sum(book**2, dim=0, keepdim=True)
        )
        return (-distance).argmax(dim=-1)


class ResidualQuantizer(nn.Module):
    def __init__(self, num_quantizers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Codebook() for _ in range(num_quantizers)])

    def quantize(self, latents: torch.Tensor) -> torch.Tensor:
        """[batch, CODE_WIDTH, steps] → codes [batch, steps, num_quantizers]."""
        batch, _, steps = latents.shape
        residual = latents
        codes = []
        for layer in self.layers:
            rows = residual.permute(0, 2, 1).reshape(batch * steps, -1)
            index = layer.nearest(rows)
            # Note (Junnan Li): The reference subtracts its straight-through value rows + (code - rows), which rounds differently from the code itself.
            quantized = rows + (nn.functional.embedding(index, layer.codebook) - rows)
            residual = residual - quantized.view(batch, steps, -1).permute(0, 2, 1)
            codes.append(index.view(batch, steps))
        return torch.stack(codes, dim=-1)

    def latents(self, codes: torch.Tensor) -> torch.Tensor:
        """Codes [batch, steps, num_quantizers] → summed code vectors [batch, CODE_WIDTH, steps]."""
        vectors = torch.stack(
            [
                layer.codebook[codes[:, :, level]]
                for level, layer in enumerate(self.layers)
            ],
            dim=0,
        )
        return vectors.sum(dim=0).permute(0, 2, 1)


class RVQVAE(nn.Module):
    def __init__(self, features: int, num_quantizers: int, context_frames: int) -> None:
        super().__init__()
        self.encoder = Encoder(features)
        self.decoder = Decoder(features)
        self.quantizer = ResidualQuantizer(num_quantizers)
        self.context_frames = context_frames

    def encode(self, motion: torch.Tensor) -> torch.Tensor:
        """Normalized motion [batch, frames, features] → codes [batch, frames / FRAMES_PER_CODE, num_quantizers], each latent read from at most context_frames frames."""
        x = motion.permute(0, 2, 1).float()
        latents = bounded(x, self.encoder, self.context_frames, FRAMES_PER_CODE, 1)
        return self.quantizer.quantize(latents)


def bounded(
    x: torch.Tensor,
    module: nn.Module,
    window: int,
    stride: int,
    output_stride: int,
    chunk: int = 128,
) -> torch.Tensor:
    """Run module on x [batch, channels, length] so each output of the last stride reads at most window inputs: the first window as is, then each later stride from its own fresh window."""
    prefix = module(x[..., :window])
    if x.shape[-1] <= window:
        return prefix
    else:
        windows = x.unfold(-1, window, stride)[:, :, 1:, :]
    batch, channels, count, _ = windows.shape
    windows = windows.permute(0, 2, 1, 3).reshape(batch * count, channels, window)
    # Note (Junnan Li): The reference runs the windows 128 at a time; batch size changes convolution rounding.
    tails = [
        module(windows[start : start + chunk])[..., -output_stride:]
        for start in range(0, len(windows), chunk)
    ]
    tail = torch.cat(tails, dim=0).reshape(batch, count, -1, output_stride)
    tail = tail.permute(0, 2, 1, 3).reshape(batch, -1, count * output_stride)
    return torch.cat([prefix, tail], dim=-1)


class FlameTokenizer(nn.Module):
    """The FLAME tokenizer: four codes for the expression outside the mouth subspace and the eye pose, two for the mouth subspace and the jaw.

    Motion codes are normalized 108-D FLAME motion: expression 0:100, head
    rotation 100:103 (left out; decodes to zero), jaw 103, eye pose 104:108.
    """

    def __init__(
        self,
        mouth_basis: torch.Tensor,
        mouth_vertex_matrix: torch.Tensor,
        nonmouth_quantizers: int,
        mouth_quantizers: int,
        context_frames: int,
    ) -> None:
        super().__init__()
        mouth_rank = mouth_basis.shape[1]
        # Note (Junnan Li): The checkpoint stores the basis column-major; the buffer keeps that layout, which sets how the matmuls round.
        self.mouth_basis = nn.Buffer(mouth_basis)
        # Loaded for a strict state dict; it serves the checkpoint's training loss only.
        self.mouth_vertex_matrix = nn.Buffer(mouth_vertex_matrix)
        self.nonmouth_vq = RVQVAE(
            EXPRESSION_FEATURES + EYE_FEATURES, nonmouth_quantizers, context_frames
        )
        self.mouth_vq = RVQVAE(mouth_rank + 1, mouth_quantizers, context_frames)
        self.mouth_rank = mouth_rank
        self.nonmouth_quantizers = nonmouth_quantizers

    def encode(self, motion: torch.Tensor) -> torch.Tensor:
        """Normalized FLAME motion [batch, frames, 108] → codes [batch, frames / FRAMES_PER_CODE, nonmouth + mouth quantizers]."""
        expression = motion[..., :EXPRESSION_FEATURES]
        mouth_expression = expression @ self.mouth_basis
        nonmouth_expression = expression - mouth_expression @ self.mouth_basis.T
        nonmouth = torch.cat([nonmouth_expression, motion[..., EYE_POSE]], dim=-1)
        mouth = torch.cat([mouth_expression, motion[..., JAW]], dim=-1)
        return torch.cat(
            [self.nonmouth_vq.encode(nonmouth), self.mouth_vq.encode(mouth)], dim=-1
        )

    def merge(self, nonmouth: torch.Tensor, mouth: torch.Tensor) -> torch.Tensor:
        """Both branches' decoded frames → normalized FLAME motion [batch, frames, 108]."""
        expression = nonmouth[..., :EXPRESSION_FEATURES]
        expression = expression - (expression @ self.mouth_basis) @ self.mouth_basis.T
        expression = expression + mouth[..., : self.mouth_rank] @ self.mouth_basis.T
        output = expression.new_zeros(*expression.shape[:-1], FLAME_FEATURES)
        output[..., :EXPRESSION_FEATURES] = expression
        output[..., JAW] = mouth[..., self.mouth_rank : self.mouth_rank + 1]
        output[..., EYE_POSE] = nonmouth[..., EXPRESSION_FEATURES:]
        return output


class StreamingDecoder:
    """Decodes one block of codes at a time, keeping the latents of the last context steps; equals decoding all codes at once with the same bounded context."""

    def __init__(self, model: RVQVAE) -> None:
        self.model = model
        self.window = model.context_frames // FRAMES_PER_CODE
        self.history: torch.Tensor | None = None

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Codes [1, steps, num_quantizers] → frames [1, steps * FRAMES_PER_CODE, features]."""
        return StreamingDecoder.decode_together([self], codes)

    @staticmethod
    def decode_together(
        decoders: list[StreamingDecoder], codes: torch.Tensor
    ) -> torch.Tensor:
        """Codes [len(decoders), steps, num_quantizers], one row per decoder of the same model → frames [len(decoders), steps * FRAMES_PER_CODE, features]; decoders whose kept latents are as long decode as one batch."""
        model = decoders[0].model
        window = decoders[0].window
        latents = model.quantizer.latents(codes)
        groups: dict[int, list[int]] = {}
        for row, decoder in enumerate(decoders):
            cached = 0 if decoder.history is None else decoder.history.shape[-1]
            groups.setdefault(cached, []).append(row)
        frames: list[torch.Tensor | None] = [None] * len(decoders)
        for cached, rows in groups.items():
            joined = (
                latents[rows]
                if cached == 0
                else torch.cat(
                    [torch.cat([decoders[row].history for row in rows]), latents[rows]],
                    dim=-1,
                )
            )
            decoded = bounded(
                joined,
                lambda steps: model.decoder(steps).permute(0, 2, 1),
                window,
                1,
                FRAMES_PER_CODE,
                # One Session's windows run 128 at a time as the reference runs them; several Sessions' all at once.
                chunk=128 if len(decoders) == 1 else len(rows) * joined.shape[-1],
            )
            kept = joined[..., -(window - 1) :]
            for index, row in enumerate(rows):
                decoders[row].history = kept[index : index + 1].clone()
                frames[row] = decoded[
                    index : index + 1, :, cached * FRAMES_PER_CODE :
                ].permute(0, 2, 1)
        # Note (Junnan Li): One decoder's frames keep the reference's layout, which sets how the axis-angle conversion after rounds.
        return frames[0] if len(frames) == 1 else torch.cat(frames)


__all__ = [
    "CODEBOOK_SIZE",
    "FlameTokenizer",
    "FRAMES_PER_CODE",
    "RVQVAE",
    "StreamingDecoder",
]
