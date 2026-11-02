# SPDX-License-Identifier: Apache-2.0
"""Mimi, the 12.5 Hz neural audio codec Moshi and PersonaPlex speak through.

24 kHz audio → SEANet encoder (25 Hz) → transformer → 2× downsample →
split residual vector quantizer (1 semantic + 7 acoustic codebooks) and back.
Written from the reference behaviour; the checkpoint's own tensor names are
kept wherever the module tree allows so loading stays a rename, not a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional

from sglang_omni.models.personaplex.architecture import MIMI, MimiSpec
from sglang_omni.models.personaplex.components.causal_conv import (
    ELU,
    CausalConv1d,
    CausalConvTranspose1d,
    ConvState,
    ConvTransposeState,
    StreamingModule,
)
from sglang_omni.models.personaplex.components.mimi_transformer import (
    AttentionState,
    MimiTransformer,
    TransformerState,
)

SEANetLayerState = ConvState | ConvTransposeState | None
SEANetState = list[SEANetLayerState | list[SEANetLayerState]]


class SEANetResnetBlock(StreamingModule):
    def __init__(
        self, dim: int, kernel_size: int, dilation: int, compress: int
    ) -> None:
        super().__init__()
        hidden = dim // compress
        self.block = nn.ModuleList(
            [
                ELU(),
                CausalConv1d(dim, hidden, kernel_size, dilation=dilation),
                ELU(),
                CausalConv1d(hidden, dim, 1),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for module in self.block:
            y = module(y)
        return x + y

    def init_state(self, batch_size: int) -> SEANetState:
        return stack_state(self.block, batch_size)

    def step(self, x: torch.Tensor, state: SEANetState) -> torch.Tensor:
        y = x
        for module, module_state in zip(self.block, state, strict=True):
            y = module.step(y, module_state) if module_state is not None else module(y)
        assert y.shape[-1] == x.shape[-1], (y.shape, x.shape)
        return x + y


def run_stack(modules: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    for module in modules:
        x = module(x)
    return x


def stack_state(modules: nn.ModuleList, batch_size: int) -> SEANetState:
    return [m.init_state(batch_size) for m in modules]


def reset_stack(state: SEANetState) -> None:
    for module_state in state:
        if isinstance(module_state, list):
            reset_stack(module_state)
        elif module_state is not None:
            module_state.reset()
        else:
            pass


def select_stack_rows(state: SEANetState, rows: torch.Tensor) -> SEANetState:
    selected: SEANetState = []
    for module_state in state:
        if isinstance(module_state, list):
            selected.append(select_stack_rows(module_state, rows))
        elif isinstance(module_state, ConvState):
            # Note (nightdecode): has_started is one flag per state, not per row.
            # Only replicate padding reads it and the decoder pads with zeros, so
            # the selection shares the flag rather than copying it.
            selected.append(
                ConvState(
                    previous=module_state.previous.index_select(0, rows),
                    has_started=module_state.has_started,
                )
            )
        elif isinstance(module_state, ConvTransposeState):
            selected.append(
                ConvTransposeState(partial=module_state.partial.index_select(0, rows))
            )
        else:
            selected.append(None)
    return selected


def stack_row_tensors(state: SEANetState) -> list[torch.Tensor]:
    """A stack's per-row tensors, in the order select_stack_rows visits them."""
    tensors: list[torch.Tensor] = []
    for module_state in state:
        if isinstance(module_state, list):
            tensors.extend(stack_row_tensors(module_state))
        elif isinstance(module_state, ConvState):
            tensors.append(module_state.previous)
        elif isinstance(module_state, ConvTransposeState):
            tensors.append(module_state.partial)
        else:
            pass
    return tensors


def step_stack(
    modules: nn.ModuleList, x: torch.Tensor, state: SEANetState
) -> torch.Tensor:
    for module, module_state in zip(modules, state, strict=True):
        x = module.step(x, module_state)
    return x


class SEANetEncoder(StreamingModule):
    """Waveform [B, 1, T] → latent [B, dim, T / hop_length]."""

    def __init__(self, spec: MimiSpec) -> None:
        super().__init__()
        mult = 1
        layers: list[nn.Module] = [
            CausalConv1d(1, mult * spec.n_filters, spec.kernel_size)
        ]
        for ratio in reversed(spec.ratios):
            channels = mult * spec.n_filters
            layers.append(
                SEANetResnetBlock(channels, spec.residual_kernel_size, 1, spec.compress)
            )
            layers.append(ELU())
            layers.append(CausalConv1d(channels, channels * 2, ratio * 2, stride=ratio))
            mult *= 2
        layers.append(ELU())
        layers.append(
            CausalConv1d(mult * spec.n_filters, spec.dim, spec.last_kernel_size)
        )
        self.model = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_stack(self.model, x)

    def init_state(self, batch_size: int) -> SEANetState:
        return stack_state(self.model, batch_size)

    def step(self, x: torch.Tensor, state: SEANetState) -> torch.Tensor:
        return step_stack(self.model, x, state)


class SEANetDecoder(StreamingModule):
    """Latent [B, dim, F] → waveform [B, 1, F * hop_length]."""

    def __init__(self, spec: MimiSpec) -> None:
        super().__init__()
        mult = 2 ** len(spec.ratios)
        layers: list[nn.Module] = [
            CausalConv1d(spec.dim, mult * spec.n_filters, spec.kernel_size)
        ]
        for ratio in spec.ratios:
            channels = mult * spec.n_filters
            layers.append(ELU())
            layers.append(
                CausalConvTranspose1d(channels, channels // 2, ratio * 2, stride=ratio)
            )
            layers.append(
                SEANetResnetBlock(
                    channels // 2, spec.residual_kernel_size, 1, spec.compress
                )
            )
            mult //= 2
        layers.append(ELU())
        layers.append(CausalConv1d(spec.n_filters, 1, spec.last_kernel_size))
        self.model = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_stack(self.model, x)

    def init_state(self, batch_size: int) -> SEANetState:
        return stack_state(self.model, batch_size)

    def step(self, x: torch.Tensor, state: SEANetState) -> torch.Tensor:
        return step_stack(self.model, x, state)


class EuclideanCodebook(nn.Module):
    """Centroids stored as EMA sums, the way the checkpoint keeps them."""

    def __init__(self, dim: int, size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("_initialized", torch.zeros(1))
        self.register_buffer("cluster_usage", torch.ones(size))
        self.register_buffer("embedding_sum", torch.zeros(size, dim))

    @property
    def embedding(self) -> torch.Tensor:
        return self.embedding_sum / self.cluster_usage.clamp(min=self.eps)[:, None]

    def encode(self, x_ND: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x_ND[None], self.embedding[None], p=2)[0].argmin(dim=-1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return functional.embedding(codes, self.embedding)


class VectorQuantization(nn.Module):
    def __init__(self, dim: int, size: int) -> None:
        super().__init__()
        self.codebook = EuclideanCodebook(dim, size)


class ResidualVQ(nn.Module):
    def __init__(self, dim: int, size: int, num_codebooks: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            VectorQuantization(dim, size) for _ in range(num_codebooks)
        )


class ResidualVectorQuantizer(nn.Module):
    """Projection in, residual codebooks, projection out."""

    def __init__(
        self,
        spec: MimiSpec,
        num_codebooks: int,
        *,
        mode: Literal["encode", "decode", "both"],
    ) -> None:
        super().__init__()
        self.input_proj: nn.Conv1d | None = (
            nn.Conv1d(spec.dim, spec.codebook_dim, 1, bias=False)
            if mode != "decode"
            else None
        )
        self.output_proj: nn.Conv1d | None = (
            nn.Conv1d(spec.codebook_dim, spec.dim, 1, bias=False)
            if mode != "encode"
            else None
        )
        self.vq = ResidualVQ(spec.codebook_dim, spec.codebook_size, num_codebooks)

    def encode(self, x_BCT: torch.Tensor) -> torch.Tensor:
        residual = rearrange(self.input_proj(x_BCT), "b d t -> b t d")
        codes = []
        for layer in self.vq.layers:
            book = layer.codebook
            index = book.encode(rearrange(residual, "b t d -> (b t) d"))
            index = index.view(residual.shape[0], residual.shape[1])
            residual = residual - book.decode(index)
            codes.append(index)
        return torch.stack(codes, dim=1)

    def decode(self, codes_BKT: torch.Tensor) -> torch.Tensor:
        quantized = None
        for k, layer in enumerate(self.vq.layers[: codes_BKT.shape[1]]):
            level = rearrange(layer.codebook.decode(codes_BKT[:, k]), "b t d -> b d t")
            quantized = level if quantized is None else quantized + level
        return self.output_proj(quantized)


class SplitResidualVectorQuantizer(nn.Module):
    def __init__(
        self, spec: MimiSpec, *, mode: Literal["encode", "decode", "both"]
    ) -> None:
        super().__init__()
        self.num_semantic = spec.num_semantic_codebooks
        self.rvq_first = ResidualVectorQuantizer(
            spec, spec.num_semantic_codebooks, mode=mode
        )
        self.rvq_rest = ResidualVectorQuantizer(
            spec, spec.num_codebooks - spec.num_semantic_codebooks, mode=mode
        )

    def encode(self, x_BCT: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.rvq_first.encode(x_BCT), self.rvq_rest.encode(x_BCT)], dim=1
        )

    def decode(self, codes_BKT: torch.Tensor) -> torch.Tensor:
        quantized = self.rvq_first.decode(codes_BKT[:, : self.num_semantic])
        if codes_BKT.shape[1] > self.num_semantic:
            quantized = quantized + self.rvq_rest.decode(
                codes_BKT[:, self.num_semantic :]
            )
        else:
            pass
        return quantized


@dataclass
class MimiEncodeState:
    encoder: SEANetState
    transformer: TransformerState
    downsample: ConvState


@dataclass
class MimiDecodeState:
    upsample: ConvTransposeState
    transformer: TransformerState
    decoder: SEANetState

    def reset(self) -> None:
        """Back to a fresh stream, in place, so captured graphs keep their addresses."""
        self.upsample.reset()
        self.transformer.reset()
        reset_stack(self.decoder)

    def reset_row(self, row: int) -> None:
        """Back to a fresh stream for one row, in place."""
        for tensor in self.row_tensors():
            tensor[row].zero_()

    def select_rows(self, rows: torch.Tensor) -> MimiDecodeState:
        """The given rows, in order, as a new state; each keeps its own offset.

        A device index keeps this capturable, so a graph can gather the rows
        it decodes from a state stacked over many streams.
        """
        return MimiDecodeState(
            upsample=ConvTransposeState(
                partial=self.upsample.partial.index_select(0, rows)
            ),
            transformer=TransformerState(
                offset=self.transformer.offset.index_select(0, rows),
                layers=[
                    AttentionState(
                        keys=layer.keys.index_select(0, rows),
                        values=layer.values.index_select(0, rows),
                    )
                    for layer in self.transformer.layers
                ],
            ),
            decoder=select_stack_rows(self.decoder, rows),
        )

    def store_rows(self, rows: torch.Tensor, selected: MimiDecodeState) -> None:
        """Write a state select_rows took, since advanced, back into its rows."""
        for tensor, selected_tensor in zip(
            self.row_tensors(), selected.row_tensors(), strict=True
        ):
            tensor.index_copy_(0, rows, selected_tensor)

    def row_tensors(self) -> list[torch.Tensor]:
        """Every tensor that holds one row per stream, in a fixed order."""
        return [
            self.upsample.partial,
            self.transformer.offset,
            *(
                tensor
                for layer in self.transformer.layers
                for tensor in (layer.keys, layer.values)
            ),
            *stack_row_tensors(self.decoder),
        ]


class MimiCodec(nn.Module):
    def __init__(
        self,
        spec: MimiSpec = MIMI,
        *,
        mode: Literal["encode", "decode", "both"] = "both",
    ) -> None:
        super().__init__()
        if mode not in ("encode", "decode", "both"):
            raise ValueError(f"Unsupported Mimi mode: {mode}")
        else:
            pass
        self.spec = spec
        self.encoder: SEANetEncoder | None = None
        self.encoder_transformer: MimiTransformer | None = None
        self.downsample: CausalConv1d | None = None
        if mode != "decode":
            self.encoder = SEANetEncoder(spec)
            self.encoder_transformer = MimiTransformer(spec)
            self.downsample = CausalConv1d(
                spec.dim,
                spec.dim,
                2 * spec.frame_ratio,
                stride=spec.frame_ratio,
                bias=False,
                pad_mode="replicate",
            )
        else:
            pass
        self.quantizer = SplitResidualVectorQuantizer(spec, mode=mode)
        self.upsample: CausalConvTranspose1d | None = None
        self.decoder_transformer: MimiTransformer | None = None
        self.decoder: SEANetDecoder | None = None
        if mode != "encode":
            self.upsample = CausalConvTranspose1d(
                spec.dim,
                spec.dim,
                2 * spec.frame_ratio,
                stride=spec.frame_ratio,
                groups=spec.dim,
                bias=False,
            )
            self.decoder_transformer = MimiTransformer(spec)
            self.decoder = SEANetDecoder(spec)
        else:
            pass

    @property
    def samples_per_frame(self) -> int:
        return self.spec.hop_length * self.spec.frame_ratio

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def encode(self, wav_B1T: torch.Tensor) -> torch.Tensor:
        """[B, 1, T] with T a multiple of 1920 → codes [B, 8, F]."""
        latent = self.encoder(wav_B1T)
        latent = self.encoder_transformer(latent)
        return self.quantizer.encode(self.downsample(latent))

    @torch.inference_mode()
    def decode(self, codes_BKF: torch.Tensor) -> torch.Tensor:
        """Codes [B, 8, F] → waveform [B, 1, F * 1920]."""
        latent = self.upsample(self.quantizer.decode(codes_BKF))
        return self.decoder(self.decoder_transformer(latent))

    def init_encode_state(self, batch_size: int) -> MimiEncodeState:
        return MimiEncodeState(
            encoder=self.encoder.init_state(batch_size),
            transformer=self.encoder_transformer.init_state(
                batch_size, max_chunk_steps=self.spec.frame_ratio
            ),
            downsample=self.downsample.init_state(batch_size),
        )

    @torch.inference_mode()
    def encode_step(
        self, wav_B1T: torch.Tensor, state: MimiEncodeState
    ) -> torch.Tensor:
        latent = self.encoder.step(wav_B1T, state.encoder)
        latent = self.encoder_transformer.step(latent, state.transformer)
        latent = self.downsample.step(latent, state.downsample)
        if latent.shape[-1] == 0:
            return latent.new_empty(
                latent.shape[0], self.spec.num_codebooks, 0, dtype=torch.long
            )
        else:
            pass
        return self.quantizer.encode(latent)

    def init_decode_state(
        self, batch_size: int, *, max_chunk_frames: int
    ) -> MimiDecodeState:
        return MimiDecodeState(
            upsample=self.upsample.init_state(batch_size),
            transformer=self.decoder_transformer.init_state(
                batch_size, max_chunk_steps=max_chunk_frames * self.spec.frame_ratio
            ),
            decoder=self.decoder.init_state(batch_size),
        )

    @torch.inference_mode()
    def decode_step(
        self, codes_BKF: torch.Tensor, state: MimiDecodeState
    ) -> torch.Tensor:
        latent = self.upsample.step(self.quantizer.decode(codes_BKF), state.upsample)
        latent = self.decoder_transformer.step(latent, state.transformer)
        return self.decoder.step(latent, state.decoder)


__all__ = [
    "MimiCodec",
    "MimiDecodeState",
    "MimiEncodeState",
]
