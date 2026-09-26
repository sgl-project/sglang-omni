# SPDX-License-Identifier: Apache-2.0
"""Mimi, the 12.5 Hz neural audio codec Moshi and PersonaPlex speak through.

24 kHz audio → SEANet encoder (25 Hz) → transformer → 2× downsample →
split residual vector quantizer (1 semantic + 7 acoustic codebooks) and back.
Written from the reference behaviour; the checkpoint's own tensor names are
kept wherever the module tree allows so loading stays a rename, not a rewrite.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from einops import rearrange
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional

from sglang_omni.models.personaplex.architecture import MIMI, MimiSpec
from sglang_omni.models.personaplex.components.causal_conv import (
    ELU,
    CausalConv1d,
    CausalConvTranspose1d,
    StreamingModule,
)


def apply_interleaved_rope(
    q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, max_period: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate adjacent pairs (2i, 2i+1), the GPT-J convention, in float32.

    Args:
        q, k: [B, H, T, D].
        positions: [T] absolute positions.
    """
    dim = q.shape[-1]
    freqs = torch.exp(
        torch.arange(dim // 2, device=q.device, dtype=torch.float32)
        * (-math.log(max_period) * 2 / dim)
    )
    angles = positions.to(torch.float32).view(-1, 1) * freqs
    cos, sin = torch.cos(angles), torch.sin(angles)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        pairs = x.float().view(*x.shape[:-1], dim // 2, 2)
        real, imag = pairs[..., 0], pairs[..., 1]
        out = torch.stack([real * cos - imag * sin, real * sin + imag * cos], dim=-1)
        return out.view(x.shape).to(x.dtype)

    return rotate(q), rotate(k)


@dataclass
class AttentionState:
    """The reference's ring cache: a fixed buffer written modulo its capacity."""

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    end_offset: int = 0


class MimiAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        context: int,
        max_period: float,
        *,
        write_chunk: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.context = context
        self.max_period = max_period
        # Note (wilsonzheng0327): Steps the reference writes to its ring per call (one
        # codec frame); the whole-sequence mask below reproduces that ring's behaviour.
        self.write_chunk = write_chunk
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, *, offset: int = 0, state: AttentionState | None = None
    ) -> torch.Tensor:
        length = x.shape[1]
        projected = functional.linear(x, self.in_proj_weight)
        q, k, v = rearrange(
            projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads
        )
        pos_q = offset + torch.arange(length, device=x.device)
        q, k = apply_interleaved_rope(q, k, pos_q, self.max_period)
        if state is None:
            pos_k = pos_q
            delta = pos_q.view(-1, 1) - pos_k.view(1, -1)
            # Note (wilsonzheng0327): The reference writes a whole chunk into its ring
            # before attending, and once the ring is full it labels the slot at the
            # write cursor as a future position. So a query sees only the keys
            # newer than cursor - context, the cursor taken after its own chunk:
            # the plain window until the ring fills, one to two keys fewer after.
            # A partial last chunk only advances the cursor by what it holds.
            # As a rule over positions this is one batched attention that matches
            # the frame-by-frame ring bit for bit.
            cursor = ((pos_q // self.write_chunk + 1) * self.write_chunk).clamp(
                max=offset + length
            )
            mask = (delta >= 0) & (
                pos_k.view(1, -1) > (cursor - self.context).view(-1, 1)
            )
        else:
            pos_k = self.write_ring(k, v, state)
            k, v = state.keys, state.values
            delta = pos_q.view(-1, 1) - pos_k.view(1, -1)
            mask = (pos_k.view(1, -1) >= 0) & (delta >= 0) & (delta < self.context)
        out = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(rearrange(out, "b h t d -> b t (h d)"))

    def write_ring(self, k, v, state: AttentionState) -> torch.Tensor:
        """Store this step in the ring and label every slot as the reference does.

        The slot about to be overwritten is labelled as a future position, so
        once the ring is full its oldest entry falls outside the window; the
        non-streaming mask in forward applies the same rule without the ring.
        """
        capacity = self.context
        if state.keys is None:
            shape = (k.shape[0], k.shape[1], capacity, k.shape[3])
            state.keys, state.values = k.new_zeros(shape), v.new_zeros(shape)
        else:
            pass
        slots = torch.arange(k.shape[2], device=k.device) + state.end_offset
        state.keys.index_copy_(2, slots % capacity, k)
        state.values.index_copy_(2, slots % capacity, v)
        state.end_offset += k.shape[2]

        indexes = torch.arange(capacity, device=k.device)
        delta = indexes - state.end_offset % capacity
        positions = torch.where(
            delta <= 0,
            state.end_offset + delta,
            state.end_offset + delta - capacity,
        )
        return torch.where(
            indexes >= state.end_offset, torch.full_like(positions, -1), positions
        )


class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class MimiTransformerLayer(nn.Module):
    def __init__(self, spec: MimiSpec) -> None:
        super().__init__()
        self.self_attn = MimiAttention(
            spec.dim,
            spec.num_heads,
            spec.context,
            spec.rope_max_period,
            write_chunk=spec.frame_ratio,
        )
        self.norm1 = nn.LayerNorm(spec.dim, eps=spec.layer_norm_eps)
        self.norm2 = nn.LayerNorm(spec.dim, eps=spec.layer_norm_eps)
        self.linear1 = nn.Linear(spec.dim, spec.ffn_dim, bias=False)
        self.linear2 = nn.Linear(spec.ffn_dim, spec.dim, bias=False)
        self.layer_scale_1 = LayerScale(spec.dim, spec.layer_scale)
        self.layer_scale_2 = LayerScale(spec.dim, spec.layer_scale)

    def forward(
        self, x: torch.Tensor, *, offset: int = 0, state: AttentionState | None = None
    ) -> torch.Tensor:
        x = x + self.layer_scale_1(
            self.self_attn(self.norm1(x), offset=offset, state=state)
        )
        return x + self.layer_scale_2(
            self.linear2(functional.gelu(self.linear1(self.norm2(x))))
        )


@dataclass
class TransformerState:
    offset: int = 0
    layers: list[AttentionState] = field(default_factory=list)


class MimiTransformer(StreamingModule):
    """Eight layers over [B, C, T] frames at the SEANet rate (25 Hz)."""

    def __init__(self, spec: MimiSpec) -> None:
        super().__init__()
        self.spec = spec
        self.layers = nn.ModuleList(
            MimiTransformerLayer(spec) for _ in range(spec.num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)

    def init_state(self) -> TransformerState:
        return TransformerState(layers=[AttentionState() for _ in self.layers])

    def step(self, x: torch.Tensor, state: TransformerState) -> torch.Tensor:
        x = x.transpose(1, 2)
        for layer, layer_state in zip(self.layers, state.layers, strict=True):
            x = layer(x, offset=state.offset, state=layer_state)
        state.offset += x.shape[1]
        return x.transpose(1, 2)


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

    def init_state(self) -> list:
        return stack_state(self.block)

    def step(self, x: torch.Tensor, state: list) -> torch.Tensor:
        y = x
        for module, module_state in zip(self.block, state, strict=True):
            y = module.step(y, module_state) if module_state is not None else module(y)
        assert y.shape[-1] == x.shape[-1], (y.shape, x.shape)
        return x + y


def run_stack(modules: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    for module in modules:
        x = module(x)
    return x


def stack_state(modules: nn.ModuleList) -> list:
    return [m.init_state() for m in modules]


def step_stack(modules: nn.ModuleList, x: torch.Tensor, state: list) -> torch.Tensor:
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

    def init_state(self) -> list:
        return stack_state(self.model)

    def step(self, x: torch.Tensor, state: list) -> torch.Tensor:
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

    def init_state(self) -> list:
        return stack_state(self.model)

    def step(self, x: torch.Tensor, state: list) -> torch.Tensor:
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

    def __init__(self, spec: MimiSpec, num_codebooks: int) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(spec.dim, spec.codebook_dim, 1, bias=False)
        self.output_proj = nn.Conv1d(spec.codebook_dim, spec.dim, 1, bias=False)
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
    def __init__(self, spec: MimiSpec) -> None:
        super().__init__()
        self.num_semantic = spec.num_semantic_codebooks
        self.rvq_first = ResidualVectorQuantizer(spec, spec.num_semantic_codebooks)
        self.rvq_rest = ResidualVectorQuantizer(
            spec, spec.num_codebooks - spec.num_semantic_codebooks
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
    encoder: list
    transformer: TransformerState
    downsample: object


@dataclass
class MimiDecodeState:
    upsample: object
    transformer: TransformerState
    decoder: list


class MimiCodec(nn.Module):
    def __init__(self, spec: MimiSpec = MIMI) -> None:
        super().__init__()
        self.spec = spec
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
        self.quantizer = SplitResidualVectorQuantizer(spec)
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

    @property
    def samples_per_frame(self) -> int:
        return self.spec.hop_length * self.spec.frame_ratio

    @property
    def device(self) -> torch.device:
        return self.downsample.conv.weight.device

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

    def init_encode_state(self) -> MimiEncodeState:
        return MimiEncodeState(
            encoder=self.encoder.init_state(),
            transformer=self.encoder_transformer.init_state(),
            downsample=self.downsample.init_state(),
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

    def init_decode_state(self) -> MimiDecodeState:
        return MimiDecodeState(
            upsample=self.upsample.init_state(),
            transformer=self.decoder_transformer.init_state(),
            decoder=self.decoder.init_state(),
        )

    @torch.inference_mode()
    def decode_step(
        self, codes_BKF: torch.Tensor, state: MimiDecodeState
    ) -> torch.Tensor:
        latent = self.upsample.step(self.quantizer.decode(codes_BKF), state.upsample)
        latent = self.decoder_transformer.step(latent, state.transformer)
        return self.decoder.step(latent, state.decoder)


RENAMES = (
    (re.compile(r"\.conv\.conv\."), ".conv."),
    (re.compile(r"\.convtr\.convtr\."), ".convtr."),
    (re.compile(r"_transformer\.transformer\."), "_transformer."),
    (re.compile(r"\._codebook\."), ".codebook."),
)
ACOUSTIC_LAYER = re.compile(r"^quantizer\.rvq_rest\.vq\.layers\.(\d+)\.")


def rename_mimi_key(name: str) -> str | None:
    """Map a checkpoint tensor name onto this module tree; None drops it."""
    match = ACOUSTIC_LAYER.match(name)
    if (
        match
        and int(match.group(1)) >= MIMI.num_codebooks - MIMI.num_semantic_codebooks
    ):
        # Note (wilsonzheng0327): Trained with 32 codebooks; Moshi uses only 8.
        return None
    else:
        pass
    for pattern, replacement in RENAMES:
        # Note (wilsonzheng0327): The reference nests convolutions up to three deep
        # (downsample.conv.conv.conv); here each is one module.
        while True:
            renamed = pattern.sub(replacement, name, count=1)
            if renamed == name:
                break
            else:
                pass
            name = renamed
    return name


def resolve_mimi_weights(model_dir: str | Path, glob: str) -> Path:
    matches = sorted(Path(model_dir).glob(glob))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one Mimi weight file matching {glob!r} in "
            f"{model_dir}, found {[m.name for m in matches]}"
        )
    else:
        pass
    return matches[0]


def load_mimi_codec(
    weights_path: str | Path, *, device: torch.device | str
) -> MimiCodec:
    """Build a Mimi codec in float32 and load the checkpoint's weight file."""
    state = {}
    for name, tensor in load_file(str(weights_path)).items():
        renamed = rename_mimi_key(name)
        if renamed is not None:
            state[renamed] = tensor
        else:
            pass
    codec = MimiCodec()
    codec.load_state_dict(state, strict=True)
    return codec.to(device=device).eval()


__all__ = [
    "MimiCodec",
    "MimiDecodeState",
    "MimiEncodeState",
    "apply_interleaved_rope",
    "load_mimi_codec",
    "rename_mimi_key",
    "resolve_mimi_weights",
]
