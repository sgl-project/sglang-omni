# SPDX-License-Identifier: Apache-2.0
"""A body part's motion codec, decoder side, one 12.5 Hz code step at a time.

Residual codebooks turn the step's codes into a latent; a causal transformer over the last 25 latents refines it; a SEANet decoder upsamples it to two 25 fps motion frames. The convolutions, residual blocks and quantizer are the Mimi components PersonaPlex runs; the transformer attends over a ring of exactly 25 steps.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional

from sglang_omni.models.miburi.architecture import MotionCodecSpec
from sglang_omni.models.miburi.components.ring_attention import (
    AttentionRing,
    attend_ring,
)
from sglang_omni.models.personaplex.architecture import MIMI
from sglang_omni.models.personaplex.components.causal_conv import (
    ELU,
    CausalConv1d,
    CausalConvTranspose1d,
)
from sglang_omni.models.personaplex.components.mimi import (
    ResidualVectorQuantizer,
    SEANetResnetBlock,
    SEANetState,
    stack_row_tensors,
    stack_state,
    step_stack,
)
from sglang_omni.models.personaplex.components.mimi_transformer import LayerScale


class CodecAttention(nn.Module):
    """Self-attention with fused query, key and value projections."""

    def __init__(self, dim: int, num_heads: int, rope_max_period: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.rope_max_period = rope_max_period
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, ring: AttentionRing
    ) -> torch.Tensor:
        q, k, v = rearrange(
            self.in_proj(x), "b t (p h d) -> p b h t d", p=3, h=self.num_heads
        )
        return self.out_proj(
            attend_ring(q, k, v, positions, ring, self.rope_max_period)
        )


class CodecTransformerLayer(nn.Module):
    def __init__(self, spec: MotionCodecSpec) -> None:
        super().__init__()
        self.self_attn = CodecAttention(spec.dim, spec.num_heads, spec.rope_max_period)
        self.norm1 = nn.LayerNorm(spec.dim)
        self.norm2 = nn.LayerNorm(spec.dim)
        self.linear1 = nn.Linear(spec.dim, spec.ffn_dim, bias=False)
        self.linear2 = nn.Linear(spec.ffn_dim, spec.dim, bias=False)
        self.layer_scale_1 = LayerScale(spec.dim, spec.layer_scale)
        self.layer_scale_2 = LayerScale(spec.dim, spec.layer_scale)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, ring: AttentionRing
    ) -> torch.Tensor:
        x = x + self.layer_scale_1(self.self_attn(self.norm1(x), positions, ring))
        return x + self.layer_scale_2(
            self.linear2(functional.gelu(self.linear1(self.norm2(x))))
        )


@dataclass
class MotionCodecState:
    """Per row: latent steps taken [rows], the transformer's rings and the decoder's convolution tails."""

    positions: torch.Tensor
    rings: list[AttentionRing]
    decoder: SEANetState

    def row_tensors(self) -> list[torch.Tensor]:
        return [
            self.positions,
            *(tensor for ring in self.rings for tensor in (ring.keys, ring.values)),
            *stack_row_tensors(self.decoder),
        ]


class CodecTransformer(nn.Module):
    """Causal layers over [rows, dim, 1] latents, channels first."""

    def __init__(self, spec: MotionCodecSpec) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            CodecTransformerLayer(spec) for _ in range(spec.num_layers)
        )

    def forward(self, latent: torch.Tensor, state: MotionCodecState) -> torch.Tensor:
        x = latent.transpose(1, 2)
        for layer, ring in zip(self.layers, state.rings, strict=True):
            x = layer(x, state.positions, ring)
        state.positions.add_(1)
        return x.transpose(1, 2)


class MotionCodec(nn.Module):
    def __init__(self, spec: MotionCodecSpec) -> None:
        super().__init__()
        self.spec = spec
        self.quantizer = ResidualVectorQuantizer(
            replace(
                MIMI,
                dim=spec.dim,
                codebook_dim=spec.codebook_dim,
                codebook_size=spec.codebook_size,
            ),
            spec.codebooks,
            mode="decode",
        )
        self.transformer = CodecTransformer(spec)
        # The one upsampling stage halves the width the first convolution widens to.
        width = 2 * spec.n_filters
        self.decoder = nn.ModuleList(
            [
                CausalConv1d(spec.dim, width, spec.kernel_size),
                ELU(),
                CausalConvTranspose1d(
                    width,
                    spec.n_filters,
                    2 * spec.upsample_ratio,
                    stride=spec.upsample_ratio,
                ),
                *(
                    SEANetResnetBlock(
                        spec.n_filters,
                        spec.residual_kernel_size,
                        spec.dilation_base**block,
                        spec.compress,
                    )
                    for block in range(spec.residual_blocks)
                ),
                ELU(),
                CausalConv1d(spec.n_filters, spec.features, spec.last_kernel_size),
            ]
        )

    def new_state(self, rows: int, device: torch.device) -> MotionCodecState:
        spec = self.spec
        return MotionCodecState(
            positions=torch.zeros(rows, dtype=torch.long, device=device),
            rings=[
                AttentionRing.zeros(
                    rows,
                    spec.num_heads,
                    spec.context,
                    spec.dim // spec.num_heads,
                    device=device,
                )
                for _ in self.transformer.layers
            ],
            decoder=stack_state(self.decoder, rows),
        )

    def decode_step(self, codes: torch.Tensor, state: MotionCodecState) -> torch.Tensor:
        """codes [rows, codebooks, 1] → motion [rows, upsample_ratio, features]."""
        latent = self.transformer(self.quantizer.decode(codes), state)
        return step_stack(self.decoder, latent, state.decoder).transpose(1, 2)


__all__ = ["MotionCodec", "MotionCodecState"]
