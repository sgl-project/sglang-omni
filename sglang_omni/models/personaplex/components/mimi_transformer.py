# SPDX-License-Identifier: Apache-2.0
"""Mimi transformer layers with a bounded streaming attention cache."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional

from sglang_omni.models.personaplex.architecture import MimiSpec
from sglang_omni.models.personaplex.components.causal_conv import StreamingModule


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

    keys: torch.Tensor
    values: torch.Tensor

    def reset(self) -> None:
        self.keys.zero_()
        self.values.zero_()


class MimiAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        context: int,
        max_period: float,
        *,
        write_chunk: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.context = context
        self.max_period = max_period
        # Note (wilsonzheng0327): Steps the reference writes to its ring per call (one
        # codec frame); the whole-sequence mask below reproduces that ring's behaviour.
        self.write_chunk = write_chunk
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def init_state(self, batch_size: int) -> AttentionState:
        dim = self.in_proj_weight.shape[1]
        ring_shape = (batch_size, self.num_heads, self.context, dim // self.num_heads)
        return AttentionState(
            keys=self.in_proj_weight.new_zeros(ring_shape),
            values=self.in_proj_weight.new_zeros(ring_shape),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        offset: int | torch.Tensor = 0,
        state: AttentionState | None = None,
    ) -> torch.Tensor:
        """With a state, offset is how many steps its ring has already taken."""
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
            pos_k = self.write_ring(k, v, offset, state)
            k, v = state.keys, state.values
            delta = pos_q.view(-1, 1) - pos_k.view(1, -1)
            mask = (pos_k.view(1, -1) >= 0) & (delta >= 0) & (delta < self.context)
        out = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(rearrange(out, "b h t d -> b t (h d)"))

    def write_ring(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        offset: int | torch.Tensor,
        state: AttentionState,
    ) -> torch.Tensor:
        """Store this step in the ring and label every slot as the reference does.

        The slot about to be overwritten is labelled as a future position, so
        once the ring is full its oldest entry falls outside the window; the
        non-streaming mask in forward applies the same rule without the ring.
        """
        capacity = self.context
        slots = torch.arange(k.shape[2], device=k.device) + offset
        state.keys.index_copy_(2, slots % capacity, k)
        state.values.index_copy_(2, slots % capacity, v)
        end_offset = offset + k.shape[2]

        indexes = torch.arange(capacity, device=k.device)
        delta = indexes - end_offset % capacity
        positions = torch.where(
            delta <= 0,
            end_offset + delta,
            end_offset + delta - capacity,
        )
        return torch.where(
            indexes >= end_offset, torch.full_like(positions, -1), positions
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
        self,
        x: torch.Tensor,
        *,
        offset: int | torch.Tensor = 0,
        state: AttentionState | None = None,
    ) -> torch.Tensor:
        x = x + self.layer_scale_1(
            self.self_attn(self.norm1(x), offset=offset, state=state)
        )
        return x + self.layer_scale_2(
            self.linear2(functional.gelu(self.linear1(self.norm2(x))))
        )


@dataclass
class TransformerState:
    """Steps taken so far, as a 0-dim tensor a device graph reads, and each ring."""

    offset: torch.Tensor
    layers: list[AttentionState]

    def reset(self) -> None:
        self.offset.zero_()
        for layer in self.layers:
            layer.reset()


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

    def init_state(self, batch_size: int) -> TransformerState:
        device = self.layers[0].self_attn.in_proj_weight.device
        return TransformerState(
            offset=torch.zeros((), dtype=torch.long, device=device),
            layers=[layer.self_attn.init_state(batch_size) for layer in self.layers],
        )

    def step(self, x: torch.Tensor, state: TransformerState) -> torch.Tensor:
        x = x.transpose(1, 2)
        for layer, layer_state in zip(self.layers, state.layers, strict=True):
            x = layer(x, offset=state.offset, state=layer_state)
        state.offset.add_(x.shape[1])
        return x.transpose(1, 2)
