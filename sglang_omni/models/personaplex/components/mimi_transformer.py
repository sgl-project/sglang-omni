# SPDX-License-Identifier: Apache-2.0
"""Mimi transformer layers with a bounded streaming attention cache."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional

from sglang_omni.models.personaplex.architecture import MimiSpec
from sglang_omni.models.personaplex.components.causal_conv import StreamingModule


def apply_interleaved_rope(
    q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, max_period: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate adjacent pairs using shared [T] or per-row [B, T] positions."""
    dimension = q.shape[-1]
    frequencies = torch.exp(
        torch.arange(dimension // 2, device=q.device, dtype=torch.float32)
        * (-math.log(max_period) * 2 / dimension)
    )
    angles = positions.to(torch.float32)[..., None] * frequencies
    if positions.ndim == 1:
        angles = angles[None, None]
    elif positions.ndim == 2:
        angles = angles[:, None]
    else:
        raise ValueError(
            f"RoPE positions must have shape [T] or [B, T], got {positions.shape}"
        )
    cosine, sine = torch.cos(angles), torch.sin(angles)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        pairs = x.float().view(*x.shape[:-1], dimension // 2, 2)
        real, imaginary = pairs[..., 0], pairs[..., 1]
        output = torch.stack(
            [
                real * cosine - imaginary * sine,
                real * sine + imaginary * cosine,
            ],
            dim=-1,
        )
        return output.view(x.shape).to(x.dtype)

    return rotate(q), rotate(k)


@dataclass
class AttentionState:
    """The reference's ring cache: a fixed buffer written modulo its capacity."""

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    end_offset: int | torch.Tensor = 0


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

    def forward(
        self,
        x: torch.Tensor,
        *,
        offset: int | torch.Tensor = 0,
        state: AttentionState | None = None,
    ) -> torch.Tensor:
        length = x.shape[1]
        projected = functional.linear(x, self.in_proj_weight)
        q, k, v = rearrange(
            projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads
        )
        relative_positions = torch.arange(length, device=x.device)
        if isinstance(offset, torch.Tensor):
            assert offset.shape == (x.shape[0],)
            query_positions = offset[:, None] + relative_positions[None]
        else:
            query_positions = offset + relative_positions
        q, k = apply_interleaved_rope(q, k, query_positions, self.max_period)
        if state is None:
            key_positions = query_positions
            position_delta = query_positions[..., :, None] - key_positions[..., None, :]
            # Note (wilsonzheng0327): The reference writes a whole chunk into its ring
            # before attending, and once the ring is full it labels the slot at the
            # write cursor as a future position. So a query sees only the keys
            # newer than cursor - context, the cursor taken after its own chunk:
            # the plain window until the ring fills, one to two keys fewer after.
            # A partial last chunk only advances the cursor by what it holds.
            # As a rule over positions this is one batched attention that matches
            # the frame-by-frame ring bit for bit.
            cursor = (query_positions // self.write_chunk + 1) * self.write_chunk
            cursor = torch.minimum(cursor, query_positions[..., :1] + length)
            mask = (position_delta >= 0) & (
                key_positions[..., None, :] > (cursor - self.context)[..., :, None]
            )
        else:
            key_positions = self.write_ring(k, v, state)
            k, v = state.keys, state.values
            position_delta = query_positions[..., :, None] - key_positions[..., None, :]
            mask = (
                (key_positions[..., None, :] >= 0)
                & (position_delta >= 0)
                & (position_delta < self.context)
            )
        if mask.ndim == 3:
            mask = mask[:, None]
        else:
            pass
        output = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(rearrange(output, "b h t d -> b t (h d)"))

    def write_ring(
        self, k: torch.Tensor, v: torch.Tensor, state: AttentionState
    ) -> torch.Tensor:
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
        indexes = torch.arange(capacity, device=k.device)
        step_positions = torch.arange(k.shape[2], device=k.device)
        if isinstance(state.end_offset, torch.Tensor):
            assert state.end_offset.shape == (k.shape[0],)
            slots = state.end_offset[:, None] + step_positions[None]
            scatter_indexes = slots.remainder(capacity)[:, None, :, None].expand_as(k)
            state.keys.scatter_(2, scatter_indexes, k)
            state.values.scatter_(2, scatter_indexes, v)
            state.end_offset = state.end_offset + k.shape[2]
            end_offsets = state.end_offset[:, None]
            position_delta = indexes[None] - end_offsets.remainder(capacity)
            positions = torch.where(
                position_delta <= 0,
                end_offsets + position_delta,
                end_offsets + position_delta - capacity,
            )
            return torch.where(
                indexes[None] >= end_offsets,
                torch.full_like(positions, -1),
                positions,
            )
        else:
            pass
        slots = step_positions + state.end_offset
        state.keys.index_copy_(2, slots.remainder(capacity), k)
        state.values.index_copy_(2, slots.remainder(capacity), v)
        state.end_offset += k.shape[2]
        position_delta = indexes - state.end_offset % capacity
        positions = torch.where(
            position_delta <= 0,
            state.end_offset + position_delta,
            state.end_offset + position_delta - capacity,
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
    offset: int | torch.Tensor = 0
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
