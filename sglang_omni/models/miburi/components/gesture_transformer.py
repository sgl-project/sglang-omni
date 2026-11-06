# SPDX-License-Identifier: Apache-2.0
"""The gesture LM's two decoder stacks.

Every layer is pre-norm: causal self-attention, then attention over the agent frame's audio and text memories fused by a SiLU projection and a gated unit, then a gated feed-forward. The temporal stack takes one step per 80 ms frame and keeps its rings across frames: 25 frames of its own states, 50 of each memory. The depth stack takes one step per codebook inside a frame, with its own weights for each step, and starts every frame empty; its memory is the current frame's only, so attending to it reduces to projecting it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional

from sglang_omni.models.miburi.architecture import GestureLMSpec
from sglang_omni.models.miburi.components.ring_attention import (
    AttentionRing,
    attend_ring,
)


class GatedFeedForward(nn.Module):
    """linear_out(silu(gate) * up), with gate and up the two halves of linear_in."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False)
        self.linear_out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.linear_in(x).chunk(2, dim=-1)
        return self.linear_out(functional.silu(gate) * up)


class GestureAttention(nn.Module):
    """Attention of a query step over a ring of key/value steps; weight set i serves depth step i."""

    def __init__(
        self, dim: int, num_heads: int, weight_sets: int, rope_max_period: float
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.rope_max_period = rope_max_period

        def projections() -> nn.ModuleList:
            return nn.ModuleList(
                nn.Linear(dim, dim, bias=False) for _ in range(weight_sets)
            )

        self.query_projs = projections()
        self.key_projs = projections()
        self.value_projs = projections()
        self.out_projs = projections()

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        positions: torch.Tensor,
        ring: AttentionRing,
        weight_set: int,
    ) -> torch.Tensor:
        q, k, v = (
            rearrange(
                projections[weight_set](source),
                "b t (h d) -> b h t d",
                h=self.num_heads,
            )
            for projections, source in (
                (self.query_projs, query),
                (self.key_projs, memory),
                (self.value_projs, memory),
            )
        )
        attended = attend_ring(q, k, v, positions, ring, self.rope_max_period)
        return self.out_projs[weight_set](attended)


class CurrentMemoryAttention(nn.Module):
    """Attention whose one key is the current step's memory: its weight is 1, so the output is the projected value; a masked step gives zero."""

    def __init__(self, dim: int, weight_sets: int) -> None:
        super().__init__()
        self.value_projs = nn.ModuleList(
            nn.Linear(dim, dim, bias=False) for _ in range(weight_sets)
        )
        self.out_projs = nn.ModuleList(
            nn.Linear(dim, dim, bias=False) for _ in range(weight_sets)
        )

    def forward(
        self, memory: torch.Tensor, weight_set: int, is_masked: bool
    ) -> torch.Tensor:
        if is_masked:
            return torch.zeros_like(memory)
        else:
            return self.out_projs[weight_set](self.value_projs[weight_set](memory))


class GestureDecoderLayer(nn.Module):
    """What both stacks' layers share: self-attention, the memory fusion and the second norm."""

    def __init__(self, spec: GestureLMSpec, num_heads: int, weight_sets: int) -> None:
        super().__init__()
        self.self_attn = GestureAttention(
            spec.dim, num_heads, weight_sets, spec.rope_max_period
        )
        self.norm1 = nn.LayerNorm(spec.dim)
        self.norm2 = nn.LayerNorm(spec.dim)
        # Over the attended audio memory, then the text.
        self.memory_fusion = nn.Linear(2 * spec.dim, spec.dim)

    def fuse_and_feed_forward(
        self,
        x: torch.Tensor,
        attended_audio: torch.Tensor,
        attended_text: torch.Tensor,
        cross_attn_feed_forward: GatedFeedForward,
        feed_forward: GatedFeedForward,
    ) -> torch.Tensor:
        fused = functional.silu(
            self.memory_fusion(torch.cat([attended_audio, attended_text], dim=-1))
        )
        x = x + cross_attn_feed_forward(fused)
        return x + feed_forward(self.norm2(x))


class TemporalDecoderLayer(GestureDecoderLayer):
    def __init__(self, spec: GestureLMSpec) -> None:
        super().__init__(spec, spec.temporal_heads, weight_sets=1)
        self.cross_attns = nn.ModuleList(
            GestureAttention(spec.dim, spec.temporal_heads, 1, spec.rope_max_period)
            for _ in range(2)
        )
        self.cross_attn_norms = nn.ModuleList(
            [nn.LayerNorm(spec.dim), nn.LayerNorm(spec.dim)]
        )
        self.cross_attn_feed_forward = GatedFeedForward(spec.dim, spec.ffn_dim)
        self.feed_forward = GatedFeedForward(spec.dim, spec.ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        memories: tuple[torch.Tensor, torch.Tensor],
        positions: torch.Tensor,
        rings: TemporalLayerRings,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.self_attn(normed, normed, positions, rings.self_attn, 0)
        audio_query = self.cross_attn_norms[0](x)
        # The text query normalizes the audio query again, not x.
        text_query = self.cross_attn_norms[1](audio_query)
        return self.fuse_and_feed_forward(
            x,
            self.cross_attns[0](audio_query, memories[0], positions, rings.audio, 0),
            self.cross_attns[1](text_query, memories[1], positions, rings.text, 0),
            self.cross_attn_feed_forward,
            self.feed_forward,
        )


class DepthDecoderLayer(GestureDecoderLayer):
    def __init__(self, spec: GestureLMSpec) -> None:
        super().__init__(spec, spec.depth_heads, weight_sets=spec.depth_steps)
        self.cross_attns = nn.ModuleList(
            CurrentMemoryAttention(spec.dim, spec.depth_steps) for _ in range(2)
        )
        self.cross_attn_feed_forward = nn.ModuleList(
            GatedFeedForward(spec.dim, spec.ffn_dim) for _ in range(spec.depth_steps)
        )
        self.feed_forward = nn.ModuleList(
            GatedFeedForward(spec.dim, spec.ffn_dim) for _ in range(spec.depth_steps)
        )

    def forward(
        self,
        x: torch.Tensor,
        memories: tuple[torch.Tensor, torch.Tensor],
        step: int,
        positions: torch.Tensor,
        ring: AttentionRing,
        is_memory_masked: bool,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.self_attn(normed, normed, positions, ring, step)
        return self.fuse_and_feed_forward(
            x,
            self.cross_attns[0](memories[0], step, is_memory_masked),
            self.cross_attns[1](memories[1], step, is_memory_masked),
            self.cross_attn_feed_forward[step],
            self.feed_forward[step],
        )


@dataclass
class TemporalLayerRings:
    self_attn: AttentionRing
    audio: AttentionRing
    text: AttentionRing


@dataclass
class TemporalState:
    """Frames each row has taken [rows] and every layer's rings."""

    positions: torch.Tensor
    layers: list[TemporalLayerRings]

    def row_tensors(self) -> list[torch.Tensor]:
        return [
            self.positions,
            *(
                tensor
                for layer in self.layers
                for ring in (layer.self_attn, layer.audio, layer.text)
                for tensor in (ring.keys, ring.values)
            ),
        ]


class TemporalTransformer(nn.Module):
    def __init__(self, spec: GestureLMSpec) -> None:
        super().__init__()
        self.spec = spec
        self.layers = nn.ModuleList(
            TemporalDecoderLayer(spec) for _ in range(spec.temporal_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        memories: tuple[torch.Tensor, torch.Tensor],
        state: TemporalState,
    ) -> torch.Tensor:
        for layer, rings in zip(self.layers, state.layers, strict=True):
            x = layer(x, memories, state.positions, rings)
        state.positions.add_(1)
        return x

    def new_state(self, rows: int, device: torch.device) -> TemporalState:
        spec = self.spec
        head_dim = spec.dim // spec.temporal_heads

        def ring(capacity: int) -> AttentionRing:
            return AttentionRing.zeros(
                rows, spec.temporal_heads, capacity, head_dim, device=device
            )

        return TemporalState(
            positions=torch.zeros(rows, dtype=torch.long, device=device),
            layers=[
                TemporalLayerRings(
                    self_attn=ring(spec.self_context),
                    audio=ring(spec.memory_context),
                    text=ring(spec.memory_context),
                )
                for _ in self.layers
            ],
        )


class DepthTransformer(nn.Module):
    def __init__(self, spec: GestureLMSpec) -> None:
        super().__init__()
        self.spec = spec
        self.layers = nn.ModuleList(
            DepthDecoderLayer(spec) for _ in range(spec.depth_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        memories: tuple[torch.Tensor, torch.Tensor],
        step: int,
        rings: list[AttentionRing],
        is_memory_masked: bool,
    ) -> torch.Tensor:
        positions = torch.full((x.shape[0],), step, dtype=torch.long, device=x.device)
        for layer, ring in zip(self.layers, rings, strict=True):
            x = layer(x, memories, step, positions, ring, is_memory_masked)
        return x

    def new_rings(self, rows: int, device: torch.device) -> list[AttentionRing]:
        """Empty rings for one frame's steps."""
        spec = self.spec
        return [
            AttentionRing.zeros(
                rows,
                spec.depth_heads,
                spec.depth_steps,
                spec.dim // spec.depth_heads,
                device=device,
            )
            for _ in self.layers
        ]


__all__ = [
    "DepthTransformer",
    "TemporalState",
    "TemporalTransformer",
]
