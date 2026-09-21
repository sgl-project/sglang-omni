# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Tencent. All rights reserved.
# Derived from Tencent-Hunyuan/AuK; see LICENSE for the MIT permission notice.
"""AuK generation backbone."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

from sglang_omni.models.auk.packed import (
    PackedLayout,
    flash_attention,
    gather_rope,
    gather_rows,
)


def modulation(value: torch.Tensor, packed: bool) -> torch.Tensor:
    """Broadcast the trajectory's modulation row [1, D] over padded [B, T, D]
    or packed [rows, D] activations."""
    return value[0] if packed else value[:, None]


@dataclass(frozen=True)
class RopeTable:
    """Rotary frequencies of one token stream plus the trig tables the fused
    Q/K kernel reads, built once per trajectory."""

    freqs: torch.Tensor
    scale: torch.Tensor | float
    cos: torch.Tensor
    sin: torch.Tensor

    @classmethod
    def build(cls, freqs: torch.Tensor, scale: torch.Tensor | float) -> RopeTable:
        return cls(freqs, scale, freqs.cos(), freqs.sin())


def attention_bias(key_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive SDPA bias [B, 1, 1, K] from a boolean key-padding mask [B, K]."""
    # -inf is safe: every request has at least one valid text token and audio
    # frame, so no softmax row is fully masked.
    bias = torch.zeros(key_mask.shape, dtype=dtype, device=key_mask.device)
    bias.masked_fill_(~key_mask, float("-inf"))
    return bias[:, None, None, :]


class SinusPositionEmbedding(nn.Module):
    """Sinusoidal embedding used for the flow-matching timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvPositionEmbedding(nn.Module):
    """Depthwise causal-agnostic conv position embedding added to a sequence."""

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        assert kernel_size % 2 != 0, "kernel_size must be odd"
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )
        self.layer_need_mask_idx = [
            i for i, layer in enumerate(self.conv1d) if isinstance(layer, nn.Conv1d)
        ]

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)
        x = x.permute(0, 2, 1)

        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        for i, block in enumerate(self.conv1d):
            x = block(x)
            if mask is not None and i in self.layer_need_mask_idx:
                x = x.masked_fill(~mask, 0.0)

        return x.permute(0, 2, 1)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        # Sinusoid arguments reach ~1000 rad; keep them fp32, cast only for the MLP.
        time_hidden = self.time_embed(timestep.float())
        return self.time_mlp(time_hidden.to(self.time_mlp[0].weight.dtype))


class AdaLayerNorm(nn.Module):
    """Timestep-conditioned LayerNorm returning modulation params."""

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, packed: bool = False):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
            emb, 6, dim=1
        )

        x = self.norm(x) * (1 + modulation(scale_msa, packed)) + modulation(
            shift_msa, packed
        )
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(nn.Module):
    """Final timestep-conditioned LayerNorm (no MLP branch to modulate)."""

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor, packed: bool = False
    ) -> torch.Tensor:
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + modulation(scale, packed)) + modulation(
            shift, packed
        )


class SwiGLU(nn.Module):
    """Parameter-free SwiGLU activation (gate/value split of the FF inner dim)."""

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: float = 3.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.act_fn = SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_out(self.act_fn(self.linear_in(x)))


class Attention(nn.Module):
    """Self-attention or joint text/audio attention."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.context_dim = context_dim
        self.qk_fusion = None

        self.to_qkv = nn.Linear(dim, 3 * self.inner_dim)
        self.q_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
        self.k_norm = nn.RMSNorm(dim_head, elementwise_affine=True)

        if self.context_dim is not None:
            self.to_qkv_c = nn.Linear(context_dim, 3 * self.inner_dim)
            self.c_q_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
            self.c_k_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
            self.to_out_c = nn.Linear(self.inner_dim, context_dim)

        self.to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, dim), nn.Dropout(dropout)]
        )

    @staticmethod
    def split_heads(t: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
        batch = t.shape[0]
        return t.view(batch, -1, heads, head_dim).transpose(1, 2)

    @staticmethod
    def apply_rope(
        q: torch.Tensor, k: torch.Tensor, rope: RopeTable
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            apply_rotary_pos_emb(q, rope.freqs, rope.scale),
            apply_rotary_pos_emb(k, rope.freqs, rope.scale**-1.0),
        )

    @staticmethod
    def attend(q, k, v, bias: torch.Tensor | None):
        """SDPA with an optional additive key bias [B, 1, 1, K], then merge heads."""
        batch = q.shape[0]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False
        )
        return out.transpose(1, 2).reshape(batch, -1, q.shape[1] * q.shape[3])

    def norm_rope(self, q, k, q_norm, k_norm, rope: RopeTable):
        if self.qk_fusion is not None:
            return self.qk_fusion(q, k, q_norm, k_norm, rope)
        return self.apply_rope(q_norm(q), k_norm(k), rope)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        rope: RopeTable | None = None,
        c_rope: RopeTable | None = None,
        c_mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        packed_layout: PackedLayout | None = None,
    ):
        if packed_layout is not None:
            return self.forward_packed(x, c, rope, c_rope, packed_layout)
        elif c is None:
            return self.forward_self(x, mask=mask, rope=rope, bias=bias)
        else:
            return self.forward_joint(
                x, c, mask=mask, rope=rope, c_rope=c_rope, c_mask=c_mask, bias=bias
            )

    def forward_joint(self, x, c, *, mask, rope, c_rope, c_mask, bias):
        audio_mask = mask
        query, key, value = self.to_qkv(x).chunk(3, dim=-1)
        c_query, c_key, c_value = self.to_qkv_c(c).chunk(3, dim=-1)

        head_dim = key.shape[-1] // self.heads
        query = self.split_heads(query, self.heads, head_dim)
        key = self.split_heads(key, self.heads, head_dim)
        value = self.split_heads(value, self.heads, head_dim)
        c_query = self.split_heads(c_query, self.heads, head_dim)
        c_key = self.split_heads(c_key, self.heads, head_dim)
        c_value = self.split_heads(c_value, self.heads, head_dim)

        query, key = self.norm_rope(query, key, self.q_norm, self.k_norm, rope)
        c_query, c_key = self.norm_rope(
            c_query, c_key, self.c_q_norm, self.c_k_norm, c_rope
        )

        out = self.attend(
            torch.cat([query, c_query], dim=2),
            torch.cat([key, c_key], dim=2),
            torch.cat([value, c_value], dim=2),
            bias,
        ).to(query.dtype)

        x_out, c_out = out[:, : x.shape[1]], out[:, x.shape[1] :]
        x_out = self.to_out[1](self.to_out[0](x_out))
        c_out = self.to_out_c(c_out)

        if audio_mask is not None:
            x_out = x_out.masked_fill(~audio_mask.unsqueeze(-1), 0.0)
        if c_mask is not None:
            c_out = c_out.masked_fill(~c_mask.unsqueeze(-1), 0.0)
        return x_out, c_out

    def forward_packed(self, x, c, rope, c_rope, layout):
        """Attention over packed [tokens, D] rows bounded by layout.cu_seqlens."""

        def project(rows, linear, q_norm, k_norm, positions):
            q, k, v = linear(rows).view(rows.shape[0], 3, self.heads, -1).unbind(1)
            # norm_rope and the fused kernel take [B, heads, seq, dim]; the
            # packed rows are one sequence whose rope was gathered per row.
            q, k = self.norm_rope(
                q.transpose(0, 1)[None],
                k.transpose(0, 1)[None],
                q_norm,
                k_norm,
                positions,
            )
            return q[0].transpose(0, 1), k[0].transpose(0, 1), v

        q, k, v = project(x, self.to_qkv, self.q_norm, self.k_norm, rope)
        if c is None:
            out = flash_attention(q, k, v, layout).flatten(1).to(q.dtype)
            return self.to_out[1](self.to_out[0](out))
        else:
            cq, ck, cv = project(c, self.to_qkv_c, self.c_q_norm, self.c_k_norm, c_rope)
            # Interleave the audio and text streams request by request so each
            # request's joint sequence is contiguous for the varlen kernel.
            q, k, v = (
                torch.cat((a, b)).index_select(0, layout.double_order)
                for a, b in ((q, cq), (k, ck), (v, cv))
            )
            out = flash_attention(q, k, v, layout).flatten(1).to(q.dtype)
            out = out.index_select(0, layout.double_inverse)
            return self.to_out[1](self.to_out[0](out[: x.shape[0]])), self.to_out_c(
                out[x.shape[0] :]
            )

    def forward_self(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        rope: RopeTable,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value = self.to_qkv(x).chunk(3, dim=-1)
        head_dim = key.shape[-1] // self.heads
        query = self.split_heads(query, self.heads, head_dim)
        key = self.split_heads(key, self.heads, head_dim)
        value = self.split_heads(value, self.heads, head_dim)

        query, key = self.norm_rope(query, key, self.q_norm, self.k_norm, rope)

        out = self.attend(query, key, value, bias).to(query.dtype)
        out = self.to_out[1](self.to_out[0](out))
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        return out


class DiTBlock(nn.Module):
    """Single-stream block over the concatenated text and audio sequence."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: float = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = SwiGLUFeedForward(dim=dim, mult=ff_mult)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: RopeTable | None = None,
        bias: torch.Tensor | None = None,
        packed_layout: PackedLayout | None = None,
    ) -> torch.Tensor:
        packed = packed_layout is not None
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(
            x, emb=t, packed=packed
        )
        x = x + modulation(gate_msa, packed) * self.attn(
            x=norm, mask=mask, rope=rope, bias=bias, packed_layout=packed_layout
        )

        norm = self.ff_norm(x) * (1 + modulation(scale_mlp, packed)) + modulation(
            shift_mlp, packed
        )
        return x + modulation(gate_mlp, packed) * self.ff(norm)


class MMDiTBlock(nn.Module):
    """Double-stream block: separate audio/text streams, joint attention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: float = 4,
        dropout: float = 0.1,
        context_dim: int | None = None,
    ):
        super().__init__()
        if context_dim is None:
            context_dim = dim

        self.attn_norm_c = AdaLayerNorm(context_dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=context_dim,
        )
        self.ff_norm_c = nn.LayerNorm(context_dim, elementwise_affine=False, eps=1e-6)
        self.ff_c = SwiGLUFeedForward(dim=context_dim, mult=ff_mult)
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = SwiGLUFeedForward(dim=dim, mult=ff_mult)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: RopeTable | None = None,
        c_rope: RopeTable | None = None,
        c_mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        packed_layout: PackedLayout | None = None,
    ):
        packed = packed_layout is not None
        norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(
            c, emb=t, packed=packed
        )
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(
            x, emb=t, packed=packed
        )
        x_attn, c_attn = self.attn(
            x=norm_x,
            c=norm_c,
            mask=mask,
            rope=rope,
            c_rope=c_rope,
            c_mask=c_mask,
            bias=bias,
            packed_layout=packed_layout,
        )

        c = c + modulation(c_gate_msa, packed) * c_attn
        norm_c = self.ff_norm_c(c) * (1 + modulation(c_scale_mlp, packed)) + modulation(
            c_shift_mlp, packed
        )
        c = c + modulation(c_gate_mlp, packed) * self.ff_c(norm_c)

        x = x + modulation(x_gate_msa, packed) * x_attn
        norm_x = self.ff_norm_x(x) * (1 + modulation(x_scale_mlp, packed)) + modulation(
            x_shift_mlp, packed
        )
        x = x + modulation(x_gate_mlp, packed) * self.ff_x(norm_x)
        return c, x


class AudioPromptEmbedding(nn.Module):
    """Linear + conv position embedding shared by the noised and prompt latents."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.linear(x)
        return self.conv_pos_embed(x, mask=mask) + x


@dataclass(frozen=True, kw_only=True)
class AuKDitPlan:
    """Inputs one trajectory's Euler steps share, embedded once by
    AuKDit.prepare so that a step only embeds the noised latent.

    audio_mask covers the reference and target frames, single_mask the text
    and audio rows of the single-stream blocks; a packed plan carries its rows
    already gathered and no masks.
    """

    text: torch.Tensor
    text_mask: torch.Tensor | None
    ref: torch.Tensor | None
    target_mask: torch.Tensor | None
    audio_mask: torch.Tensor | None
    single_mask: torch.Tensor | None
    joint_bias: torch.Tensor | None
    single_bias: torch.Tensor | None
    rope_audio: RopeTable
    rope_text: RopeTable
    rope_joint: RopeTable
    time_embeddings: torch.Tensor
    packed_layout: PackedLayout | None
    cfg: bool
    prompt_len: int


@dataclass
class AuKDitConfig:
    """Backbone hyperparameters."""

    dim: int = 1024
    heads: int = 16
    dim_head: int = 64
    dropout: float = 0.1
    ff_mult: float = 2.0
    text_hidden_dim: int = 2048
    num_layers: int = 8
    num_single_layers: int = 24
    latent_dim: int = 64
    attn_mask_enabled: bool = True
    depth: int = 8

    @classmethod
    def from_dict(cls, config_dict: dict | None) -> AuKDitConfig:
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (config_dict or {}).items() if k in valid})


class AuKDit(nn.Module):
    """Flux2Edit backbone."""

    def __init__(
        self,
        *,
        dim: int = 1024,
        heads: int = 16,
        dim_head: int = 64,
        dropout: float = 0.1,
        ff_mult: float = 2.0,
        latent_dim: int = 64,
        text_hidden_dim: int = 2048,
        num_layers: int = 8,
        num_single_layers: int = 24,
        attn_mask_enabled: bool = True,
        **_ignored,
    ):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.attn_mask_enabled = attn_mask_enabled

        self.time_embed = TimestepEmbedding(dim)
        self.txt_norm = nn.RMSNorm(dim, elementwise_affine=True)
        self.txt_proj = nn.Linear(text_hidden_dim, dim)

        self.audio_embed = AudioPromptEmbedding(latent_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.transformer_blocks = nn.ModuleList(
            MMDiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                ff_mult=ff_mult,
            )
            for _ in range(num_layers)
        )
        self.single_transformer_blocks = nn.ModuleList(
            DiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(num_single_layers)
        )

        self.norm_out = AdaLayerNormFinal(dim)
        self.proj_out = nn.Linear(dim, latent_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm_x.linear.weight, 0)
            nn.init.constant_(block.attn_norm_x.linear.bias, 0)
            nn.init.constant_(block.attn_norm_c.linear.weight, 0)
            nn.init.constant_(block.attn_norm_c.linear.bias, 0)
        for block in self.single_transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    @property
    def dtype(self) -> torch.dtype:
        return self.proj_out.weight.dtype

    def prepare(
        self,
        *,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        target_mask: torch.Tensor | None,
        target_len: int,
        audio_positions: torch.Tensor | None,
        joint_positions: torch.Tensor | None,
        packed_layout: PackedLayout | None,
        time_grid: torch.Tensor,
        cfg: bool,
    ) -> AuKDitPlan:
        """Embed the inputs one trajectory's Euler steps share.

        ref may be zero-width. target_mask and the position rows are None for
        a singleton batch, which then runs exactly as upstream does. time_grid
        holds the steps' start times plus the final one.
        """
        batch = text.shape[0]
        c = self.txt_norm(self.txt_proj(text))
        prompt_len = ref.shape[1]
        if prompt_len == 0:
            ref_emb, audio_mask = None, target_mask
        else:
            ref_emb = self.audio_embed(ref, mask=ref_mask)
            if target_mask is None:
                target_mask_rows = torch.ones(
                    batch, target_len, dtype=torch.bool, device=text.device
                )
            else:
                target_mask_rows = target_mask
            audio_mask = torch.cat([ref_mask, target_mask_rows], dim=1)
        if cfg:
            # The unconditional twins drop the text and the reference; their
            # rows are stacked below the conditional ones.
            c = torch.cat((c, torch.zeros_like(c)), dim=0)
            if ref_emb is not None:
                ref_emb = torch.cat(
                    (ref_emb, self.audio_embed(torch.zeros_like(ref), mask=ref_mask)),
                    dim=0,
                )
            text_mask = text_mask.repeat(2, 1)
            if audio_mask is not None:
                audio_mask = audio_mask.repeat(2, 1)
            if audio_positions is not None:
                audio_positions = audio_positions.repeat(2, 1)
                joint_positions = joint_positions.repeat(2, 1)

        seq_len = prompt_len + target_len
        text_len = c.shape[1]
        rope_audio = (
            self.rotary_embed.forward_from_seq_len(seq_len)
            if audio_positions is None
            else self.rotary_embed(audio_positions)
        )
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)
        rope_joint = (
            self.rotary_embed.forward_from_seq_len(text_len + seq_len)
            if joint_positions is None
            else self.rotary_embed(joint_positions)
        )
        if packed_layout is not None:
            rows = packed_layout.batch
            rope_audio = gather_rope(rope_audio, packed_layout.audio_indices, rows)
            rope_text = gather_rope(rope_text, packed_layout.text_indices, rows)
            rope_joint = gather_rope(rope_joint, packed_layout.joint_indices, rows)
            c = gather_rows(c, packed_layout.text_indices)
            # Packed rows carry no padding, so there is nothing to mask.
            text_mask = audio_mask = single_mask = joint_bias = single_bias = None
        elif audio_mask is None:
            single_mask = joint_bias = single_bias = None
        else:
            single_mask = torch.cat([text_mask, audio_mask], dim=1)
            if self.attn_mask_enabled:
                joint_bias = attention_bias(
                    torch.cat([audio_mask, text_mask], dim=1), self.dtype
                )
                single_bias = attention_bias(single_mask, self.dtype)
            else:
                joint_bias = single_bias = None
        return AuKDitPlan(
            text=c,
            text_mask=text_mask,
            ref=ref_emb,
            target_mask=target_mask,
            audio_mask=audio_mask,
            single_mask=single_mask,
            joint_bias=joint_bias,
            single_bias=single_bias,
            rope_audio=RopeTable.build(*rope_audio),
            rope_text=RopeTable.build(*rope_text),
            rope_joint=RopeTable.build(*rope_joint),
            time_embeddings=self.time_embed(time_grid[:-1]),
            packed_layout=packed_layout,
            cfg=cfg,
            prompt_len=prompt_len,
        )

    def forward(
        self, x: torch.Tensor, time_embedding: torch.Tensor, plan: AuKDitPlan
    ) -> torch.Tensor:
        """Velocity of the noised latent x [B, T, latent_dim] at one step.

        time_embedding is that step's row of plan.time_embeddings; with cfg
        the result stacks the conditional rows above the unconditional ones.
        """
        layout = plan.packed_layout
        packed = layout is not None
        x = self.audio_embed(x, mask=plan.target_mask)
        if plan.cfg:
            x = torch.cat((x, x), dim=0)
        if plan.ref is not None:
            x = torch.cat([plan.ref, x], dim=1)
        if packed:
            x = gather_rows(x, layout.audio_indices)

        c = plan.text
        for block in self.transformer_blocks:
            c, x = block(
                x,
                c,
                time_embedding,
                mask=plan.audio_mask,
                rope=plan.rope_audio,
                c_rope=plan.rope_text,
                c_mask=plan.text_mask,
                bias=plan.joint_bias,
                packed_layout=layout,
            )

        if packed:
            x = torch.cat([c, x], dim=0).index_select(0, layout.single_order)
        else:
            x = torch.cat([c, x], dim=1)
        for block in self.single_transformer_blocks:
            x = block(
                x,
                time_embedding,
                mask=plan.single_mask,
                rope=plan.rope_joint,
                bias=plan.single_bias,
                packed_layout=layout,
            )

        if packed:
            x = x.index_select(0, layout.target_rows)
            x = self.proj_out(self.norm_out(x, time_embedding, packed=True))
            return layout.unpack_target(x)
        else:
            x = x[:, plan.text.shape[1] + plan.prompt_len :]
            return self.proj_out(self.norm_out(x, time_embedding))
