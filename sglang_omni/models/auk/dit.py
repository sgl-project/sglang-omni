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

from sglang_omni.models.auk.modulation import gated_residual, scale_shift


def _attention_bias(key_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive SDPA bias ``[B, 1, 1, K]`` from a boolean key-padding mask ``[B, K]``."""
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

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
            emb, 6, dim=1
        )

        x = scale_shift(self.norm(x), scale_msa, shift_msa)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(nn.Module):
    """Final timestep-conditioned LayerNorm (no MLP branch to modulate)."""

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return scale_shift(self.norm(x), scale, shift)


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
    def _split_heads(t: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
        batch = t.shape[0]
        return t.view(batch, -1, heads, head_dim).transpose(1, 2)

    @staticmethod
    def _apply_rope(
        q: torch.Tensor, k: torch.Tensor, rope
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs, xpos_scale = rope
        q_scale, k_scale = (
            (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
        )
        return (
            apply_rotary_pos_emb(q, freqs, q_scale),
            apply_rotary_pos_emb(k, freqs, k_scale),
        )

    @staticmethod
    def _attend(q, k, v, bias: torch.Tensor | None):
        """SDPA with an optional additive key bias ``[B, 1, 1, K]``, then merge heads."""
        batch = q.shape[0]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False
        )
        return out.transpose(1, 2).reshape(batch, -1, q.shape[1] * q.shape[3])

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        rope=None,
        c_rope=None,
        c_mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ):
        if c is None:
            return self._forward_self(x, mask=mask, rope=rope, bias=bias)

        audio_mask = mask
        query, key, value = self.to_qkv(x).chunk(3, dim=-1)
        c_query, c_key, c_value = self.to_qkv_c(c).chunk(3, dim=-1)

        head_dim = key.shape[-1] // self.heads
        query = self._split_heads(query, self.heads, head_dim)
        key = self._split_heads(key, self.heads, head_dim)
        value = self._split_heads(value, self.heads, head_dim)
        c_query = self._split_heads(c_query, self.heads, head_dim)
        c_key = self._split_heads(c_key, self.heads, head_dim)
        c_value = self._split_heads(c_value, self.heads, head_dim)

        query, key = self.q_norm(query), self.k_norm(key)
        c_query, c_key = self.c_q_norm(c_query), self.c_k_norm(c_key)
        if rope is not None:
            query, key = self._apply_rope(query, key, rope)
        if c_rope is not None:
            c_query, c_key = self._apply_rope(c_query, c_key, c_rope)

        out = self._attend(
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

    def _forward_self(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        rope=None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value = self.to_qkv(x).chunk(3, dim=-1)
        head_dim = key.shape[-1] // self.heads
        query = self._split_heads(query, self.heads, head_dim)
        key = self._split_heads(key, self.heads, head_dim)
        value = self._split_heads(value, self.heads, head_dim)

        query, key = self.q_norm(query), self.k_norm(key)
        if rope is not None:
            query, key = self._apply_rope(query, key, rope)

        out = self._attend(query, key, value, bias).to(query.dtype)
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
        rope=None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        x = gated_residual(
            x, gate_msa, self.attn(x=norm, mask=mask, rope=rope, bias=bias)
        )

        norm = scale_shift(self.ff_norm(x), scale_mlp, shift_mlp)
        return gated_residual(x, gate_mlp, self.ff(norm))


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
        rope=None,
        c_rope=None,
        c_mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ):
        norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(
            c, emb=t
        )
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(
            x, emb=t
        )
        x_attn, c_attn = self.attn(
            x=norm_x,
            c=norm_c,
            mask=mask,
            rope=rope,
            c_rope=c_rope,
            c_mask=c_mask,
            bias=bias,
        )

        c = gated_residual(c, c_gate_msa, c_attn)
        norm_c = scale_shift(self.ff_norm_c(c), c_scale_mlp, c_shift_mlp)
        c = gated_residual(c, c_gate_mlp, self.ff_c(norm_c))

        x = gated_residual(x, x_gate_msa, x_attn)
        norm_x = scale_shift(self.ff_norm_x(x), x_scale_mlp, x_shift_mlp)
        x = gated_residual(x, x_gate_mlp, self.ff_x(norm_x))
        return c, x


class AudioPromptEmbedding(nn.Module):
    """Linear + conv position embedding shared by the noised and prompt latents."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def _embed(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.linear(x)
        return self.conv_pos_embed(x, mask=mask) + x

    def forward(
        self,
        x: torch.Tensor,
        ref: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        mask: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
    ):
        x_emb = self._embed(x, mask=mask)
        if ref is None:
            return x_emb
        if drop_audio_cond:
            ref = torch.zeros_like(ref)
        return x_emb, self._embed(ref, mask=ref_mask)


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

        self.text_cond: torch.Tensor | None = None
        self.text_uncond: torch.Tensor | None = None

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

    def clear_cache(self) -> None:
        self.text_cond, self.text_uncond = None, None

    @property
    def dtype(self) -> torch.dtype:
        return self.proj_out.weight.dtype

    def project_text(self, text: torch.Tensor, drop_text: bool = False) -> torch.Tensor:
        c = self.txt_norm(self.txt_proj(text))
        return torch.zeros_like(c) if drop_text else c

    def _embed_audio(
        self,
        x: torch.Tensor,
        ref: torch.Tensor | None,
        drop_audio_cond: bool,
        mask: torch.Tensor | None,
        ref_mask: torch.Tensor | None,
    ):
        if ref is not None and ref.shape[1] == 0:
            ref = None
            ref_mask = None

        if ref is None:
            return (
                self.audio_embed(x, drop_audio_cond=drop_audio_cond, mask=mask),
                mask,
                0,
            )

        x_emb, ref_emb = self.audio_embed(
            x,
            ref=ref,
            drop_audio_cond=drop_audio_cond,
            mask=mask,
            ref_mask=ref_mask,
        )
        prompt_len = ref_emb.shape[1]
        audio = torch.cat([ref_emb, x_emb], dim=1)

        batch, n = x_emb.shape[:2]
        if mask is None:
            mask = torch.ones(batch, n, dtype=torch.bool, device=x_emb.device)
        if ref_mask is None:
            ref_mask = torch.ones(
                batch, prompt_len, dtype=torch.bool, device=ref_emb.device
            )
        return audio, torch.cat([ref_mask, mask], dim=1), prompt_len

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        mask: torch.Tensor | None = None,
        c_mask: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        ref: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
        audio_positions: torch.Tensor | None = None,
        joint_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)
        t = self.time_embed(time)

        if c_mask is None:
            c_mask = text.abs().sum(-1) > 0

        if cfg_infer:
            if cache and self.text_cond is not None:
                c_cond = self.text_cond
            else:
                c_cond = self.project_text(text, drop_text=False)
                if cache:
                    self.text_cond = c_cond
            x_cond, a_mask_cond, prompt_len = self._embed_audio(
                x, ref, drop_audio_cond=False, mask=mask, ref_mask=ref_mask
            )

            if cache and self.text_uncond is not None:
                c_uncond = self.text_uncond
            else:
                c_uncond = self.project_text(text, drop_text=True)
                if cache:
                    self.text_uncond = c_uncond
            x_uncond, a_mask_uncond, _ = self._embed_audio(
                x, ref, drop_audio_cond=True, mask=mask, ref_mask=ref_mask
            )

            x = torch.cat((x_cond, x_uncond), dim=0)
            c = torch.cat((c_cond, c_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            audio_mask = (
                torch.cat((a_mask_cond, a_mask_uncond), dim=0)
                if a_mask_cond is not None and a_mask_uncond is not None
                else None
            )
            c_mask = torch.cat((c_mask, c_mask), dim=0)
            if audio_positions is not None:
                audio_positions = audio_positions.repeat(2, 1)
                joint_positions = joint_positions.repeat(2, 1)
        else:
            c = self.project_text(text, drop_text=drop_text)
            x, audio_mask, prompt_len = self._embed_audio(
                x, ref, drop_audio_cond=drop_audio_cond, mask=mask, ref_mask=ref_mask
            )

        seq_len = x.shape[1]
        text_len = c.shape[1]
        rope_audio = (
            self.rotary_embed.forward_from_seq_len(seq_len)
            if audio_positions is None
            else self.rotary_embed(audio_positions)
        )
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)

        joint_bias = single_bias = single_mask = None
        if audio_mask is not None:
            single_mask = torch.cat([c_mask, audio_mask], dim=1)
            if self.attn_mask_enabled:
                joint_bias = _attention_bias(
                    torch.cat([audio_mask, c_mask], dim=1), x.dtype
                )
                single_bias = _attention_bias(single_mask, x.dtype)

        for block in self.transformer_blocks:
            c, x = block(
                x,
                c,
                t,
                mask=audio_mask,
                rope=rope_audio,
                c_rope=rope_text,
                c_mask=c_mask,
                bias=joint_bias,
            )

        x = torch.cat([c, x], dim=1)
        rope = (
            self.rotary_embed.forward_from_seq_len(text_len + seq_len)
            if joint_positions is None
            else self.rotary_embed(joint_positions)
        )
        for block in self.single_transformer_blocks:
            x = block(x, t, mask=single_mask, rope=rope, bias=single_bias)

        x = x[:, text_len + prompt_len :]
        return self.proj_out(self.norm_out(x, t))
