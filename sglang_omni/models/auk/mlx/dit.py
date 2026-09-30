# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Tencent. All rights reserved.
"""Native MLX AuK generation backbone."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from sglang_omni.models.auk.mlx.blocks import (
    AdaLayerNormFinal,
    AudioPromptEmbedding,
    DiTBlock,
    MMDiTBlock,
    RotaryEmbedding,
    TimestepEmbedding,
)


class AuKDit(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 1024,
        heads: int = 16,
        dim_head: int = 64,
        ff_mult: float = 2.0,
        latent_dim: int = 64,
        text_hidden_dim: int = 2048,
        num_layers: int = 8,
        num_single_layers: int = 24,
        attn_mask_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.attn_mask_enabled = attn_mask_enabled
        self.time_embed = TimestepEmbedding(dim)
        self.txt_norm = nn.RMSNorm(dim, eps=mx.finfo(mx.float32).eps)
        self.txt_proj = nn.Linear(text_hidden_dim, dim)
        self.audio_embed = AudioPromptEmbedding(latent_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.transformer_blocks = [
            MMDiTBlock(dim, heads, dim_head, ff_mult) for _ in range(num_layers)
        ]
        self.single_transformer_blocks = [
            DiTBlock(dim, heads, dim_head, ff_mult) for _ in range(num_single_layers)
        ]
        self.norm_out = AdaLayerNormFinal(dim)
        self.proj_out = nn.Linear(dim, latent_dim)

    @property
    def dtype(self) -> mx.Dtype:
        return self.proj_out.weight.dtype

    def project_text(self, text: mx.array) -> mx.array:
        return self.txt_norm(self.txt_proj(text))

    def prepend_reference(
        self,
        x: mx.array,
        ref: mx.array | None,
        drop_audio_cond: bool,
        mask: mx.array | None,
        ref_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array | None, int]:
        if ref is None or ref.shape[1] == 0:
            return x, mask, 0
        else:
            prompt_len = ref.shape[1]
            ref_emb = self.audio_embed(
                mx.zeros_like(ref) if drop_audio_cond else ref, ref_mask
            )
            audio_mask = mx.ones(x.shape[:2], dtype=mx.bool_) if mask is None else mask
            prompt_mask = (
                mx.ones(ref.shape[:2], dtype=mx.bool_) if ref_mask is None else ref_mask
            )
            return (
                mx.concatenate([ref_emb, x], axis=1),
                mx.concatenate([prompt_mask, audio_mask], axis=1),
                prompt_len,
            )

    def __call__(
        self,
        x: mx.array,
        text: mx.array,
        time: mx.array,
        mask: mx.array | None = None,
        c_mask: mx.array | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        ref: mx.array | None = None,
        ref_mask: mx.array | None = None,
        audio_positions: mx.array | None = None,
        joint_positions: mx.array | None = None,
        projected_text: mx.array | None = None,
    ) -> mx.array:
        time = mx.broadcast_to(time, (x.shape[0],)) if time.ndim == 0 else time
        t = self.time_embed(time)
        c_mask = mx.sum(mx.abs(text), axis=-1) > 0 if c_mask is None else c_mask
        projected_text = (
            self.project_text(text) if projected_text is None else projected_text
        )
        x = self.audio_embed(x, mask)
        if cfg_infer:
            x_cond, audio_mask, prompt_len = self.prepend_reference(
                x, ref, False, mask, ref_mask
            )
            x_uncond, _, _ = self.prepend_reference(x, ref, True, mask, ref_mask)
            x = mx.concatenate([x_cond, x_uncond], axis=0)
            c = mx.concatenate([projected_text, mx.zeros_like(projected_text)], axis=0)
            t = mx.concatenate([t, t], axis=0)
            if audio_mask is not None:
                audio_mask = mx.concatenate([audio_mask, audio_mask], axis=0)
            else:
                pass
            c_mask = mx.concatenate([c_mask, c_mask], axis=0)
            if audio_positions is not None:
                audio_positions = mx.concatenate(
                    [audio_positions, audio_positions], axis=0
                )
            else:
                pass
            if joint_positions is not None:
                joint_positions = mx.concatenate(
                    [joint_positions, joint_positions], axis=0
                )
            else:
                pass
        else:
            c = mx.zeros_like(projected_text) if drop_text else projected_text
            x, audio_mask, prompt_len = self.prepend_reference(
                x, ref, drop_audio_cond, mask, ref_mask
            )

        seq_len, text_len = x.shape[1], c.shape[1]
        rope_audio = self.rotary_embed(
            mx.arange(seq_len) if audio_positions is None else audio_positions
        )
        rope_text = self.rotary_embed(mx.arange(text_len))
        joint_bias = single_bias = single_mask = None
        if audio_mask is not None:
            single_mask = mx.concatenate([c_mask, audio_mask], axis=1)
            if self.attn_mask_enabled:
                joint_bias = mx.concatenate([audio_mask, c_mask], axis=1)[
                    :, None, None, :
                ]
                single_bias = single_mask[:, None, None, :]
            else:
                pass
        else:
            pass
        for block in self.transformer_blocks:
            c, x = block(x, c, t, rope_audio, rope_text, audio_mask, c_mask, joint_bias)
        x = mx.concatenate([c, x], axis=1)
        rope = self.rotary_embed(
            mx.arange(text_len + seq_len)
            if joint_positions is None
            else joint_positions
        )
        for block in self.single_transformer_blocks:
            x = block(x, t, rope, single_mask, single_bias)
        return self.proj_out(self.norm_out(x[:, text_len + prompt_len :], t))
