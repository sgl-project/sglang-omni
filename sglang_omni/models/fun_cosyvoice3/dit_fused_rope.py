# SPDX-License-Identifier: Apache-2.0
"""Opt-in partial Q/K RoPE fusion for the CosyVoice3 Flow DiT.

CosyVoice applies RoPE to [B, T, 1024] *before* splitting heads. Only the
first 64 channels rotate; adding Q/K norm or rotating every head would change
the model. Trig tables are local to each DiT forward and shared by its blocks.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class _RopeTables(NamedTuple):
    cos: torch.Tensor
    sin: torch.Tensor


class _RotaryTablesForward:
    def __init__(self, native_forward):
        self.native_forward = native_forward

    def __call__(self, seq_len):
        freqs, _ = self.native_forward(seq_len)
        return _RopeTables(freqs.cos().contiguous(), freqs.sin().contiguous())


class _FusedRopeAttnProcessor:
    def __init__(self, fused_rope):
        self.fused_rope = fused_rope

    def __call__(self, attn, x, mask=None, rope=None):
        # Keep projections, SDPA, and output masking identical to CosyVoice's
        # AttnProcessor. Only the two apply_rotary_pos_emb calls are replaced.
        query, key, value = attn.to_q(x), attn.to_k(x), attn.to_v(x)
        if rope is not None:
            query, key = self.fused_rope(query, key, rope.cos, rope.sin)

        batch = x.shape[0]
        head_dim = attn.inner_dim // attn.heads
        query = query.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, attn.heads, head_dim).transpose(1, 2)

        attn_mask = mask
        if mask is not None and mask.dim() == 2:
            attn_mask = mask[:, None, None, :].expand(
                batch, attn.heads, query.shape[-2], key.shape[-2]
            )
        x = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        x = x.transpose(1, 2).reshape(batch, -1, attn.inner_dim).to(query.dtype)
        x = attn.to_out[1](attn.to_out[0](x))
        if mask is not None:
            output_mask = mask if mask.dim() == 2 else mask[:, 0, -1]
            x = x.masked_fill(~output_mask.unsqueeze(-1), 0.0)
        return x


def install_dit_fused_rope(estimator: torch.nn.Module) -> None:
    """Install after loading weights, before torch.compile; leave state keys intact."""
    param = next(estimator.parameters())
    if param.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("enable_dit_fused_rope requires NVIDIA CUDA")

    from cosyvoice.flow.DiT.modules import AttnProcessor

    rotary = estimator.rotary_embed
    if isinstance(rotary.forward_from_seq_len, _RotaryTablesForward):
        return
    # The released model uses FP32 frequencies, including under BF16 autocast.
    # Half-weight loading also halves inv_freq and has different rounding.
    if (
        rotary.inv_freq.dtype != torch.float32
        or rotary.inv_freq.numel() != 32
        or rotary.scale is not None
    ):
        raise ValueError("Fused DiT RoPE requires 64-D FP32 frequencies without XPos")
    attentions = [block.attn for block in estimator.transformer_blocks]
    if not attentions or any(
        type(attn.processor) is not AttnProcessor
        or attn.inner_dim != 1024
        or attn.heads != 16
        for attn in attentions
    ):
        raise ValueError("Fused DiT RoPE requires the CosyVoice3 attention layout")

    # No Triton import on the default path or on unsupported platforms.
    from sglang_omni.models.fun_cosyvoice3.dit_fused_rope_kernel import fused_qk_rope

    rotary.forward_from_seq_len = _RotaryTablesForward(rotary.forward_from_seq_len)
    for attn in attentions:
        attn.processor = _FusedRopeAttnProcessor(fused_qk_rope)
