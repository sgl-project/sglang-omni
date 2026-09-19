# SPDX-License-Identifier: Apache-2.0
"""Partial Q/K RoPE fusion for the CosyVoice3 Flow DiT."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from cosyvoice.flow.DiT.dit import DiT
    from cosyvoice.flow.DiT.modules import Attention

_PROJECTED_DIM = 1024
_ROTARY_DIM = 64
_FusedRope = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]


@dataclass(frozen=True, kw_only=True, slots=True)
class _RopeTables:
    cos: torch.Tensor
    sin: torch.Tensor


class _RotaryTablesForward:
    def __init__(
        self, native_forward: Callable[[int], tuple[torch.Tensor, float]]
    ) -> None:
        self.native_forward = native_forward

    def __call__(self, seq_len: int) -> _RopeTables:
        freqs, _ = self.native_forward(seq_len)
        return _RopeTables(cos=freqs.cos().contiguous(), sin=freqs.sin().contiguous())


class _FusedRopeAttnProcessor:
    def __init__(self, fused_rope: _FusedRope) -> None:
        self.fused_rope = fused_rope

    def __call__(
        self,
        attn: Attention,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: _RopeTables | None = None,
    ) -> torch.Tensor:
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


def install_dit_fused_rope(estimator: DiT) -> None:
    """Install after loading weights, before torch.compile; leave state keys intact."""
    param = next(estimator.parameters())
    if param.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("enable_dit_fused_rope requires NVIDIA CUDA")

    from cosyvoice.flow.DiT.modules import AttnProcessor

    rotary = estimator.rotary_embed
    # note (wirybeaver): Half-weight loading changes the native frequency rounding.
    if (
        rotary.inv_freq.dtype != torch.float32
        or rotary.inv_freq.numel() != _ROTARY_DIM // 2
        or rotary.scale is not None
    ):
        raise ValueError("Fused DiT RoPE requires 64-D FP32 frequencies without XPos")
    attentions = [block.attn for block in estimator.transformer_blocks]
    if not attentions or any(
        type(attn.processor) is not AttnProcessor
        or attn.inner_dim != _PROJECTED_DIM
        or attn.heads * _ROTARY_DIM != _PROJECTED_DIM
        for attn in attentions
    ):
        raise ValueError("Fused DiT RoPE requires the CosyVoice3 attention layout")

    # note (wirybeaver): Keep Triton optional for non-CUDA imports.
    from sglang_omni.models.fun_cosyvoice3.dit_fused_rope_kernel import fused_qk_rope

    rotary.forward_from_seq_len = _RotaryTablesForward(rotary.forward_from_seq_len)
    for attn in attentions:
        attn.processor = _FusedRopeAttnProcessor(fused_qk_rope)


__all__ = ["install_dit_fused_rope"]
