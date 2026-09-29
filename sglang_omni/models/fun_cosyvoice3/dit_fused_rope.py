# SPDX-License-Identifier: Apache-2.0
"""Partial Q/K RoPE fusion for the CosyVoice3 Flow DiT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


class RotaryForward(Protocol):
    def __call__(self, sequence_length: int) -> tuple[torch.Tensor, float]: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class RopeTables:
    cosine: torch.Tensor
    sine: torch.Tensor

    def apply(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # note (wirybeaver): Keep Triton optional for non-CUDA imports.
        from sglang_omni.models.fun_cosyvoice3.dit_fused_rope_kernel import (
            fused_qk_rope,
        )

        return fused_qk_rope(query, key, self.cosine, self.sine)


class RotaryTablesForward:
    def __init__(self, native_forward: RotaryForward) -> None:
        self.native_forward: RotaryForward = native_forward

    def __call__(self, sequence_length: int) -> RopeTables:
        rotary_frequencies, _ = self.native_forward(sequence_length)
        return RopeTables(
            cosine=rotary_frequencies.cos().contiguous(),
            sine=rotary_frequencies.sin().contiguous(),
        )


class FusedRopeAttentionProcessor:
    def __call__(
        self,
        attention: torch.nn.Module,
        x: torch.Tensor,
        mask: torch.Tensor,
        rope: RopeTables,
    ) -> torch.Tensor:
        query, key, value = attention.to_q(x), attention.to_k(x), attention.to_v(x)
        query, key = rope.apply(query, key)

        batch_size = x.shape[0]
        head_dimension = attention.inner_dim // attention.heads
        query = query.view(batch_size, -1, attention.heads, head_dimension).transpose(
            1, 2
        )
        key = key.view(batch_size, -1, attention.heads, head_dimension).transpose(1, 2)
        value = value.view(batch_size, -1, attention.heads, head_dimension).transpose(
            1, 2
        )

        x = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        x = x.transpose(1, 2).reshape(batch_size, -1, attention.inner_dim)
        x = attention.to_out[1](attention.to_out[0](x.to(query.dtype)))
        return x.masked_fill(~mask[:, 0, -1].unsqueeze(-1), 0.0)


def install_dit_fused_rope(dit_estimator: torch.nn.Module) -> None:
    """Install after loading weights, before torch.compile; leave state keys intact."""
    rotary_embedding = dit_estimator.rotary_embed
    rotary_embedding.forward_from_seq_len = RotaryTablesForward(
        rotary_embedding.forward_from_seq_len
    )
    for transformer_block in dit_estimator.transformer_blocks:
        transformer_block.attn.processor = FusedRopeAttentionProcessor()


__all__ = ["RopeTables", "install_dit_fused_rope"]
