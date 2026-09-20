# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

if TYPE_CHECKING:
    from sglang_omni.platforms.interface import JointRopeInplaceKernel


@dataclass(frozen=True)
class RotaryInputs:
    cos_sin_cache: torch.Tensor
    positions: torch.Tensor
    kernel: JointRopeInplaceKernel


def validate_rotary_config(
    kernel: JointRopeInplaceKernel | None,
    *,
    num_heads: int,
    qk_norm: str | None,
    pe_attn_head: int | None,
    grad_checkpointing: bool = False,
) -> None:
    # Note(yzxiao): Ming's joint-RoPE path supports full-head rotation without
    # Q/K norm or gradient checkpointing; these are model integration limits.
    if kernel is None:
        return
    if (
        qk_norm is not None
        or pe_attn_head not in (None, num_heads)
        or grad_checkpointing
    ):
        raise ValueError(
            "Joint RoPE requires full-head rotation "
            "and no Q/K norm or gradient checkpointing; received "
            f"num_heads={num_heads}, qk_norm={qk_norm!r}, "
            f"pe_attn_head={pe_attn_head!r}, grad_checkpointing={grad_checkpointing}"
        )


class CachedRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        *,
        kernel: JointRopeInplaceKernel,
        seq_len: int,
        max_batch_size: int,
    ) -> None:
        with torch.autocast(device_type="cuda", enabled=False):
            super().__init__(dim)
            freqs, _ = self.forward_from_seq_len(seq_len)
            freqs = freqs.reshape(seq_len, dim)
            phase = freqs[:, 0::2]
            cache = torch.cat((phase.cos(), phase.sin()), dim=-1).contiguous()

        self.kernel = kernel
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self.register_buffer(
            "positions",
            torch.arange(
                seq_len, device=self.inv_freq.device, dtype=torch.int64
            ).repeat(max_batch_size),
            persistent=False,
        )

    def for_batch(self, batch_size: int) -> RotaryInputs:
        seq_len = self.cos_sin_cache.shape[0]
        num_tokens = batch_size * seq_len
        positions = self.positions.narrow(0, 0, num_tokens)
        return RotaryInputs(self.cos_sin_cache, positions, self.kernel)


def build_rotary_embedding(
    dim: int,
    *,
    kernel: JointRopeInplaceKernel | None = None,
    seq_len: int | None = None,
    max_batch_size: int | None = None,
) -> RotaryEmbedding:
    if kernel is None:
        return RotaryEmbedding(dim)
    return CachedRotaryEmbedding(
        dim, kernel=kernel, seq_len=seq_len, max_batch_size=max_batch_size
    )


def get_rotary_inputs(
    rotary: RotaryEmbedding, batch_size: int, seq_len: int
) -> RotaryInputs | tuple[torch.Tensor, float | torch.Tensor]:
    if isinstance(rotary, CachedRotaryEmbedding):
        return rotary.for_batch(batch_size)
    return rotary.forward_from_seq_len(seq_len)


def apply_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    rope: RotaryInputs | tuple[torch.Tensor, float | torch.Tensor | None] | None,
    *,
    pe_attn_head: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(rope, RotaryInputs):
        apply_rotary_inplace(query, key, rope)
    elif rope is not None:
        freqs, xpos_scale = rope
        q_xpos_scale, k_xpos_scale = (
            (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
        )

        if pe_attn_head is not None:
            pn = pe_attn_head
            query[:, :pn, :, :] = apply_rotary_pos_emb(
                query[:, :pn, :, :], freqs, q_xpos_scale
            )
            key[:, :pn, :, :] = apply_rotary_pos_emb(
                key[:, :pn, :, :], freqs, k_xpos_scale
            )
        else:
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
    return query, key


def apply_rotary_inplace(
    query: torch.Tensor, key: torch.Tensor, rope: RotaryInputs
) -> None:
    batch_size, heads, seq_len, head_dim = query.shape
    # Note(yzxiao): Undo the attention head view to recover the Linear outputs'
    # token-major layout. view must alias the original Q/K, never copy them.
    query_tokens = query.transpose(1, 2).view(batch_size * seq_len, heads, head_dim)
    key_tokens = key.transpose(1, 2).view(batch_size * seq_len, heads, head_dim)
    rope.kernel(
        query_tokens,
        key_tokens,
        rope.cos_sin_cache,
        rope.positions,
        is_neox=False,
    )
