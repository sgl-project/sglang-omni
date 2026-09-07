# SPDX-License-Identifier: Apache-2.0
"""Component-owned rotary inputs for native and cached joint Q/K execution."""

from dataclasses import dataclass
from typing import Literal

import torch
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb


@dataclass(frozen=True)
class RotaryInputs:
    cos_sin_cache: torch.Tensor
    positions: torch.Tensor


def validate_rotary_config(
    backend: str,
    *,
    num_heads: int,
    qk_norm: str | None,
    pe_attn_head: int | None,
    grad_checkpointing: bool = False,
) -> None:
    """Check external component options once, before constructing its layers."""
    if backend == "native":
        return
    if (
        backend != "sglang"
        or qk_norm is not None
        or pe_attn_head not in (None, num_heads)
        or grad_checkpointing
    ):
        raise ValueError(
            "Joint RoPE requires backend='sglang', full-head rotation, "
            "and no Q/K norm or gradient checkpointing; received "
            f"backend={backend!r}, num_heads={num_heads}, qk_norm={qk_norm!r}, "
            f"pe_attn_head={pe_attn_head!r}, grad_checkpointing={grad_checkpointing}"
        )


class CachedRotaryEmbedding(RotaryEmbedding):
    """Fixed FP32 coefficients and bounded positions for CUDA inference."""

    def __init__(self, dim: int, *, seq_len: int, max_batch_size: int) -> None:
        with torch.autocast(device_type="cuda", enabled=False):
            super().__init__(dim)
            freqs, _ = self.forward_from_seq_len(seq_len)
            freqs = freqs.reshape(seq_len, dim)
            phase = freqs[:, 0::2]
            cache = torch.cat((phase.cos(), phase.sin()), dim=-1).contiguous()

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
        if num_tokens <= self.positions.numel():
            positions = self.positions[:num_tokens]
        else:
            # Note(yzxiao): Large references borrow a temporary position vector;
            # replacing the resident buffer would invalidate captured tail graphs.
            positions = torch.arange(
                seq_len, device=self.positions.device, dtype=self.positions.dtype
            ).repeat(batch_size)
        return RotaryInputs(self.cos_sin_cache, positions)


def build_rotary_embedding(
    dim: int,
    *,
    backend: Literal["native", "sglang"] = "native",
    seq_len: int | None = None,
    max_batch_size: int | None = None,
) -> RotaryEmbedding:
    if backend == "native":
        return RotaryEmbedding(dim)
    return CachedRotaryEmbedding(dim, seq_len=seq_len, max_batch_size=max_batch_size)


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
    """Apply the selected rotary operation and return its Q/K tensors."""
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
    from sglang.kernels.ops.attention.rope import apply_rope_inplace

    batch_size, heads, seq_len, head_dim = query.shape
    # Note(yzxiao): Undo the attention head view to recover the Linear outputs'
    # token-major layout. view must alias the original Q/K, never copy them.
    query_tokens = query.transpose(1, 2).view(batch_size * seq_len, heads, head_dim)
    key_tokens = key.transpose(1, 2).view(batch_size * seq_len, heads, head_dim)
    apply_rope_inplace(
        query_tokens,
        key_tokens,
        rope.cos_sin_cache,
        rope.positions,
        is_neox=False,
    )
