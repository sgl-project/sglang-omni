# SPDX-License-Identifier: Apache-2.0
"""Shared inference kernels for the MOSS audio-tokenizer vocoder."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None


_EXACT_ROPE_BLOCK_SIZE = 256
_EXACT_ROPE_NUM_WARPS = 4


if triton is not None and hasattr(tl, "inline_asm_elementwise"):

    @triton.jit
    def _exact_interleaved_rope_kernel(
        q,
        k,
        cos_sin,
        positions,
        total_pairs,
        stride_qt,
        stride_qh,
        stride_kt,
        stride_kh,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        half_dim: tl.constexpr = head_dim // 2
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total_pairs
        token = offsets // (num_heads * half_dim)
        pair_in_token = offsets % (num_heads * half_dim)
        head = pair_in_token // half_dim
        pair = pair_in_token % half_dim
        q_base = q + token * stride_qt + head * stride_qh + pair * 2
        k_base = k + token * stride_kt + head * stride_kh + pair * 2
        position = tl.load(positions + token, mask=mask, other=0)
        cos = tl.load(
            cos_sin + position * head_dim + pair,
            mask=mask,
        ).to(tl.float32)
        sin = tl.load(
            cos_sin + position * head_dim + half_dim + pair,
            mask=mask,
        ).to(tl.float32)
        qr = tl.load(q_base, mask=mask).to(tl.float32)
        qi = tl.load(q_base + 1, mask=mask).to(tl.float32)
        kr = tl.load(k_base, mask=mask).to(tl.float32)
        ki = tl.load(k_base + 1, mask=mask).to(tl.float32)

        # Note (Zhang Yiyang): The source implementation materializes each FP32
        # multiply before the add/subtract. Explicit rounding prevents contraction
        # into FMAs and keeps the fused kernel bitwise-equivalent after the
        # low-precision store.
        qor, qoi, kor, koi = tl.inline_asm_elementwise(
            asm="""
            {
                .reg .f32 a;
                .reg .f32 b;
                mul.rn.f32 a, $4, $8;
                mul.rn.f32 b, $5, $9;
                sub.rn.f32 $0, a, b;
                mul.rn.f32 a, $4, $9;
                mul.rn.f32 b, $5, $8;
                add.rn.f32 $1, a, b;
                mul.rn.f32 a, $6, $8;
                mul.rn.f32 b, $7, $9;
                sub.rn.f32 $2, a, b;
                mul.rn.f32 a, $6, $9;
                mul.rn.f32 b, $7, $8;
                add.rn.f32 $3, a, b;
            }
            """,
            constraints="=f,=f,=f,=f,f,f,f,f,f,f",
            args=[qr, qi, kr, ki, cos, sin],
            dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
            is_pure=True,
            pack=1,
        )
        tl.store(q_base, qor, mask=mask)
        tl.store(q_base + 1, qoi, mask=mask)
        tl.store(k_base, kor, mask=mask)
        tl.store(k_base + 1, koi, mask=mask)

else:
    _exact_interleaved_rope_kernel = None


def apply_exact_interleaved_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
) -> bool:
    """Apply source-equivalent interleaved RoPE in one CUDA kernel if supported."""

    if (
        _exact_interleaved_rope_kernel is None
        or torch.version.hip is not None
        or q.device.type != "cuda"
        or k.device != q.device
        or cos_sin_cache.device != q.device
        or position_ids.device != q.device
        or q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or cos_sin_cache.dtype != torch.float32
        or position_ids.dtype not in (torch.int32, torch.int64)
        or q.ndim != 3
        or k.shape != q.shape
        or cos_sin_cache.ndim != 2
        or int(cos_sin_cache.shape[0]) == 0
        or int(cos_sin_cache.shape[1]) != int(q.shape[2])
        or position_ids.ndim != 1
        or int(position_ids.numel()) != int(q.shape[0])
        or q.stride(2) != 1
        or k.stride(2) != 1
        or position_ids.stride(0) != 1
        or not cos_sin_cache.is_contiguous()
    ):
        return False

    tokens, num_heads, head_dim = map(int, q.shape)
    if tokens == 0 or num_heads <= 0 or head_dim <= 0 or head_dim % 2 != 0:
        return False
    total_pairs = tokens * num_heads * (head_dim // 2)
    block_size = _EXACT_ROPE_BLOCK_SIZE
    _exact_interleaved_rope_kernel[(triton.cdiv(total_pairs, block_size),)](
        q,
        k,
        cos_sin_cache,
        position_ids,
        total_pairs,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_warps=_EXACT_ROPE_NUM_WARPS,
    )
    return True


__all__ = ["apply_exact_interleaved_rope_inplace"]
