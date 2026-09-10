# SPDX-License-Identifier: Apache-2.0
"""Shared inference kernels for the MOSS-Audio-Tokenizer vocoder."""

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
_STREAMING_KV_BLOCK_SIZE = 512


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
    with torch.cuda.device(q.device):
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


if triton is not None:

    @triton.jit
    def _streaming_kv_gather_kernel(
        cached_k,
        cached_v,
        cached_positions,
        current_k,
        current_v,
        query_positions,
        slots,
        all_k,
        all_v,
        key_positions,
        cached_k_strides: tl.constexpr,
        cached_v_strides: tl.constexpr,
        current_k_strides: tl.constexpr,
        current_v_strides: tl.constexpr,
        num_heads: tl.constexpr,
        context: tl.constexpr,
        chunk_length: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        index = tl.program_id(1) * block_size + tl.arange(0, block_size)
        length: tl.constexpr = context + chunk_length
        mask = index < num_heads * length * head_dim
        head = index // (length * head_dim)
        position = (index // head_dim) % length
        dim = index % head_dim
        old = position < context
        slot = tl.load(slots + row)
        old_k = tl.load(
            cached_k
            + slot * cached_k_strides[0]
            + head * cached_k_strides[1]
            + position * cached_k_strides[2]
            + dim * cached_k_strides[3],
            mask & old,
            other=0,
        )
        old_v = tl.load(
            cached_v
            + slot * cached_v_strides[0]
            + head * cached_v_strides[1]
            + position * cached_v_strides[2]
            + dim * cached_v_strides[3],
            mask & old,
            other=0,
        )
        new_k = tl.load(
            current_k
            + row * current_k_strides[0]
            + head * current_k_strides[1]
            + (position - context) * current_k_strides[2]
            + dim * current_k_strides[3],
            mask & ~old,
            other=0,
        )
        new_v = tl.load(
            current_v
            + row * current_v_strides[0]
            + head * current_v_strides[1]
            + (position - context) * current_v_strides[2]
            + dim * current_v_strides[3],
            mask & ~old,
            other=0,
        )
        output_index = row * num_heads * length * head_dim + index
        tl.store(all_k + output_index, tl.where(old, old_k, new_k), mask)
        tl.store(all_v + output_index, tl.where(old, old_v, new_v), mask)

        position_mask = mask & (head == 0) & (dim == 0)
        old_position = tl.load(
            cached_positions + slot * context + position,
            position_mask & old,
            other=-1,
        )
        new_position = tl.load(
            query_positions + row * chunk_length + position - context,
            position_mask & ~old,
            other=0,
        )
        tl.store(
            key_positions + row * length + position,
            tl.where(old, old_position, new_position),
            position_mask,
        )

    @triton.jit
    def _streaming_kv_commit_kernel(
        cached_k,
        cached_v,
        cached_positions,
        offsets,
        all_k,
        all_v,
        key_positions,
        slots,
        valid_rows,
        cached_k_strides: tl.constexpr,
        cached_v_strides: tl.constexpr,
        num_heads: tl.constexpr,
        context: tl.constexpr,
        chunk_length: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        index = tl.program_id(1) * block_size + tl.arange(0, block_size)
        valid = tl.load(valid_rows + row)
        slot = tl.load(slots + row)
        mask = (index < num_heads * context * head_dim) & valid
        head = index // (context * head_dim)
        position = (index // head_dim) % context
        dim = index % head_dim
        # note (Zhang Yiyang): Read the separate gather output to avoid races
        # from shifting persistent rows in place. The suffix starts at T,
        # including T >= C.
        source = (
            (row * num_heads + head) * (context + chunk_length)
            + position
            + chunk_length
        ) * head_dim + dim
        k = tl.load(all_k + source, mask, other=0)
        v = tl.load(all_v + source, mask, other=0)
        tl.store(
            cached_k
            + slot * cached_k_strides[0]
            + head * cached_k_strides[1]
            + position * cached_k_strides[2]
            + dim * cached_k_strides[3],
            k,
            mask,
        )
        tl.store(
            cached_v
            + slot * cached_v_strides[0]
            + head * cached_v_strides[1]
            + position * cached_v_strides[2]
            + dim * cached_v_strides[3],
            v,
            mask,
        )
        position_mask = mask & (head == 0) & (dim == 0)
        next_position = tl.load(
            key_positions + row * (context + chunk_length) + position + chunk_length,
            position_mask,
            other=-1,
        )
        tl.store(
            cached_positions + slot * context + position, next_position, position_mask
        )
        if tl.program_id(1) == 0:
            offset = tl.load(offsets + slot, valid, other=0)
            tl.store(offsets + slot, offset + chunk_length, valid)

else:
    _streaming_kv_gather_kernel = None
    _streaming_kv_commit_kernel = None


def can_fuse_streaming_kv(
    cached_k: torch.Tensor,
    cached_v: torch.Tensor,
    cached_positions: torch.Tensor,
    offsets: torch.Tensor,
    slots: torch.Tensor,
    valid_rows: torch.Tensor,
) -> bool:
    """Check the inference/layout boundary, independently of execution B and T."""
    if (
        _streaming_kv_gather_kernel is None
        or torch.version.hip is not None
        or torch.is_grad_enabled()
        or cached_k.device.type != "cuda"
        or cached_k.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or cached_v.dtype != cached_k.dtype
        or cached_k.ndim != 4
        or cached_v.shape != cached_k.shape
        or any(size <= 0 for size in cached_k.shape)
        or any(stride <= 0 for stride in (*cached_k.stride(), *cached_v.stride()))
        or slots.ndim != 1
        or slots.numel() == 0
    ):
        return False
    capacity, _, context, _ = cached_k.shape
    return (
        all(
            t.device == cached_k.device
            for t in (cached_v, cached_positions, offsets, slots, valid_rows)
        )
        and cached_positions.dtype == offsets.dtype == slots.dtype == torch.long
        and valid_rows.dtype == torch.bool
        and cached_positions.shape == (capacity, context)
        and offsets.shape == (capacity,)
        and valid_rows.shape == slots.shape
        and cached_positions.is_contiguous()
        and offsets.stride() == slots.stride() == valid_rows.stride() == (1,)
    )


def gather_streaming_kv(
    cached_k: torch.Tensor,
    cached_v: torch.Tensor,
    cached_positions: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    query_positions: torch.Tensor,
    slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather and append chronological K/V for eligible indexed inference.

    The caller checks ``can_fuse_streaming_kv`` first. Current K/V must match
    the cache dtype/head shape; query_positions is contiguous [B, T]. K/V may
    be strided. The returned tensors own separate contiguous storage.
    """
    batch_size, num_heads, chunk_length, head_dim = current_k.shape
    context = cached_k.shape[2]
    shape = (batch_size, num_heads, context + chunk_length, head_dim)
    all_k = current_k.new_empty(shape)
    all_v = torch.empty_like(all_k)
    key_positions = query_positions.new_empty((batch_size, context + chunk_length))
    block_size = _STREAMING_KV_BLOCK_SIZE
    with torch.cuda.device(current_k.device):
        _streaming_kv_gather_kernel[
            (
                batch_size,
                triton.cdiv(
                    num_heads * (context + chunk_length) * head_dim, block_size
                ),
            )
        ](
            cached_k,
            cached_v,
            cached_positions,
            current_k,
            current_v,
            query_positions,
            slots,
            all_k,
            all_v,
            key_positions,
            cached_k.stride(),
            cached_v.stride(),
            current_k.stride(),
            current_v.stride(),
            num_heads,
            context,
            chunk_length,
            head_dim,
            block_size,
            num_warps=4,
        )
    return all_k, all_v, key_positions


def commit_streaming_kv_(
    cached_k: torch.Tensor,
    cached_v: torch.Tensor,
    cached_positions: torch.Tensor,
    offsets: torch.Tensor,
    all_k: torch.Tensor,
    all_v: torch.Tensor,
    key_positions: torch.Tensor,
    slots: torch.Tensor,
    valid_rows: torch.Tensor,
) -> None:
    """Commit a gather result to unique valid slots, preserving inactive rows.

    Inputs follow ``can_fuse_streaming_kv`` and ``gather_streaming_kv``. Run
    gather and commit in order on the same stream; the gather outputs must not
    alias the persistent cache. Invalid rows never access persistent offsets.
    """
    batch_size, num_heads, length, head_dim = all_k.shape
    context = cached_k.shape[2]
    chunk_length = length - context
    block_size = _STREAMING_KV_BLOCK_SIZE
    with torch.cuda.device(all_k.device):
        _streaming_kv_commit_kernel[
            (batch_size, triton.cdiv(num_heads * context * head_dim, block_size))
        ](
            cached_k,
            cached_v,
            cached_positions,
            offsets,
            all_k,
            all_v,
            key_positions,
            slots,
            valid_rows,
            cached_k.stride(),
            cached_v.stride(),
            num_heads,
            context,
            chunk_length,
            head_dim,
            block_size,
            num_warps=4,
        )


__all__ = [
    "apply_exact_interleaved_rope_inplace",
    "can_fuse_streaming_kv",
    "commit_streaming_kv_",
    "gather_streaming_kv",
]
