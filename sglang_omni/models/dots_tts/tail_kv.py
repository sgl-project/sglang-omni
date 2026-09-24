# SPDX-License-Identifier: Apache-2.0
"""KV copies for acoustic-tail padding without a persistent dummy KV row."""

import torch
import triton
import triton.language as tl


@triton.jit
def gather_kv_kernel(
    K,
    V,
    Slots,
    OutK,
    OutV,
    N: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    DST_STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    head = row % H
    batch = row // H % B
    layer = row // (H * B)
    slot = tl.load(Slots + batch)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, feature = offset // D, offset % D
    source = layer * SRC_STRIDES[0] + slot * SRC_STRIDES[1] + head * SRC_STRIDES[2]
    source += token * SRC_STRIDES[3] + feature * SRC_STRIDES[4]
    target = layer * DST_STRIDES[0] + batch * DST_STRIDES[1] + head * DST_STRIDES[2]
    target += token * DST_STRIDES[3] + feature * DST_STRIDES[4]
    # note (0xtoward): Dummy IDs are out of range; mask the access itself.
    valid = (offset < T * D) & (slot >= 0) & (slot < N)
    key = tl.load(K + source, mask=valid, other=0)
    value = tl.load(V + source, mask=valid, other=0)
    tl.store(OutK + target, key, mask=offset < T * D)
    tl.store(OutV + target, value, mask=offset < T * D)


@triton.jit
def scatter_kv_kernel(
    K,
    V,
    Slots,
    Starts,
    OutK,
    OutV,
    N: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    POOL_T: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    DST_STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    head = row % H
    batch = row // H % B
    layer = row // (H * B)
    slot = tl.load(Slots + batch)
    start = tl.load(Starts + batch)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, feature = offset // D, offset % D
    source = layer * SRC_STRIDES[0] + batch * SRC_STRIDES[1] + head * SRC_STRIDES[2]
    source += token * SRC_STRIDES[3] + feature * SRC_STRIDES[4]
    target = layer * DST_STRIDES[0] + slot * DST_STRIDES[1] + head * DST_STRIDES[2]
    target += (start + token) * DST_STRIDES[3] + feature * DST_STRIDES[4]
    valid = (offset < T * D) & (slot >= 0) & (slot < N)
    valid &= (start + token >= 0) & (start + token < POOL_T)
    key = tl.load(K + source, mask=valid, other=0)
    value = tl.load(V + source, mask=valid, other=0)
    # note (0xtoward): Real slots are unique; repeated dummy slots never store.
    tl.store(OutK + target, key, mask=valid)
    tl.store(OutV + target, value, mask=valid)


def gather_kv(
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    slots: torch.Tensor,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
) -> None:
    """Gather [L,N,H,T,D] to [L,B,H,T,D], zeroing invalid slot IDs.

    Capacity views may be strided. K and V in each pair share shape/strides.
    """
    layers, rows, heads, tokens, dim = out_k.shape
    assert pool_k.shape == pool_v.shape and pool_k.stride() == pool_v.stride()
    assert out_k.shape == out_v.shape and out_k.stride() == out_v.stride()
    assert slots.shape == (rows,) and slots.is_contiguous()
    assert (layers, heads, dim) == (pool_k.size(0), pool_k.size(2), pool_k.size(4))
    assert tokens <= pool_k.size(3)
    if tokens == 0:
        return
    block = 1024
    with torch.cuda.device_of(pool_k):
        gather_kv_kernel[(triton.cdiv(tokens * dim, block), layers * rows * heads)](
            pool_k,
            pool_v,
            slots,
            out_k,
            out_v,
            pool_k.size(1),
            rows,
            heads,
            tokens,
            dim,
            pool_k.stride(),
            out_k.stride(),
            block,
        )


def scatter_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    slots: torch.Tensor,
    starts: torch.Tensor,
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
) -> None:
    """Promote [L,B,H,T,D] at per-row starts; skip invalid slot IDs.

    Live slot IDs must be unique and their promoted tokens must fit the pool.
    """
    layers, rows, heads, tokens, dim = keys.shape
    assert keys.shape == values.shape and keys.stride() == values.stride()
    assert pool_k.shape == pool_v.shape and pool_k.stride() == pool_v.stride()
    assert slots.shape == starts.shape == (rows,)
    assert slots.is_contiguous() and starts.is_contiguous()
    assert (layers, heads, dim) == (pool_k.size(0), pool_k.size(2), pool_k.size(4))
    if tokens == 0:
        return
    block = min(1024, triton.next_power_of_2(tokens * dim))
    with torch.cuda.device_of(pool_k):
        scatter_kv_kernel[(triton.cdiv(tokens * dim, block), layers * rows * heads)](
            keys,
            values,
            slots,
            starts,
            pool_k,
            pool_v,
            pool_k.size(1),
            rows,
            heads,
            tokens,
            dim,
            pool_k.size(3),
            keys.stride(),
            pool_k.stride(),
            block,
        )
