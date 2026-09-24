# SPDX-License-Identifier: Apache-2.0
"""AuK's per-head RMSNorm and interleaved rotary embedding in one launch."""

import torch
import triton
import triton.language as tl
from torch import nn

from sglang_omni.models.auk.dit import Rope


@triton.jit(
    do_not_specialize=("QB", "KB", "CB", "SEQ"),
    do_not_specialize_on_alignment=("QB", "KB", "CB", "SEQ"),
)
def norm_rope_kernel(
    Q,
    K,
    Q_WEIGHT,
    K_WEIGHT,
    COS,
    SIN,
    Q_OUT,
    K_OUT,
    QB,
    QH: tl.constexpr,
    QS: tl.constexpr,
    QD: tl.constexpr,
    KB,
    KH: tl.constexpr,
    KS: tl.constexpr,
    KD: tl.constexpr,
    CB,
    CS: tl.constexpr,
    CD: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQ,
    EPS: tl.constexpr,
    ROUND_NORM: tl.constexpr,
):
    position = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    row = (batch * HEADS + head) * SEQ + position
    q_start = batch * QB + head * QH + position * QS
    k_start = batch * KB + head * KH + position * KS
    lane = tl.arange(0, HEAD_DIM // 4)

    # Match CUDA RMSNorm's four consecutive values per lane before its
    # shuffle reduction. A flat tl.sum changes rounding and the DiT trajectory.
    qa = tl.load(Q + q_start + lane * 4 * QD).to(tl.float32)
    qb = tl.load(Q + q_start + (lane * 4 + 1) * QD).to(tl.float32)
    qc = tl.load(Q + q_start + (lane * 4 + 2) * QD).to(tl.float32)
    qd = tl.load(Q + q_start + (lane * 4 + 3) * QD).to(tl.float32)
    ka = tl.load(K + k_start + lane * 4 * KD).to(tl.float32)
    kb = tl.load(K + k_start + (lane * 4 + 1) * KD).to(tl.float32)
    kc = tl.load(K + k_start + (lane * 4 + 2) * KD).to(tl.float32)
    kd = tl.load(K + k_start + (lane * 4 + 3) * KD).to(tl.float32)
    q_sum = ((qa * qa + qb * qb) + qc * qc) + qd * qd
    k_sum = ((ka * ka + kb * kb) + kc * kc) + kd * kd
    q_scale = tl.rsqrt(tl.sum(q_sum, 0) / HEAD_DIM + EPS)
    k_scale = tl.rsqrt(tl.sum(k_sum, 0) / HEAD_DIM + EPS)

    dim = tl.arange(0, HEAD_DIM)
    partner = dim ^ 1
    sign = tl.where(dim % 2 == 0, -1.0, 1.0)
    q = tl.load(Q + q_start + dim * QD).to(tl.float32)
    k = tl.load(K + k_start + dim * KD).to(tl.float32)
    q_pair = tl.load(Q + q_start + partner * QD).to(tl.float32)
    k_pair = tl.load(K + k_start + partner * KD).to(tl.float32)
    q = q * q_scale * tl.load(Q_WEIGHT + dim).to(tl.float32)
    k = k * k_scale * tl.load(K_WEIGHT + dim).to(tl.float32)
    q_pair = q_pair * q_scale * tl.load(Q_WEIGHT + partner).to(tl.float32) * sign
    k_pair = k_pair * k_scale * tl.load(K_WEIGHT + partner).to(tl.float32) * sign
    if ROUND_NORM:
        q = q.to(Q_OUT.dtype.element_ty).to(tl.float32)
        k = k.to(K_OUT.dtype.element_ty).to(tl.float32)
        q_pair = q_pair.to(Q_OUT.dtype.element_ty).to(tl.float32)
        k_pair = k_pair.to(K_OUT.dtype.element_ty).to(tl.float32)
    else:
        pass
    table_offset = batch * CB + position * CS + dim * CD
    cosine = tl.load(COS + table_offset).to(tl.float32)
    sine = tl.load(SIN + table_offset).to(tl.float32)
    tl.store(Q_OUT + row * HEAD_DIM + dim, q * cosine + q_pair * sine)
    tl.store(K_OUT + row * HEAD_DIM + dim, k * cosine + k_pair * sine)


def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.RMSNorm,
    k_norm: nn.RMSNorm,
    rope: Rope,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Norm and rotate Q and K in one launch, from the caller's trig tables.

    The tables come in on the rope tuple rather than from a cache of this
    module's own: the blocks that call this are compiled, and a lookup keyed on
    the freqs pointer would make every trajectory a new guard to recompile for.
    """
    if rope.scale != 1.0:
        raise ValueError("AuK Q/K fusion requires XPos disabled")
    else:
        pass
    cosine, sine = rope.cos, rope.sin
    if cosine.ndim == 2:
        strides = (0, *cosine.stride())
    else:
        strides = (
            0 if cosine.shape[0] == 1 else cosine.stride(0),
            cosine.stride(1),
            cosine.stride(2),
        )
    output_dtype = q_norm.weight.dtype
    q_out = torch.empty(q.shape, device=q.device, dtype=output_dtype)
    k_out = torch.empty_like(q_out)
    epsilon = torch.finfo(output_dtype).eps if q_norm.eps is None else q_norm.eps
    # note(Dayuxiaoshui): runtime sequence and outer strides, so one kernel is
    # shared across request lengths.
    norm_rope_kernel[(q.shape[2], q.shape[1], q.shape[0])](
        q,
        k,
        q_norm.weight,
        k_norm.weight,
        cosine,
        sine,
        q_out,
        k_out,
        *q.stride(),
        *k.stride(),
        *strides,
        HEADS=q.shape[1],
        HEAD_DIM=q.shape[3],
        SEQ=q.shape[2],
        EPS=epsilon,
        ROUND_NORM=output_dtype != torch.float32,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return q_out, k_out
