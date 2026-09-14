# SPDX-License-Identifier: Apache-2.0
"""One launch for CosyVoice3's two partial rotary embeddings."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["seq_len"])
def _partial_qk_rope(
    Q,
    K,
    COS,
    SIN,
    Q_OUT,
    K_OUT,
    seq_len,
    WIDTH: tl.constexpr,
    ROT_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    batch = tl.program_id(1)
    channel = tl.arange(0, BLOCK)
    offset = (batch * seq_len + token) * WIDTH + channel
    rotated = channel < ROT_DIM
    cos = tl.load(COS + token * ROT_DIM + channel, rotated, other=0)
    sin = tl.load(SIN + token * ROT_DIM + channel, rotated, other=0)
    # Interleaved pairs: [-x1, x0, -x3, x2, ...]. The tail is copied verbatim.
    partner = (batch * seq_len + token) * WIDTH + (channel ^ 1)
    sign = tl.where(channel % 2 == 0, -1.0, 1.0)
    q = tl.load(Q + offset, channel < WIDTH, other=0).to(tl.float32)
    k = tl.load(K + offset, channel < WIDTH, other=0).to(tl.float32)
    q_pair = tl.load(Q + partner, rotated, other=0).to(tl.float32)
    k_pair = tl.load(K + partner, rotated, other=0).to(tl.float32)
    q_rot = q * cos + (sign * q_pair) * sin
    k_rot = k * cos + (sign * k_pair) * sin
    tl.store(Q_OUT + offset, tl.where(rotated, q_rot, q), channel < WIDTH)
    tl.store(K_OUT + offset, tl.where(rotated, k_rot, k), channel < WIDTH)


def fused_qk_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Private inference ABI: projections are contiguous [B, T, 1024], and
    # tables are contiguous [1, T, 64] FP32. No input is modified or aliased.
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    with torch.cuda.device(q.device):
        _partial_qk_rope[(q.shape[1], q.shape[0])](
            q,
            k,
            cos,
            sin,
            q_out,
            k_out,
            q.shape[1],
            WIDTH=1024,
            ROT_DIM=64,
            BLOCK=1024,
            num_warps=4,
            # Match the native separate multiply/add rounding.
            enable_fp_fusion=False,
        )
    return q_out, k_out
