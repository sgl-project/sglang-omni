# SPDX-License-Identifier: Apache-2.0
"""One launch for CosyVoice3's two partial rotary embeddings."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_PROJECTED_DIM = 1024
_ROTARY_DIM = 64
_NUM_WARPS = 4


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
    # note (wirybeaver): CosyVoice rotates interleaved pairs before splitting heads.
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
    """Apply partial interleaved RoPE to contiguous Q and K projections."""
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
            WIDTH=_PROJECTED_DIM,
            ROT_DIM=_ROTARY_DIM,
            BLOCK=_PROJECTED_DIM,
            num_warps=_NUM_WARPS,
            # note (wirybeaver): Native rotary uses separate multiply and add rounding.
            enable_fp_fusion=False,
        )
    return q_out, k_out


__all__ = ["fused_qk_rope"]
