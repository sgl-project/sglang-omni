# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for BF16 RNN-T greedy decoding in Nemotron ASR.

Provides register-tiled BF16 joint argmax (``greedy_joint_bf16``), single-kernel
step bookkeeping (``advance_rnnt_triton``), and fused FP32-state LSTM gate
updates with BF16 hidden cache writeback (``lstm_gate_update_triton``).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments without Triton
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def joint_tiles_kernel(
        enc,
        pred,
        times,
        lens,
        weight,
        bias,
        partial,
        ids,
        batch,
        stride_b,
        stride_t,
        stride_c,
        pred_b,
        width: tl.constexpr,
        vocab: tl.constexpr,
        tiles: tl.constexpr,
        steps: tl.constexpr,
        bm: tl.constexpr,
        bn: tl.constexpr,
        bk: tl.constexpr,
    ):
        m = tl.program_id(0) * bm + tl.arange(0, bm)
        tile = tl.program_id(1)
        n = tile * bn + tl.arange(0, bn)
        row = m // steps
        time = tl.load(times + row, row < batch, 0) + m % steps
        last = tl.load(lens + row, row < batch, 1) - 1
        time = tl.maximum(0, tl.minimum(time, last))
        k = tl.arange(0, bk)
        acc = tl.full((bm, bn), 0, tl.float32)
        for i in range(triton.cdiv(width, bk)):
            col = i * bk + k
            f = tl.load(
                enc
                + row[:, None] * stride_b
                + time[:, None] * stride_t
                + col[None, :] * stride_c,
                (row < batch)[:, None] & (col < width)[None, :],
                0,
            )
            g = tl.load(
                pred + row[:, None] * pred_b + col[None, :],
                (row < batch)[:, None] & (col < width)[None, :],
                0,
            )
            h = tl.maximum((f.to(tl.float32) + g.to(tl.float32)).to(tl.bfloat16), 0).to(
                tl.bfloat16
            )
            w = tl.load(
                weight + n[:, None] * width + col[None, :],
                (n < vocab)[:, None] & (col < width)[None, :],
                0,
            )
            acc = tl.dot(h, tl.trans(w), acc)
        b = tl.load(bias + n, n < vocab, 0)
        logits = (acc + b[None, :]).to(tl.bfloat16).to(tl.float32)
        logits = tl.where(n[None, :] < vocab, logits, -float("inf"))
        best = tl.max(logits, 1)
        label = tl.min(tl.where(logits == best[:, None], n[None, :], vocab), 1)
        offset = m * tiles + tile
        tl.store(partial + offset, best, row < batch)
        tl.store(ids + offset, label, row < batch)

    @triton.jit
    def joint_winners_kernel(
        partial, ids, scores, labels, tiles: tl.constexpr, block: tl.constexpr
    ):
        row = tl.program_id(0)
        k = tl.arange(0, block)
        x = tl.load(partial + row * tiles + k, k < tiles, -float("inf")).to(tl.float32)
        indices = tl.load(ids + row * tiles + k, k < tiles, 2147483647)
        top = tl.max(x, 0)
        tl.store(scores + row, top)
        tl.store(labels + row, tl.min(tl.where(x == top, indices, 2147483647), 0))

    @triton.jit
    def advance_rnnt_kernel(
        scan,
        time,
        lengths,
        active,
        counts,
        labels,
        stamps,
        run,
        last_frame,
        pending,
        pending_label,
        batch,
        capacity,
        blank: tl.constexpr,
        max_symbols: tl.constexpr,
        steps: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0) * block + tl.arange(0, block)
        live = row < batch
        k = tl.arange(0, steps)
        t = tl.load(time + row, live, 0)
        n = tl.load(lengths + row, live, 0)
        act = tl.load(active + row, live, 0) != 0
        ids = tl.load(
            scan + row[:, None] * steps + k[None, :], live[:, None], blank
        ).to(tl.int64)
        valid = ((t[:, None] + k[None, :]) < n[:, None]) & act[:, None]
        first = tl.min(tl.where(valid & (ids != blank), k[None, :], steps), axis=1)
        emit = first < steps
        label = tl.sum(tl.where(k[None, :] == first[:, None], ids, 0), axis=1)
        t = tl.where(emit, t + first, t + steps)
        count = tl.load(counts + row, live, 0)
        slot = row.to(tl.int64) * capacity + tl.minimum(count, capacity - 1)
        tl.store(labels + slot, label, live & emit)
        tl.store(stamps + slot, t, live & emit)
        tl.store(counts + row, count + emit.to(tl.int64), live)
        previous = tl.load(last_frame + row, live, 0)
        symbols = tl.load(run + row, live, 0)
        symbols = tl.where(emit, tl.where(t == previous, symbols + 1, 1), symbols)
        forced = emit & (symbols >= max_symbols)
        tl.store(last_frame + row, tl.where(emit, t, previous), live)
        tl.store(run + row, tl.where(forced, 0, symbols), live)
        t = t + forced.to(tl.int64)
        tl.store(time + row, t, live)
        tl.store(active + row, t < n, live)
        prev_pending = tl.load(pending + row, live, 0) != 0
        prev_label = tl.load(pending_label + row, live, blank)
        tl.store(pending + row, prev_pending | emit, live)
        tl.store(pending_label + row, tl.where(emit, label, prev_label), live)

    @triton.jit
    def lstm_gate_update_kernel(
        G_ptr,
        C_ptr,
        H_ptr,
        H_bf16_ptr,
        M_ptr,
        H: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        b = tl.program_id(0)
        off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = off < H
        apply = tl.load(M_ptr + b)
        if apply != 0:
            c_old = tl.load(C_ptr + b * H + off, mask=mask, other=0.0).to(tl.float32)
            base = b * (4 * H)
            i_g = tl.sigmoid(
                tl.load(G_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
            )
            f_g = tl.sigmoid(
                tl.load(G_ptr + base + H + off, mask=mask, other=0.0).to(tl.float32)
            )
            g_g = tl.extra.cuda.libdevice.tanh(
                tl.load(G_ptr + base + 2 * H + off, mask=mask, other=0.0).to(tl.float32)
            )
            o_g = tl.sigmoid(
                tl.load(G_ptr + base + 3 * H + off, mask=mask, other=0.0).to(tl.float32)
            )
            c_new = f_g * c_old + i_g * g_g
            h_new = o_g * tl.extra.cuda.libdevice.tanh(c_new)
            tl.store(C_ptr + b * H + off, c_new, mask=mask)
            tl.store(H_ptr + b * H + off, h_new, mask=mask)
            tl.store(H_bf16_ptr + b * H + off, h_new.to(tl.bfloat16), mask=mask)
        else:
            pass

else:
    pass


def lstm_gate_update_triton(
    gates: torch.Tensor,
    cell: torch.Tensor,
    hidden: torch.Tensor,
    hidden_bf16: torch.Tensor,
    apply: torch.Tensor,
) -> None:
    """Apply fused LSTM gates in FP32 and conditionally update FP32 state + BF16 cache."""
    batch, width = cell.shape
    block = 128
    lstm_gate_update_kernel[(batch, triton.cdiv(width, block))](
        gates,
        cell,
        hidden,
        hidden_bf16,
        apply,
        width,
        block,
    )


def greedy_joint_bf16(
    enc: torch.Tensor,
    pred: torch.Tensor,
    times: torch.Tensor,
    lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """BF16 register-tiled RNN-T greedy joint returning argmax labels ``(B, steps)``."""
    batch, _, width = enc.shape
    vocab = weight.shape[0]
    tiles = triton.cdiv(vocab, 64)
    partial = enc.new_empty((batch * steps, tiles))
    ids = torch.empty_like(partial, dtype=torch.int32)
    scores = enc.new_empty((batch, steps))
    labels = torch.empty_like(scores, dtype=torch.int32)
    joint_tiles_kernel[(triton.cdiv(batch * steps, 16), tiles)](
        enc,
        pred,
        times,
        lengths,
        weight,
        bias,
        partial,
        ids,
        batch,
        *enc.stride(),
        pred.stride(0),
        width,
        vocab,
        tiles,
        steps,
        16,
        64,
        64,
        num_warps=4,
    )
    joint_winners_kernel[(batch * steps,)](
        partial, ids, scores, labels, tiles, triton.next_power_of_2(tiles)
    )
    return labels


def advance_rnnt_triton(
    scan: torch.Tensor,
    time: torch.Tensor,
    lengths: torch.Tensor,
    active: torch.Tensor,
    counts: torch.Tensor,
    labels: torch.Tensor,
    stamps: torch.Tensor,
    run: torch.Tensor,
    last_frame: torch.Tensor,
    pending: torch.Tensor,
    pending_label: torch.Tensor,
    *,
    capacity: int,
    blank: int,
    max_symbols: int,
    steps: int,
) -> None:
    """One-kernel RNN-T greedy step bookkeeping."""
    batch = time.shape[0]
    advance_rnnt_kernel[(triton.cdiv(batch, 32),)](
        scan,
        time,
        lengths,
        active,
        counts,
        labels,
        stamps,
        run,
        last_frame,
        pending,
        pending_label,
        batch,
        capacity,
        blank,
        max_symbols,
        steps,
        32,
    )
