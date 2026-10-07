# SPDX-License-Identifier: Apache-2.0
"""The fused layers of the Qwen3 talker's code predictor, shared by Qwen3-Omni and Qwen3-TTS.

The exact add-norm, for the plain path: the residual add and the RMSNorm in one launch,
bit for bit torch's bf16 add followed by the default RMSNorm kernel (the sum rounded to
bf16, the kernel's reduction reproduced: one warp per row, lane t holding columns 8t + 256b + j, a
sequential per-lane fold in (b, j) order, a butterfly with offsets 1, 2, 4, 8, 16, an
approximate rsqrt, x * rstd * w rounded once); its equality test guards it.

The fused layer, for plain bf16 weights on one rank on CUDA: three kernels, launched
four times per decoder layer around sdpa. attention_inputs norms the rows, projects them
to qkv, norms and rotates q and k, writes q to a buffer and k and v into the rows' cache
slots. o_proj_add and down_add multiply by the projection and add the residual. mlp_up
norms the residual, projects it to gate and up and applies silu(gate) * up. Every bf16
rounding sits where Qwen3-Omni's plain path rounds. Qwen3-TTS's plain path rounds later
at two boundaries: its o_proj adds the residual before its one rounding, and its add-norm
after down normalizes the fp32 sum of the residual and the projection while storing the
sum in bf16, where the fused path normalizes the stored bf16 residual. Beyond those
roundings, what differs is the GEMMs' fp32 accumulation order (K blocks on the tensor
cores, fp32 split partials summed in index order, so deterministic), the norm's row scale
applied to the accumulated product instead of to the input, and libdevice's exp in the
activation. Rows never mix, so a row's result does not depend on the batch.

One program covers a launch's rows, so the launch reads each weight once, as the GEMM it
replaces does. FusedPredictorLayers owns the launches' buffers, runs the layer loop, and
covers passes of up to MAX_FUSED_ROWS rows; larger passes keep the plain path.

The module imports where Triton is unavailable; the resolvers then keep the plain path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.runtime_context import get_exec

from sglang_omni.platforms import current_platform

try:  # keep the module importable where Triton is unavailable
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    libdevice = None
    gdc_launch_dependents = None
    gdc_wait = None
    HAS_TRITON = False

HIDDEN_SIZE = 1024
LANES = 32
VEC = 8
VEC_BLOCKS = HIDDEN_SIZE // (LANES * VEC)
# note (ratish): the accumulation group of every projection but gate_up; it must divide
# the head so an o_proj K block never crosses a head of the attention output.
BLOCK_K = 128
# note (ratish): one MMA n-tile; the head launch uses the head as its tile.
BLOCK_N = 32
# note (ratish): the gate_up launch has no split, so its wider K block keeps the
# per-block cost of the norm scale small.
MLP_BLOCK_K = 256
MIN_BLOCK_M = 16
# note (ratish): above 64 rows the qkv launch's fp32 accumulators, the rows' square and
# their head, exceed 255 registers per thread; a second program rereads the weights.
MAX_FUSED_ROWS = 64
NUM_WARPS = 4
NUM_STAGES = 3


if HAS_TRITON:

    @triton.jit
    def butterfly_sum(lane_partials):
        """Sum 32 lane partials pairing lane i with i ^ 1, then i ^ 2, up to i ^ 16."""
        partials = tl.sum(tl.reshape(lane_partials, [16, 2]), 1)
        partials = tl.sum(tl.reshape(partials, [8, 2]), 1)
        partials = tl.sum(tl.reshape(partials, [4, 2]), 1)
        partials = tl.sum(tl.reshape(partials, [2, 2]), 1)
        return tl.sum(partials, 0)

    @triton.jit
    def add_rmsnorm_rounded_kernel(
        X,
        RESIDUAL,
        WEIGHT,
        eps,
        HIDDEN: tl.constexpr,
        BLOCKS: tl.constexpr,
        LANE_VEC: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = tl.arange(0, 32)
        j = tl.arange(0, LANE_VEC)
        row_base = row * HIDDEN
        acc = tl.zeros([32], dtype=tl.float32)
        for block in tl.static_range(BLOCKS):
            # One 16-byte vector per lane; the lane's eight squares are then added in
            # element order, each through a masked sum, which is exact.
            col = LANE_VEC * lane[:, None] + 32 * LANE_VEC * block + j[None, :]
            total = tl.load(X + row_base + col).to(tl.float32) + tl.load(
                RESIDUAL + row_base + col
            ).to(tl.float32)
            rounded = total.to(tl.bfloat16)
            tl.store(RESIDUAL + row_base + col, rounded)
            value = rounded.to(tl.float32)
            squares = value * value
            for element in tl.static_range(LANE_VEC):
                acc += tl.sum(tl.where(j[None, :] == element, squares, 0.0), 1)
        sum_sq = butterfly_sum(acc)
        rstd = libdevice.rsqrt(sum_sq / HIDDEN + eps)
        block = tl.arange(0, BLOCKS)
        j = tl.arange(0, LANE_VEC)
        col = (
            LANE_VEC * lane[:, None, None]
            + 32 * LANE_VEC * block[None, :, None]
            + j[None, None, :]
        )
        value = tl.load(RESIDUAL + row_base + col).to(tl.float32)
        weight = tl.load(WEIGHT + col).to(tl.float32)
        tl.store(X + row_base + col, (value * rstd * (weight + 0.0)).to(tl.bfloat16))

    @triton.jit
    def row_offsets(stride_row, stride_t, T: tl.constexpr, BLOCK_M: tl.constexpr):
        """Offsets of the rows laid out (row group, token) major."""
        rm = tl.arange(0, BLOCK_M)
        return rm // T * stride_row + rm % T * stride_t

    @triton.jit
    def diagonal(square):
        """The diagonal of a [BLOCK_M, BLOCK_M] tile as a [BLOCK_M] vector."""
        rm = tl.arange(0, square.shape[0])
        return tl.sum(tl.where(rm[:, None] == rm[None, :], square, 0.0), 1)

    @triton.jit
    def scaled_tile(x, NORM_W, k0, BLOCK_K: tl.constexpr):
        """The input tile times the norm weight, rounded to bf16 for the MMA."""
        rk = tl.arange(0, BLOCK_K)
        w = tl.load(NORM_W + k0 + rk).to(tl.float32)
        return (x.to(tl.float32) * w[None, :]).to(tl.bfloat16)

    @triton.jit
    def weight_tile(W, cols, k0, K: tl.constexpr, BLOCK_K: tl.constexpr):
        rk = tl.arange(0, BLOCK_K)
        return tl.load(W + cols[:, None] * K + (k0 + rk)[None, :])

    @triton.jit
    def norm_qkv_rope_store_kernel(
        X,
        x_stride_row,
        x_stride_t,
        rows,
        NORM_W,
        norm_eps,
        W,
        Q_OUT,
        QN_W,
        KN_W,
        qk_eps,
        COS_SIN,
        POS,
        K_CACHE,
        V_CACHE,
        cache_stride_row,
        cache_stride_slot,
        cache_stride_head,
        PARTIALS,
        SUM_SQ_PARTIALS,
        COUNTERS,
        T: tl.constexpr,
        K: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PDL: tl.constexpr,
    ):
        """One program per head and K split: the weighted input against the head's qkv
        columns over the split's K range, with the rows' sum of squares alongside; the
        last program of the head sums the partials, applies the norm scale, then norms
        and rotates q and k; q to Q_OUT, k and v to the row's slot."""
        if PDL:
            gdc_wait()
        else:
            pass
        head = tl.program_id(0)
        pid_s = tl.program_id(1)
        HALF: tl.constexpr = HEAD_DIM // 2
        N_TOTAL: tl.constexpr = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
        BLOCKS_PER_SPLIT: tl.constexpr = K // BLOCK_K // SPLIT
        rm = tl.arange(0, BLOCK_M)
        rh = tl.arange(0, HALF)
        rk = tl.arange(0, BLOCK_K)
        row_mask = rm < rows
        rows_off = row_offsets(x_stride_row, x_stride_t, T, BLOCK_M)
        lo_cols = head * HEAD_DIM + rh
        hi_cols = lo_cols + HALF
        square = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)
        acc_lo = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
        acc_hi = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
        for block in range(pid_s * BLOCKS_PER_SPLIT, (pid_s + 1) * BLOCKS_PER_SPLIT):
            k0 = block * BLOCK_K
            x = tl.load(
                X + rows_off[:, None] + (k0 + rk)[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            square = tl.dot(x, tl.trans(x), square)
            scaled = scaled_tile(x, NORM_W, k0, BLOCK_K)
            acc_lo = tl.dot(
                scaled, tl.trans(weight_tile(W, lo_cols, k0, K, BLOCK_K)), acc_lo
            )
            acc_hi = tl.dot(
                scaled, tl.trans(weight_tile(W, hi_cols, k0, K, BLOCK_K)), acc_hi
            )
        gdc_launch_dependents()
        sum_sq = diagonal(square)
        if SPLIT > 1:
            partial_rows = (pid_s * BLOCK_M + rm)[:, None] * N_TOTAL
            tl.store(PARTIALS + partial_rows + lo_cols[None, :], acc_lo)
            tl.store(PARTIALS + partial_rows + hi_cols[None, :], acc_hi)
            tl.store(SUM_SQ_PARTIALS + pid_s * BLOCK_M + rm, sum_sq)
            tl.debug_barrier()
            arrived = tl.atomic_add(COUNTERS + head, 1, sem="acq_rel", scope="gpu")
            is_last = arrived == SPLIT - 1
        else:
            is_last = pid_s == 0
        if is_last:
            if SPLIT > 1:
                acc_lo = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
                acc_hi = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
                sum_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
                for split in tl.static_range(SPLIT):
                    partial_rows = (split * BLOCK_M + rm)[:, None] * N_TOTAL
                    acc_lo += tl.load(
                        PARTIALS + partial_rows + lo_cols[None, :], cache_modifier=".cg"
                    )
                    acc_hi += tl.load(
                        PARTIALS + partial_rows + hi_cols[None, :], cache_modifier=".cg"
                    )
                    sum_sq += tl.load(
                        SUM_SQ_PARTIALS + split * BLOCK_M + rm, cache_modifier=".cg"
                    )
                tl.atomic_xchg(COUNTERS + head, 0)
            else:
                pass
            rstd = libdevice.rsqrt(sum_sq / K + norm_eps)
            lo = (acc_lo * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
            hi = (acc_hi * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
            pos = tl.load(POS + rm, mask=row_mask, other=0)
            cache_rows = rm // T * cache_stride_row + pos * cache_stride_slot
            if head < NUM_Q_HEADS + NUM_KV_HEADS:
                if head < NUM_Q_HEADS:
                    w_lo = tl.load(QN_W + rh).to(tl.float32)
                    w_hi = tl.load(QN_W + HALF + rh).to(tl.float32)
                else:
                    w_lo = tl.load(KN_W + rh).to(tl.float32)
                    w_hi = tl.load(KN_W + HALF + rh).to(tl.float32)
                head_sum_sq = tl.sum(lo * lo, 1) + tl.sum(hi * hi, 1)
                head_rstd = libdevice.rsqrt(head_sum_sq / HEAD_DIM + qk_eps)
                lo = (lo * head_rstd[:, None] * w_lo[None, :]).to(tl.bfloat16)
                hi = (hi * head_rstd[:, None] * w_hi[None, :]).to(tl.bfloat16)
                lo = lo.to(tl.float32)
                hi = hi.to(tl.float32)
                cos = tl.load(COS_SIN + pos[:, None] * HEAD_DIM + rh[None, :])
                sin = tl.load(COS_SIN + pos[:, None] * HEAD_DIM + HALF + rh[None, :])
                out_lo = (lo * cos - hi * sin).to(tl.bfloat16)
                out_hi = (lo * sin + hi * cos).to(tl.bfloat16)
                if head < NUM_Q_HEADS:
                    q_rows = rm * (NUM_Q_HEADS * HEAD_DIM)
                    tl.store(
                        Q_OUT + q_rows[:, None] + lo_cols[None, :],
                        out_lo,
                        mask=row_mask[:, None],
                    )
                    tl.store(
                        Q_OUT + q_rows[:, None] + hi_cols[None, :],
                        out_hi,
                        mask=row_mask[:, None],
                    )
                else:
                    slots = cache_rows + (head - NUM_Q_HEADS) * cache_stride_head
                    tl.store(
                        K_CACHE + slots[:, None] + rh[None, :],
                        out_lo,
                        mask=row_mask[:, None],
                    )
                    tl.store(
                        K_CACHE + slots[:, None] + HALF + rh[None, :],
                        out_hi,
                        mask=row_mask[:, None],
                    )
            else:
                v_head = head - NUM_Q_HEADS - NUM_KV_HEADS
                slots = cache_rows + v_head * cache_stride_head
                tl.store(
                    V_CACHE + slots[:, None] + rh[None, :],
                    lo.to(tl.bfloat16),
                    mask=row_mask[:, None],
                )
                tl.store(
                    V_CACHE + slots[:, None] + HALF + rh[None, :],
                    hi.to(tl.bfloat16),
                    mask=row_mask[:, None],
                )
        else:
            pass

    @triton.jit
    def gemv_add_kernel(
        X,
        x_stride_row,
        x_stride_t,
        x_stride_h,
        rows,
        W,
        RES_IN,
        res_in_stride,
        RES_OUT,
        PARTIALS,
        COUNTERS,
        T: tl.constexpr,
        D: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PDL: tl.constexpr,
    ):
        """One program per output tile and K split: X against the tile's weight rows
        over the split's K range; the last program of the tile sums the partials in
        index order, rounds the product, then writes the rounded sum with RES_IN."""
        if PDL:
            gdc_wait()
        else:
            pass
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        HEADS_PER_SPLIT: tl.constexpr = K // D // SPLIT
        rm = tl.arange(0, BLOCK_M)
        rk = tl.arange(0, BLOCK_K)
        row_mask = rm < rows
        rows_off = row_offsets(x_stride_row, x_stride_t, T, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for head in range(pid_s * HEADS_PER_SPLIT, (pid_s + 1) * HEADS_PER_SPLIT):
            for kk in tl.static_range(0, D, BLOCK_K):
                k0 = head * D + kk
                x = tl.load(
                    X + rows_off[:, None] + (head * x_stride_h + kk + rk)[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                )
                acc = tl.dot(x, tl.trans(weight_tile(W, cols, k0, K, BLOCK_K)), acc)
        gdc_launch_dependents()
        if SPLIT > 1:
            partial_offsets = (pid_s * BLOCK_M + rm)[:, None] * N + cols[None, :]
            tl.store(PARTIALS + partial_offsets, acc)
            tl.debug_barrier()
            arrived = tl.atomic_add(COUNTERS + pid_n, 1, sem="acq_rel", scope="gpu")
            is_last = arrived == SPLIT - 1
        else:
            is_last = pid_s == 0
        if is_last:
            if SPLIT > 1:
                acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                for split in tl.static_range(SPLIT):
                    acc += tl.load(
                        PARTIALS + (split * BLOCK_M + rm)[:, None] * N + cols[None, :],
                        cache_modifier=".cg",
                    )
                tl.atomic_xchg(COUNTERS + pid_n, 0)
            else:
                pass
            product = acc.to(tl.bfloat16).to(tl.float32)
            residual = tl.load(
                RES_IN + rm[:, None] * res_in_stride + cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            tl.store(
                RES_OUT + rm[:, None] * N + cols[None, :],
                (product + residual).to(tl.bfloat16),
                mask=row_mask[:, None],
            )
        else:
            pass

    @triton.jit
    def norm_gate_up_silu_kernel(
        X,
        x_stride_row,
        x_stride_t,
        rows,
        NORM_W,
        norm_eps,
        W,
        OUT,
        T: tl.constexpr,
        K: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PDL: tl.constexpr,
    ):
        """One program per activation tile: the weighted input against the tile's gate
        and up rows with the rows' sum of squares alongside, the norm scale applied to
        both products, both rounded, then silu(gate) * up rounded once."""
        if PDL:
            gdc_wait()
        else:
            pass
        pid = tl.program_id(0)
        rm = tl.arange(0, BLOCK_M)
        rk = tl.arange(0, BLOCK_K)
        row_mask = rm < rows
        rows_off = row_offsets(x_stride_row, x_stride_t, T, BLOCK_M)
        gate_cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        up_cols = INTERMEDIATE + gate_cols
        square = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)
        acc_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        acc_up = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for block in range(0, K // BLOCK_K):
            k0 = block * BLOCK_K
            x = tl.load(
                X + rows_off[:, None] + (k0 + rk)[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            square = tl.dot(x, tl.trans(x), square)
            scaled = scaled_tile(x, NORM_W, k0, BLOCK_K)
            acc_gate = tl.dot(
                scaled, tl.trans(weight_tile(W, gate_cols, k0, K, BLOCK_K)), acc_gate
            )
            acc_up = tl.dot(
                scaled, tl.trans(weight_tile(W, up_cols, k0, K, BLOCK_K)), acc_up
            )
        gdc_launch_dependents()
        rstd = libdevice.rsqrt(diagonal(square) / K + norm_eps)
        gate = (acc_gate * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        up = (acc_up * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        activated = gate / (1.0 + libdevice.exp(-gate)) * up
        offsets = rm[:, None] * INTERMEDIATE + gate_cols[None, :]
        tl.store(OUT + offsets, activated.to(tl.bfloat16), mask=row_mask[:, None])

    @triton.jit
    def codebook_step_kernel(
        LOGITS,
        logits_stride,
        EMBEDDING,
        CODES,
        codes_stride,
        CODEBOOK_INPUT,
        SUMMED,
        summed_stride,
        VOCAB: tl.constexpr,
        HIDDEN: tl.constexpr,
    ):
        """One program per row: the code torch.argmax picks (the first NaN, else the first
        maximum) into the row's code slot, its embedding row as the next pass's input, and
        that row added into the summed embedding with one bf16 rounding of the fp32 sum.
        """
        row = tl.program_id(0)
        vocab = tl.arange(0, VOCAB)
        logits = tl.load(LOGITS + row * logits_stride + vocab).to(tl.float32)
        is_nan = logits != logits
        best = tl.max(tl.where(is_nan, float("-inf"), logits), 0)
        picked = tl.where(tl.max(is_nan.to(tl.int32), 0) > 0, is_nan, logits == best)
        code = tl.min(tl.where(picked, vocab, VOCAB), 0)
        tl.store(CODES + row * codes_stride, code.to(tl.int64))
        hidden = tl.arange(0, HIDDEN)
        embedding = tl.load(EMBEDDING + code * HIDDEN + hidden)
        tl.store(CODEBOOK_INPUT + row * HIDDEN + hidden, embedding)
        summed = tl.load(SUMMED + row * summed_stride + hidden).to(tl.float32)
        tl.store(
            SUMMED + row * summed_stride + hidden,
            (summed + embedding.to(tl.float32)).to(tl.bfloat16),
        )

else:
    pass


def supports_exact_add_rmsnorm(
    hidden_size: int, dtype: torch.dtype, device: torch.device
) -> bool:
    """The kernel reproduces flashinfer's tree for the predictor's shape only."""
    return (
        HAS_TRITON
        and hidden_size == HIDDEN_SIZE
        and dtype == torch.bfloat16
        and device.type == "cuda"
        and current_platform.is_cuda()
    )


def add_rmsnorm_rounded(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """In place: residual becomes bf16(x + residual), x becomes RMSNorm(residual) * weight.

    Returns (normed, residual) with sglang's residual norm contract.
    """
    rows = x.shape[0]
    add_rmsnorm_rounded_kernel[(rows,)](
        x,
        residual,
        weight,
        eps,
        HIDDEN=HIDDEN_SIZE,
        BLOCKS=VEC_BLOCKS,
        LANE_VEC=VEC,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return x, residual


@dataclass(frozen=True)
class PredictorLayerShape:
    """The predictor layer's dimensions and the K split of its residual launches."""

    hidden_size: int
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    intermediate_size: int
    split_qkv: int
    split_hidden: int


def is_plain_bf16_linear(linear: torch.nn.Module) -> bool:
    """Unquantized, bias free, bf16 and whole on this rank: the launches read the
    weight as stored and add the residual without a reduction across ranks."""
    return (
        isinstance(linear.quant_method, UnquantizedLinearMethod)
        and linear.bias is None
        and linear.weight.dtype == torch.bfloat16
        and linear.tp_size == 1
    )


def split_count(output_tiles: int, k_blocks: int, sm_count: int) -> int:
    """The power-of-two K split whose program count lands nearest the SM count."""
    candidates = [split for split in (1, 2, 4, 8) if k_blocks % split == 0]
    return min(candidates, key=lambda split: abs(output_tiles * split - sm_count))


def resolve_predictor_layer_shape(
    code_predictor: torch.nn.Module, predictor_len: int, device: torch.device
) -> PredictorLayerShape | None:
    """Plain bf16 layers with a neox rope on an fp32 cache and dimensions the blocks
    divide; None keeps the plain path."""
    if device.type != "cuda" or not HAS_TRITON:
        return None
    else:
        pass
    layers = code_predictor.model.layers
    attention = layers[0].self_attn
    mlp = layers[0].mlp
    hidden_size = attention.hidden_size
    intermediate_size = mlp.down_proj.weight.shape[1]
    dims_divide = (
        attention.head_dim % BLOCK_K == 0
        and hidden_size % BLOCK_K == 0
        and hidden_size % BLOCK_N == 0
        and hidden_size % MLP_BLOCK_K == 0
        and intermediate_size % BLOCK_K == 0
        and intermediate_size % BLOCK_N == 0
    )
    if not dims_divide:
        return None
    else:
        pass
    for layer in layers:
        attention = layer.self_attn
        rope = attention.rotary_emb
        linears = (
            attention.qkv_proj,
            attention.o_proj,
            layer.mlp.gate_up_proj,
            layer.mlp.down_proj,
        )
        rope_matches = (
            type(rope) is RotaryEmbedding
            and not rope.use_fallback_kernel
            and rope.is_neox_style
            and rope.rotary_dim == attention.head_dim
            and rope.cos_sin_cache.dtype == torch.float32
            and rope.cos_sin_cache.shape[0] >= predictor_len
        )
        if not (
            rope_matches and all(is_plain_bf16_linear(linear) for linear in linears)
        ):
            return None
        else:
            pass
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    heads = attention.num_heads + 2 * attention.num_kv_heads
    split_hidden = min(
        split_count(hidden_size // BLOCK_N, attention.num_heads, sm_count),
        split_count(hidden_size // BLOCK_N, intermediate_size // BLOCK_K, sm_count),
    )
    return PredictorLayerShape(
        hidden_size=hidden_size,
        head_dim=attention.head_dim,
        num_q_heads=attention.num_heads,
        num_kv_heads=attention.num_kv_heads,
        intermediate_size=intermediate_size,
        split_qkv=split_count(heads, hidden_size // BLOCK_K, sm_count),
        split_hidden=split_hidden,
    )


def block_rows(rows: int) -> int:
    return max(MIN_BLOCK_M, triton.next_power_of_2(rows))


def allocate_split_scratch(
    shape: PredictorLayerShape, max_rows: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The split launches' scratch for up to max_rows rows: the fp32 partials as one
    flat run that each launch strides by its own output width, the fp32 partial sums
    of squares, and the int32 tile counters. The dtypes are explicit because the model
    loader constructs the talker under a bf16 default dtype."""
    block_m = block_rows(max_rows)
    qkv_width = (shape.num_q_heads + 2 * shape.num_kv_heads) * shape.head_dim
    partials = torch.zeros(
        block_m
        * max(shape.split_qkv * qkv_width, shape.split_hidden * shape.hidden_size),
        device=device,
        dtype=torch.float32,
    )
    sum_sq_partials = torch.zeros(
        shape.split_qkv * block_m, device=device, dtype=torch.float32
    )
    counters = torch.zeros(
        max(shape.num_q_heads + 2 * shape.num_kv_heads, shape.hidden_size // BLOCK_N),
        device=device,
        dtype=torch.int32,
    )
    return partials, sum_sq_partials, counters


def attention_inputs(
    *,
    x: torch.Tensor,
    tokens_per_row: int,
    norm: torch.nn.Module,
    attention: torch.nn.Module,
    q_out: torch.Tensor,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    partials: torch.Tensor,
    sum_sq_partials: torch.Tensor,
    counters: torch.Tensor,
    shape: PredictorLayerShape,
    pdl: bool,
) -> None:
    """Norm x's rows, project them, norm and rotate q and k; q into q_out and k and v
    into the caches at each row's position. x is (rows, hidden) with unit column stride;
    the caches are (rows // tokens_per_row, slots, kv heads, head dim)."""
    rows = x.shape[0]
    heads = shape.num_q_heads + 2 * shape.num_kv_heads
    norm_qkv_rope_store_kernel[(heads, shape.split_qkv)](
        x,
        tokens_per_row * x.stride(0),
        x.stride(0),
        rows,
        norm.weight,
        norm.variance_epsilon,
        attention.qkv_proj.weight,
        q_out,
        attention.q_norm.weight,
        attention.k_norm.weight,
        attention.q_norm.variance_epsilon,
        attention.rotary_emb.cos_sin_cache,
        positions,
        k_cache,
        v_cache,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        partials,
        sum_sq_partials,
        counters,
        T=tokens_per_row,
        K=shape.hidden_size,
        HEAD_DIM=shape.head_dim,
        NUM_Q_HEADS=shape.num_q_heads,
        NUM_KV_HEADS=shape.num_kv_heads,
        SPLIT=shape.split_qkv,
        BLOCK_M=block_rows(rows),
        BLOCK_K=BLOCK_K,
        PDL=pdl,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        launch_pdl=pdl,
    )


def o_proj_add(
    *,
    attention_output: torch.Tensor,
    tokens_per_row: int,
    weight: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    shape: PredictorLayerShape,
) -> None:
    """residual_out = bf16(bf16(attention_output @ weight.T) + residual_in), reading the
    attention output in its (row group, head, token, head dim) layout and residual_in
    with its own row stride."""
    rows = residual_out.shape[0]
    gemv_add_kernel[(shape.hidden_size // BLOCK_N, shape.split_hidden)](
        attention_output,
        attention_output.stride(0),
        attention_output.stride(2),
        attention_output.stride(1),
        rows,
        weight,
        residual_in,
        residual_in.stride(0),
        residual_out,
        partials,
        counters,
        T=tokens_per_row,
        D=shape.head_dim,
        K=shape.num_q_heads * shape.head_dim,
        N=shape.hidden_size,
        SPLIT=shape.split_hidden,
        BLOCK_M=block_rows(rows),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        # note (ratish): its predecessor is sdpa, which never releases its dependents
        # early, so this launch has nothing to overlap.
        PDL=False,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )


def mlp_up(
    *,
    residual: torch.Tensor,
    norm: torch.nn.Module,
    weight: torch.Tensor,
    activated: torch.Tensor,
    shape: PredictorLayerShape,
) -> None:
    """activated = bf16(silu(bf16(normed @ gate.T)) * bf16(normed @ up.T)) with the
    normed residual, gate and up being the halves of the merged weight."""
    rows = residual.shape[0]
    norm_gate_up_silu_kernel[(shape.intermediate_size // BLOCK_N,)](
        residual,
        residual.stride(0),
        residual.stride(0),
        rows,
        norm.weight,
        norm.variance_epsilon,
        weight,
        activated,
        T=1,
        K=shape.hidden_size,
        INTERMEDIATE=shape.intermediate_size,
        BLOCK_M=block_rows(rows),
        BLOCK_N=BLOCK_N,
        BLOCK_K=MLP_BLOCK_K,
        PDL=True,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        launch_pdl=True,
    )


def down_add(
    *,
    activated: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    shape: PredictorLayerShape,
) -> None:
    """residual = bf16(bf16(activated @ weight.T) + residual), in place."""
    rows = residual.shape[0]
    gemv_add_kernel[(shape.hidden_size // BLOCK_N, shape.split_hidden)](
        activated,
        activated.stride(0),
        activated.stride(0),
        BLOCK_K,
        rows,
        weight,
        residual,
        residual.stride(0),
        residual,
        partials,
        counters,
        T=1,
        D=BLOCK_K,
        K=shape.intermediate_size,
        N=shape.hidden_size,
        SPLIT=shape.split_hidden,
        BLOCK_M=block_rows(rows),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        PDL=True,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        launch_pdl=True,
    )


def supports_codebook_step(
    embedding_weight: torch.Tensor, activation_dtype: torch.dtype
) -> bool:
    """The fused codebook step reproduces the argmax, gather and add bit for bit for
    bf16 embeddings added into bf16 activations, with power-of-two widths, on CUDA."""
    vocab_size, hidden_size = embedding_weight.shape
    return (
        HAS_TRITON
        and embedding_weight.device.type == "cuda"
        and current_platform.is_cuda()
        and embedding_weight.dtype == torch.bfloat16
        and activation_dtype == torch.bfloat16
        and vocab_size == triton.next_power_of_2(vocab_size)
        and hidden_size == triton.next_power_of_2(hidden_size)
    )


def codebook_step(
    logits: torch.Tensor,
    embedding_weight: torch.Tensor,
    codes: torch.Tensor,
    summed: torch.Tensor,
) -> torch.Tensor:
    """codes = argmax(logits) as torch.argmax picks it, summed += embedding[codes]
    in place, in one launch; returns the embedding rows (rows, 1, hidden) the next
    predictor pass takes. logits is (rows, vocab); codes is a (rows,) int64 view."""
    rows, vocab_size = logits.shape
    hidden_size = embedding_weight.shape[1]
    codebook_input = torch.empty(
        (rows, 1, hidden_size), device=summed.device, dtype=summed.dtype
    )
    codebook_step_kernel[(rows,)](
        logits,
        logits.stride(0),
        embedding_weight,
        codes,
        codes.stride(0),
        codebook_input,
        summed,
        summed.stride(0),
        VOCAB=vocab_size,
        HIDDEN=hidden_size,
        num_warps=NUM_WARPS,
    )
    return codebook_input


class FusedPredictorLayers:
    """The predictor's decoder layers as the four fused launches per layer around SDPA,
    with q, the residual stream and the MLP activation in buffers of max_rows rows."""

    def __init__(
        self,
        shape: PredictorLayerShape,
        max_rows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.shape = shape
        self.max_rows = max_rows
        self.q = torch.zeros(
            max_rows, shape.num_q_heads * shape.head_dim, device=device, dtype=dtype
        )
        self.residual = torch.zeros(
            max_rows, shape.hidden_size, device=device, dtype=dtype
        )
        self.activated = torch.zeros(
            max_rows, shape.intermediate_size, device=device, dtype=dtype
        )
        self.partials, self.sum_sq_partials, self.counters = allocate_split_scratch(
            shape, max_rows, device
        )

    def covers(self, rows: int) -> bool:
        return rows <= self.max_rows

    def forward(
        self,
        *,
        layers: torch.nn.ModuleList,
        final_norm: torch.nn.Module,
        token_embeds: torch.Tensor,
        batch_size: int,
        cache_len: int,
        positions: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """token_embeds (batch, tokens, hidden) at slots cache_len onward; the caches are
        (layer, row, slot, kv head, head dim), the slot being the position. Returns the
        final norm of the residual stream as (batch, tokens, hidden)."""
        shape = self.shape
        num_tokens, hidden_size = token_embeds.shape[1:]
        rows = batch_size * num_tokens
        end = cache_len + num_tokens
        hidden = token_embeds.reshape(rows, hidden_size)
        residual = self.residual[:rows]
        q_out = self.q[:rows]
        activated = self.activated[:rows]
        for layer_idx, layer in enumerate(layers):
            attention = layer.self_attn
            layer_k_cache = k_cache[layer_idx, :batch_size]
            layer_v_cache = v_cache[layer_idx, :batch_size]
            attention_inputs(
                x=hidden,
                tokens_per_row=num_tokens,
                norm=layer.input_layernorm,
                attention=attention,
                q_out=q_out,
                positions=positions,
                k_cache=layer_k_cache,
                v_cache=layer_v_cache,
                partials=self.partials,
                sum_sq_partials=self.sum_sq_partials,
                counters=self.counters,
                shape=shape,
                # note (ratish): the first layer follows the caller's embedding ops, which
                # never release their dependents early; later layers follow down_add.
                pdl=layer_idx > 0,
            )
            attention_output = torch.nn.functional.scaled_dot_product_attention(
                q_out.view(
                    batch_size, num_tokens, shape.num_q_heads, shape.head_dim
                ).transpose(1, 2),
                layer_k_cache[:, :end].transpose(1, 2),
                layer_v_cache[:, :end].transpose(1, 2),
                is_causal=num_tokens > 1,
                enable_gqa=shape.num_q_heads != shape.num_kv_heads,
            )
            o_proj_add(
                attention_output=attention_output,
                tokens_per_row=num_tokens,
                weight=attention.o_proj.weight,
                residual_in=hidden,
                residual_out=residual,
                partials=self.partials,
                counters=self.counters,
                shape=shape,
            )
            hidden = residual
            mlp_up(
                residual=residual,
                norm=layer.post_attention_layernorm,
                weight=layer.mlp.gate_up_proj.weight,
                activated=activated,
                shape=shape,
            )
            down_add(
                activated=activated,
                weight=layer.mlp.down_proj.weight,
                residual=residual,
                partials=self.partials,
                counters=self.counters,
                shape=shape,
            )
        normed = final_norm(residual)
        return normed.reshape(batch_size, num_tokens, hidden_size)


def resolve_fused_predictor_layers(
    code_predictor: torch.nn.Module,
    predictor_len: int,
    max_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> FusedPredictorLayers | None:
    """The fused layers for this predictor, or None where it keeps the plain path. The
    opening pair runs as one pass of two rows per request; the layers cover passes of
    up to MAX_FUSED_ROWS rows."""
    shape = resolve_predictor_layer_shape(code_predictor, predictor_len, device)
    # note (ratish): deterministic inference promises the same bits at every batch size,
    # and a pass above MAX_FUSED_ROWS runs plain.
    if shape is None or get_exec().deterministic.enable_deterministic_inference:
        return None
    else:
        max_rows = min(2 * max_batch_size, MAX_FUSED_ROWS)
        return FusedPredictorLayers(shape, max_rows, device, dtype)
