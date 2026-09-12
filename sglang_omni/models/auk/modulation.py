# SPDX-License-Identifier: Apache-2.0
"""AuK modulation fusions preserving eager intermediate rounding."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 512}, num_warps=4),
            triton.Config({"BLOCK": 1024}, num_warps=4),
        ],
        # Tune once per hidden size/operator, not for every request length.
        key=["D", "MODULATE"],
    )
    @triton.jit(do_not_specialize=["N", "T", "stride_a", "stride_b"])
    def _modulation_kernel(
        X,
        A,
        B,
        Y,
        N,
        T,
        stride_a,
        stride_b,
        D: tl.constexpr,
        MODULATE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        batch = i // (T * D)
        col = i % D
        x = tl.load(X + i, mask, other=0).to(tl.float32)
        a = tl.load(A + batch * stride_a + col, mask, other=0).to(tl.float32)
        if MODULATE:
            b = tl.load(B + batch * stride_b + col, mask, other=0).to(tl.float32)
            scale = (1 + a).to(A.dtype.element_ty).to(tl.float32)
            product = (x * scale).to(Y.dtype.element_ty).to(tl.float32)
            y = product + b
        else:
            b = tl.load(B + i, mask, other=0).to(tl.float32)
            # gate * update rounds before the (often FP32) residual addition.
            product = (a * b).to(B.dtype.element_ty).to(tl.float32)
            y = x + product
        tl.store(Y + i, y.to(Y.dtype.element_ty), mask)


def _can_fuse(x, a, b):
    # Small tensors do not amortize the Python/Triton launch overhead.
    return (
        x.numel() >= 2**22
        and triton is not None
        and x.is_cuda
        and not torch.is_grad_enabled()
        and x.is_contiguous()
        and a.stride(-1) == b.stride(-1) == 1
        and a.dtype == b.dtype
        and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and a.dtype in (torch.float32, torch.bfloat16, torch.float16)
    )


def _fused(x, a, b, *, modulate):
    out = torch.empty_like(x, dtype=torch.promote_types(x.dtype, a.dtype))
    _modulation_kernel[lambda meta: (triton.cdiv(x.numel(), meta["BLOCK"]),)](
        x,
        a,
        b,
        out,
        x.numel(),
        x.shape[1],
        a.stride(0),
        b.stride(0),
        x.shape[-1],
        modulate,
        enable_fp_fusion=False,
    )
    return out


def scale_shift(x, scale, shift):
    """Apply per-request AdaLN scale/shift to an already normalized [B,T,D]."""
    if _can_fuse(x, scale, shift):
        return _fused(x, scale, shift, modulate=True)
    return x * (1 + scale[:, None]) + shift[:, None]


def gated_residual(x, gate, update):
    """Apply a [B,D] gate and add the update with eager dtype promotion."""
    if _can_fuse(x, gate, update) and update.is_contiguous():
        return _fused(x, gate, update, modulate=False)
    return x + gate[:, None] * update
