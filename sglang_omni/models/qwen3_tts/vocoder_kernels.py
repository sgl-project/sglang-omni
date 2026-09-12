# SPDX-License-Identifier: Apache-2.0
"""Fused SnakeBeta activation for the Qwen3-TTS 12Hz vocoder decoder.

The qwen-tts SnakeBeta.forward evaluates, on a [B, C, T] bf16 tensor::

    alpha = torch.exp(self.alpha[None, :, None])
    beta = torch.exp(self.beta[None, :, None])
    x = x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)

as eight separate elementwise CUDA kernels. Every one of those ops computes
in fp32 and rounds to bf16 once (PyTorch opmath semantics), so the fused
kernel replays the identical fp32 op chain with one round-to-nearest-even
bf16 conversion after every step and is bitwise identical to eager:

    a  = bf16(expf(alpha_c))         # torch.exp(alpha)
    b  = bf16(expf(beta_c))          # torch.exp(beta)
    t  = bf16(f32(b) + 1e-9f)        # beta + no_div_by_zero
    r  = bf16(1.0f / f32(t))         # 1.0 / (...)   (div.rn)
    s  = bf16(f32(x) * f32(a))       # x * alpha
    sn = bf16(sinf(f32(s)))          # torch.sin
    p  = bf16(f32(sn) * f32(sn))     # torch.pow(., 2)
    m  = bf16(f32(r) * f32(p))       # (1/..) * pow
    y  = bf16(f32(x) + f32(m))       # x + ...

Triton is compiled with enable_reflect_ftz=False so libdevice
expf/sinf/div.rn keep denormal handling identical to PyTorch's (nvcc
default -ftz=false), and with enable_fp_fusion=False so LLVM
cannot contract mul + add across the intermediate bf16 roundings into an
FMA (which would skip one rounding step and break exact ties). Both were
validated exhaustively over all 65536 bf16 input encodings per op.

This module must import on hosts without a GPU: Triton usage is guarded
and every entry point degrades to None / the eager arithmetic.
"""

from __future__ import annotations

import logging

import torch

try:  # keep the module importable when Triton is unavailable
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    libdevice = None
    _HAS_TRITON = False

_ALLOWED_CHANNELS = frozenset((1536, 768, 384, 192, 96))
_MAX_BATCH = 8
_MAX_T = 65536

logger = logging.getLogger(__name__)

if _HAS_TRITON:

    @triton.jit(do_not_specialize=["C", "T"])
    def _snake_beta_kernel(
        x_ptr,
        y_ptr,
        alpha_ptr,
        beta_ptr,
        C,
        T,
        BLOCK: tl.constexpr,
    ):
        # One program handles BLOCK elements of one (b, c) row. Every
        # intermediate is rounded fp32 -> bf16 (RNE) exactly where the eager
        # chain materializes a bf16 tensor.
        row = tl.program_id(0).to(tl.int64)
        c = row % C
        a_raw = tl.load(alpha_ptr + c).to(tl.float32)
        b_raw = tl.load(beta_ptr + c).to(tl.float32)
        a = libdevice.exp(a_raw).to(tl.bfloat16).to(tl.float32)
        b = libdevice.exp(b_raw).to(tl.bfloat16).to(tl.float32)
        t = (b + 1e-9).to(tl.bfloat16).to(tl.float32)
        r = libdevice.div_rn(1.0, t).to(tl.bfloat16).to(tl.float32)

        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < T
        ptrs = row * T + offs
        xv = tl.load(x_ptr + ptrs, mask=mask, other=0).to(tl.float32)
        s = (xv * a).to(tl.bfloat16).to(tl.float32)
        sn = libdevice.sin(s).to(tl.bfloat16).to(tl.float32)
        p = (sn * sn).to(tl.bfloat16).to(tl.float32)
        m = (r * p).to(tl.bfloat16).to(tl.float32)
        y = (xv + m).to(tl.bfloat16)
        tl.store(y_ptr + ptrs, y, mask=mask)


def _block_for(t: int) -> int:
    if t <= 64:
        return 64
    if t <= 128:
        return 128
    if t <= 256:
        return 256
    return 1024


def _launch(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    batch, channels, t = x.shape
    out = torch.empty_like(x)
    block = _block_for(t)
    grid = (batch * channels, triton.cdiv(t, block))
    with torch.cuda.device_of(x):
        _snake_beta_kernel[grid](
            x,
            out,
            alpha,
            beta,
            channels,
            t,
            BLOCK=block,
            num_warps=4,
            enable_reflect_ftz=False,
            enable_fp_fusion=False,
        )
    return out


def fused_snake_beta(
    x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor | None:
    """Fused SnakeBeta. Returns None outside the supported envelope.

    Envelope: x [B, C, T] bfloat16 contiguous CUDA, alpha/beta
    bfloat16 [C] on the same device, 1 <= B <= 8, C in {1536, 768, 384,
    192, 96}, 1 <= T <= 65536. Inside the envelope the result is bitwise
    identical to the eager qwen-tts SnakeBeta.forward. Never raises and
    never synchronizes with the host (safe under CUDA graph capture).
    """
    try:
        if not _HAS_TRITON:
            return None
        if (
            not isinstance(x, torch.Tensor)
            or not isinstance(alpha, torch.Tensor)
            or not isinstance(beta, torch.Tensor)
        ):
            return None
        if (
            x.dtype is not torch.bfloat16
            or alpha.dtype is not torch.bfloat16
            or beta.dtype is not torch.bfloat16
        ):
            return None
        if x.device.type != "cuda":
            return None
        if alpha.device != x.device or beta.device != x.device:
            return None
        if x.dim() != 3 or not x.is_contiguous():
            return None
        batch, channels, t = x.shape
        if channels not in _ALLOWED_CHANNELS:
            return None
        if not (1 <= batch <= _MAX_BATCH) or not (1 <= t <= _MAX_T):
            return None
        if alpha.shape != (channels,) or beta.shape != (channels,):
            return None
        if not alpha.is_contiguous():
            alpha = alpha.contiguous()
        if not beta.is_contiguous():
            beta = beta.contiguous()
    except Exception:
        return None
    return _launch(x, alpha, beta)


class FusedSnakeBeta(torch.nn.Module):
    """Drop in SnakeBeta: fused kernel inside the contract, eager outside."""

    def __init__(self, original: torch.nn.Module) -> None:
        super().__init__()
        self.in_features = getattr(original, "in_features", None)
        self.alpha = original.alpha
        self.beta = original.beta
        self.no_div_by_zero = original.no_div_by_zero

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        fused = fused_snake_beta(hidden_states, self.alpha, self.beta)
        if fused is not None:
            return fused
        # Original qwen-tts SnakeBeta arithmetic, op for op.
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        hidden_states = hidden_states + (
            1.0 / (beta + self.no_div_by_zero)
        ) * torch.pow(torch.sin(hidden_states * alpha), 2)
        return hidden_states


def _is_snake_beta(module: torch.nn.Module) -> bool:
    return (
        type(module).__name__ == "SnakeBeta"
        and isinstance(getattr(module, "alpha", None), torch.Tensor)
        and isinstance(getattr(module, "beta", None), torch.Tensor)
    )


def _prewarm(device: torch.device) -> None:
    """Compile every kernel variant before CUDA graph capture can begin.

    All integer arguments are do_not_specialize, so one binary per BLOCK
    covers every envelope shape; a JIT compile can then never happen inside
    a stream capture.
    """
    if not _HAS_TRITON or device.type != "cuda":
        return
    with torch.cuda.device(device):
        for t in (2, 128, 256, 1024):  # one T per BLOCK bucket
            x = torch.zeros((1, 96, t), dtype=torch.bfloat16, device=device)
            ab = torch.zeros((96,), dtype=torch.bfloat16, device=device)
            _launch(x, ab, ab)


def _prewarm_replacements(
    replacements: list[tuple[torch.nn.Module, str]],
) -> None:
    devices = {
        getattr(parent, name).alpha.device
        for parent, name in replacements
        if getattr(parent, name).alpha.device.type == "cuda"
    }
    for device in devices:
        _prewarm(device)


def fuse_vocoder_decoder(decoder: torch.nn.Module) -> int:
    """Replace every SnakeBeta in decoder with FusedSnakeBeta.

    Returns the number of modules replaced. Safe to call more than once
    (already fused modules are left alone).
    """
    if not isinstance(decoder, torch.nn.Module):
        return 0
    replacements: list[tuple[torch.nn.Module, str]] = []
    for module in decoder.modules():
        for name, child in module.named_children():
            if _is_snake_beta(child):
                replacements.append((module, name))

    if replacements and _HAS_TRITON and torch.cuda.is_available():
        try:
            # Note(Jiaxin): compile before mutating the decoder so a failed
            # prewarm leaves the proven eager implementation intact.
            _prewarm_replacements(replacements)
        except Exception:
            logger.warning(
                "Qwen3-TTS fused SnakeBeta prewarm failed; keeping eager modules",
                exc_info=True,
            )
            return 0

    for parent, name in replacements:
        setattr(parent, name, FusedSnakeBeta(getattr(parent, name)))
    return len(replacements)


__all__ = ["fused_snake_beta", "fuse_vocoder_decoder", "FusedSnakeBeta"]
