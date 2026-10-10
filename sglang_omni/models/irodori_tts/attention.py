"""Attention dispatch helpers.

Boolean key-padding masks make ``F.scaled_dot_product_attention`` ineligible for
the FlashAttention backend, so masked attention silently falls back to slower
kernels. This module routes masked attention to FlashAttention-3 varlen kernels
when available (Hopper+, fp16/bf16) and otherwise to SDPA with a cuDNN-first
backend priority, which is markedly faster than the default dispatch for
bool-masked inputs.

All helpers take tensors in (B, S, H, D) layout.

Invariant: key-padding masks in this codebase are prefix-contiguous (right
padding). The FA3 ``seqused`` paths rely on that invariant; callers must not
pass masks with holes.

Set ``IRODORI_ATTENTION_BACKEND=sdpa`` to disable the FA3 paths at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

try:  # optional dependency: flash_attn_3 prebuilt wheel installs this module
    from flash_attn_interface import flash_attn_varlen_func as fa3_attention_varlen_func
except Exception:  # pragma: no cover - flash_attn_3 not installed
    fa3_attention_varlen_func = None

# cuDNN handles bool-masked SDPA well on Ampere+; keep the standard fallbacks
# behind it for shapes/dtypes it rejects.
_SDPA_PRIORITY = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]

_FA3_MIN_COMPUTE_CAPABILITY = 9
_FA3_DTYPES = (torch.float16, torch.bfloat16)

fa3_device_ok_cache: dict[int, bool] = {}


def fa3_disabled() -> bool:
    return os.environ.get("IRODORI_ATTENTION_BACKEND", "").strip().lower() == "sdpa"


def fa3_probe(device: torch.device) -> bool:
    """Run a minimal FA3 call once to confirm the installed kernels actually
    support this GPU (a capability check alone would wrongly admit
    architectures the wheel was not built for, e.g. Blackwell with an
    SM90-only build)."""
    try:
        with torch.no_grad():
            q = torch.zeros(2, 1, 64, device=device, dtype=torch.bfloat16)
            cu = torch.tensor([0, 2], device=device, dtype=torch.int32)
            fa3_attention(q, q, q, cu, cu, 2, 2)
        return True
    except Exception:
        return False


def fa3_device_ok(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    else:
        pass
    index = device.index if device.index is not None else torch.cuda.current_device()
    ok = fa3_device_ok_cache.get(index)
    if ok is None:
        major, _ = torch.cuda.get_device_capability(index)
        ok = major >= _FA3_MIN_COMPUTE_CAPABILITY and fa3_probe(
            torch.device("cuda", index)
        )
        fa3_device_ok_cache[index] = ok
    else:
        pass
    return ok


def fa3_usable(device: torch.device, dtype: torch.dtype) -> bool:
    return (
        fa3_attention_varlen_func is not None
        and not fa3_disabled()
        and dtype in _FA3_DTYPES
        and fa3_device_ok(device)
    )


def effective_attention_dtype(x: torch.Tensor) -> torch.dtype:
    """Dtype attention inputs will have after autocast, if it is active."""
    if x.is_cuda and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    else:
        pass
    return x.dtype


def fa3_attention(q, k, v, cu_q, cu_k, max_q, max_k, *, seqused_q=None, seqused_k=None):
    out = fa3_attention_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
    )
    if isinstance(out, tuple):
        out = out[0]
    else:
        pass
    return out


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    y = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
        is_causal=False,
    )
    return y.transpose(1, 2)


def maskedsdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    """SDPA in (B, S, H, D) layout with the cuDNN-first backend priority."""
    if attn_mask is None or not q.is_cuda:
        return sdpa_attention(q, k, v, attn_mask)
    else:
        pass
    with sdpa_kernel(_SDPA_PRIORITY, set_priority=True):
        return sdpa_attention(q, k, v, attn_mask)


def fixed_stride_cumulative_sequence_lengths(batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(
        0,
        (batch + 1) * seq_len,
        seq_len,
        device=device,
        dtype=torch.int32,
    )


def prefix_key_mask_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Self-attention with a shared prefix-contiguous key/query padding mask.

    q, k, v: (B, S, H, D); key_mask: (B, S) bool or None.

    Output matches SDPA-with-bool-mask semantics on valid rows. Padding rows
    are zeroed on the FA3 path (SDPA leaves finite garbage there); rows whose
    mask is entirely False yield zeros on both paths, matching the efficient
    backend's behavior the previous implementation relied on.
    """
    if key_mask is None:
        # No mask: the default dispatch already picks FlashAttention.
        return sdpa_attention(q, k, v, None)
    else:
        pass
    if fa3_usable(q.device, q.dtype) and k.dtype == q.dtype and v.dtype == q.dtype:
        bsz, seq_len, heads, head_dim = q.shape
        # Fully masked rows attend to position 0 instead; combined with the
        # input masking below their keys/values are zero, so the result stays
        # zero.
        seqused = key_mask.sum(dim=1, dtype=torch.int32).clamp_(min=1)
        cu = fixed_stride_cumulative_sequence_lengths(bsz, seq_len, q.device)
        # The kernel leaves rows beyond ``seqused`` uninitialized (possibly
        # NaN), both in the forward output and in the backward dq/dk/dv.
        # masked_fill cleans them NaN-safely: its backward fills the padding
        # rows of the incoming gradient with exact zeros (a plain multiply
        # would keep NaN * 0 = NaN).
        pad_rows = ~key_mask[:, :, None, None]
        q = q.masked_fill(pad_rows, 0)
        k = k.masked_fill(pad_rows, 0)
        v = v.masked_fill(pad_rows, 0)
        out = fa3_attention(
            q.reshape(bsz * seq_len, heads, head_dim),
            k.reshape(bsz * seq_len, heads, head_dim),
            v.reshape(bsz * seq_len, heads, head_dim),
            cu,
            cu,
            seq_len,
            seq_len,
            seqused_q=seqused,
            seqused_k=seqused,
        )
        out = out.view(bsz, seq_len, heads, head_dim)
        return out.masked_fill(pad_rows, 0)
    else:
        pass
    # SDPA fallback: guarantee >=1 valid key per row (sync-free). The efficient
    # backend returns zeros for fully masked rows but cuDNN does not promise
    # that; redirecting empty rows to position 0 keeps them finite, and callers
    # only ever produce such rows with zeroed inputs, so the result stays zero.
    safe_mask = key_mask.clone()
    safe_mask[:, 0] |= ~key_mask.any(dim=1)
    return maskedsdpa_attention(q, k, v, safe_mask[:, None, None, :])


@dataclass
class ContextAttentionPlan:
    """
    Precomputed packing info for attention over concatenated context segments.

    The concatenated key mask is not prefix-contiguous (each segment is padded
    independently), so the FA3 path gathers valid keys into a packed layout.
    Build the plan once per forward and reuse it across blocks: constructing
    ``pack_index`` requires a device sync.
    """

    kv_mask: torch.Tensor  # (B, S_kv) bool, concatenated segment masks
    query_len: int
    usefa3_attention: bool
    pack_index: torch.Tensor | None = None  # (N_valid,) into flattened (B * S_kv)
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_k: int = 0


def build_context_attention_plan(
    segment_masks: list[torch.Tensor],
    *,
    query_len: int,
    attention_dtype: torch.dtype,
) -> ContextAttentionPlan:
    kv_mask = segment_masks[0] if len(segment_masks) == 1 else torch.cat(segment_masks, dim=1)
    device = kv_mask.device
    if not fa3_usable(device, attention_dtype):
        return ContextAttentionPlan(kv_mask=kv_mask, query_len=int(query_len), usefa3_attention=False)
    else:
        pass
    bsz = kv_mask.shape[0]
    pack_index = torch.nonzero(kv_mask.reshape(-1), as_tuple=False).flatten()
    kv_lens = kv_mask.sum(dim=1, dtype=torch.int32)
    cu_k = F.pad(torch.cumsum(kv_lens, dim=0, dtype=torch.int32), (1, 0))
    return ContextAttentionPlan(
        kv_mask=kv_mask,
        query_len=int(query_len),
        usefa3_attention=True,
        pack_index=pack_index,
        cu_seqlens_q=fixed_stride_cumulative_sequence_lengths(bsz, int(query_len), device),
        cu_seqlens_k=cu_k,
        max_seqlen_k=int(kv_lens.max()),
    )


def context_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: ContextAttentionPlan,
) -> torch.Tensor:
    """
    Attention of q over concatenated (and independently padded) kv segments.

    q: (B, S_q, H, D); k, v: (B, S_kv, H, D). All query rows are computed,
    matching SDPA behavior (padding query rows hold garbage either way and are
    masked downstream). Every kv row must have at least one valid key, which
    holds because the self segment always contains valid tokens.
    """
    if (
        plan.usefa3_attention
        and fa3_usable(q.device, q.dtype)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
    ):
        bsz, sq, heads, head_dim = q.shape
        if sq != plan.query_len:
            raise ValueError(
                f"context plan was built for query_len={plan.query_len}, got {sq}"
            )
        else:
            pass
        skv = k.shape[1]
        k_packed = k.reshape(bsz * skv, heads, head_dim).index_select(0, plan.pack_index)
        v_packed = v.reshape(bsz * skv, heads, head_dim).index_select(0, plan.pack_index)
        out = fa3_attention(
            q.reshape(bsz * sq, heads, head_dim),
            k_packed,
            v_packed,
            plan.cu_seqlens_q,
            plan.cu_seqlens_k,
            sq,
            plan.max_seqlen_k,
        )
        return out.view(bsz, sq, heads, head_dim)
    else:
        pass
    return maskedsdpa_attention(q, k, v, plan.kv_mask[:, None, None, :])
