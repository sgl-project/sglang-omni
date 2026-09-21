# SPDX-License-Identifier: Apache-2.0
"""Request-scoped layouts for AuK's non-causal packed transformer trunk."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, partial

import torch


def upstream_flash():
    """SGLang's own device predicates and varlen entry points.

    Imported lazily: the padded path and CPU model loading never need Flash,
    and tests substitute this to describe other hardware.
    """
    from types import SimpleNamespace

    from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
    from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported
    from sglang.kernels.ops.attention.flash_attention_v4 import (
        is_flash_attention_v4_available,
    )
    from sglang.srt.utils import is_blackwell

    return SimpleNamespace(
        is_blackwell=is_blackwell,
        is_fa3_supported=_is_fa3_supported,
        is_fa4_available=is_flash_attention_v4_available,
        flash_attn_varlen_func=flash_attn_varlen_func,
    )


@cache
def resolve_flash_version(device: torch.device) -> int:
    """Pick the SGLang varlen FlashAttention version that serves device.

    The decision reuses SGLang's predicates instead of an SM table of its own:
    is_blackwell (sm100/sm110/sm120 with CUDA >= 12.8) selects FA4, the
    version SGLang's FlashAttentionBackend runs on every Blackwell part, and
    _is_fa3_supported (sm80-sm90 with CUDA >= 12.3) selects FA3. Both look
    at the current CUDA device, as they do upstream. Anything else is rejected
    with a reason at startup instead of failing on the first batch.
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(f"Packed AuK DiT requires a CUDA device, got {device}")
    elif torch.version.hip is not None:
        raise ValueError(
            "Packed AuK DiT is unsupported on HIP: SGLang ships no FA3/FA4 varlen "
            "kernel there"
        )
    flash = upstream_flash()
    major, minor = torch.cuda.get_device_capability(device)
    if flash.is_blackwell():
        if flash.is_fa4_available():
            return 4
        raise ValueError(
            f"Packed AuK DiT needs FlashAttention 4 on sm{major}{minor}, but "
            "SGLang's FA4 varlen kernel is unavailable (install flash-attn-4)"
        )
    elif flash.is_fa3_supported():
        return 3
    else:
        raise ValueError(
            f"Packed AuK DiT is unsupported on sm{major}{minor}: SGLang selects "
            "FA3 on sm80-sm90 (CUDA >= 12.3) and FA4 on Blackwell (CUDA >= 12.8)"
        )


@cache
def varlen_func(version: int, device: torch.device):
    """The varlen kernel SGLang's FlashAttentionBackend binds for version.

    FA4 on sm12x goes through flash_attention_v4_sm120, SGLang's own SM120
    launch path; every other combination goes through the generic
    flash_attn_varlen_func(ver=...) dispatcher.
    """
    if version == 4 and torch.cuda.get_device_capability(device)[0] == 12:
        from sglang.kernels.ops.attention.flash_attention_v4_sm120 import (
            flash_attn_varlen_func,
        )

        return flash_attn_varlen_func
    return partial(upstream_flash().flash_attn_varlen_func, ver=version)


def probe_flash_attention(device: torch.device, *, heads: int, head_dim: int) -> None:
    """Run one tiny varlen call with the DiT's head shape.

    Missing kernels, JIT failures and unsupported head sizes then surface at
    stage construction rather than on the first multi-request batch.
    """
    mask = torch.ones(2, 4, dtype=torch.bool, device=device)
    layout = PackedLayout.build(
        mask, mask, prompt_width=0, target_width=4, max_seqlen=8
    )
    q = torch.randn(16, heads, head_dim, dtype=torch.bfloat16, device=device)
    out = flash_attention(q, q, q, layout)
    if torch.isfinite(out).all():
        return None
    else:
        raise RuntimeError(
            f"Packed AuK DiT FlashAttention probe returned non-finite values for "
            f"{tuple(q.shape)} on {device}"
        )


def flash_attention(q, k, v, layout):
    # Keyword arguments: the SM120 entry point orders its parameters differently.
    return varlen_func(layout.flash_version, q.device)(
        q,
        k,
        v,
        cu_seqlens_q=layout.cu_seqlens,
        cu_seqlens_k=layout.cu_seqlens,
        max_seqlen_q=layout.max_seqlen,
        max_seqlen_k=layout.max_seqlen,
        causal=False,
    )


def gather_rows(x, indices):
    return x.flatten(0, 1).index_select(0, indices)


def gather_rope(rope, indices, batch):
    """Packed rows of [T, D] (shared) or [B, T, D] (per-request) frequencies.

    The DiT's RotaryEmbedding has no xpos, so scale is a scalar and
    passes through unchanged.
    """
    freqs, scale = rope
    freqs = freqs.reshape(-1, *freqs.shape[-2:]).expand(batch, -1, -1)
    return gather_rows(freqs, indices).unsqueeze(0), scale


@dataclass(frozen=True)
class PackedLayout:
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    joint_indices: torch.Tensor
    # Permutations between separate packed streams and request-major streams.
    double_order: torch.Tensor
    double_inverse: torch.Tensor
    single_order: torch.Tensor
    target_rows: torch.Tensor
    target_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    batch: int
    target_width: int
    flash_version: int

    @classmethod
    def build(cls, audio_mask, text_mask, prompt_width, target_width, max_seqlen):
        """Build once per trajectory, outside the Euler loop and graph capture.

        Select the masks themselves, not a prefix inferred from their sums:
        text and reference tensors can contain stored padding and holes.
        max_seqlen bounds every request's joint length from host-side
        lengths, so the build only synchronizes for the nonzero calls.
        """
        batch, audio_width = audio_mask.shape
        text_width = text_mask.shape[1]
        joint_width = text_width + audio_width
        ai = audio_mask.flatten().nonzero().flatten()
        ci = text_mask.flatten().nonzero().flatten()
        ab = ai // audio_width
        cb = ci // text_width
        double_order = torch.argsort(torch.cat((ab, cb)), stable=True)
        single_order = torch.argsort(torch.cat((cb, ab)), stable=True)
        # Row-major indices into [batch, text ++ audio], in single-stream order.
        ji = torch.cat(
            (
                cb * joint_width + ci % text_width,
                ab * joint_width + text_width + ai % audio_width,
            )
        ).index_select(0, single_order)
        local = ji % joint_width - text_width - prompt_width
        target_rows = (local >= 0).nonzero().flatten()
        target_indices = (ji // joint_width) * target_width + local
        lengths = torch.cat((text_mask, audio_mask), dim=1).sum(1, dtype=torch.int32)
        cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
        return cls(
            ai,
            ci,
            ji,
            double_order,
            torch.argsort(double_order),
            single_order,
            target_rows,
            target_indices.index_select(0, target_rows),
            cu,
            max_seqlen,
            batch,
            target_width,
            # The kernel is CUDA-only; a non-CUDA layout only ever reaches a
            # substituted attention, so its version is never dispatched on.
            resolve_flash_version(audio_mask.device) if audio_mask.is_cuda else 3,
        )

    def unpack_target(self, x):
        output = x.new_zeros(self.batch * self.target_width, x.shape[-1])
        output.index_copy_(0, self.target_indices, x)
        return output.view(self.batch, self.target_width, x.shape[-1])
