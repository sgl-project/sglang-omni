# SPDX-License-Identifier: Apache-2.0
"""Request-scoped layouts for AuK's non-causal packed transformer trunk."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch


@cache
def resolve_flash_version(device: torch.device) -> int:
    """Pick the SGLang varlen FlashAttention build that serves ``device``.

    FA4 on Blackwell (sm100 and the sm120 consumer parts, where the varlen
    kernel is validated; sm103 excluded as in SGLang's VisionAttention) and
    FA3 on sm80-sm90 via the MOSS-Audio-Tokenizer gate (``_is_fa3_supported``),
    so an unsupported device is rejected with a reason instead of failing on
    the first batch.
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(f"Packed AuK DiT requires a CUDA device, got {device}")
    elif torch.version.hip is not None:
        raise ValueError(
            "Packed AuK DiT is unsupported on HIP: SGLang ships no FA3/FA4 varlen "
            "kernel there"
        )
    else:
        # Lazy imports: the padded path and CPU model loading never need Flash.
        from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported
        from sglang.kernels.ops.attention.flash_attention_v4 import (
            is_flash_attention_v4_available,
        )

        major, minor = torch.cuda.get_device_capability(device)
        blackwell = major >= 10 and (major, minor) != (10, 3)
        if blackwell and is_flash_attention_v4_available():
            return 4
        elif blackwell:
            raise ValueError(
                f"Packed AuK DiT needs FlashAttention 4 on sm{major}{minor}, but "
                "SGLang's FA4 varlen kernel is unavailable (install flash-attn-4)"
            )
        elif _is_fa3_supported(device):
            return 3
        else:
            raise ValueError(
                f"Packed AuK DiT is unsupported on sm{major}{minor}: SGLang FA3 "
                "needs sm80-sm90 with CUDA >= 12.3"
            )


def probe_flash_attention(device: torch.device, *, heads: int, head_dim: int) -> None:
    """Run one tiny varlen call with the DiT's head shape.

    Missing kernels, JIT failures and unsupported head sizes then surface at
    stage construction rather than on the first multi-request batch.
    """
    mask = torch.ones(2, 4, dtype=torch.bool, device=device)
    layout = PackedLayout.build(mask, mask, prompt_width=0, target_width=4)
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
    # Lazy import: the default path and CPU model loading do not require Flash.
    from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func

    return flash_attn_varlen_func(
        q,
        k,
        v,
        layout.cu_seqlens,
        layout.cu_seqlens,
        layout.max_seqlen,
        layout.max_seqlen,
        causal=False,
        ver=layout.flash_version,
    )


def gather_rows(x, indices):
    return x.flatten(0, 1).index_select(0, indices)


def gather_rope(rope, indices, batch):
    """Packed rows of ``[T, D]`` (shared) or ``[B, T, D]`` (per-request) frequencies.

    The DiT's ``RotaryEmbedding`` has no xpos, so ``scale`` is a scalar and
    passes through unchanged.
    """
    freqs, scale = rope
    freqs = freqs.reshape(-1, *freqs.shape[-2:]).expand(batch, -1, -1)
    return gather_rows(freqs, indices).unsqueeze(0), scale


@dataclass
class PackedLayout:
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    audio_batch: torch.Tensor
    text_batch: torch.Tensor
    joint_indices: torch.Tensor
    # Permutations between separate packed streams and request-major streams.
    double_order: torch.Tensor
    double_inverse: torch.Tensor
    single_order: torch.Tensor
    single_batch: torch.Tensor
    target_rows: torch.Tensor
    target_indices: torch.Tensor
    target_batch: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    batch: int
    target_width: int
    flash_version: int

    @classmethod
    def build(cls, audio_mask, text_mask, prompt_width, target_width):
        """Build once per trajectory, outside the Euler loop and graph capture.

        Select the masks themselves, not a prefix inferred from their sums:
        text and reference tensors can contain stored padding and holes.
        """
        batch, audio_width = audio_mask.shape
        text_width = text_mask.shape[1]
        ai = audio_mask.flatten().nonzero().flatten()
        ci = text_mask.flatten().nonzero().flatten()
        ab = ai // audio_width
        cb = ci // text_width
        double_order = torch.argsort(torch.cat((ab, cb)), stable=True)
        single_order = torch.argsort(torch.cat((cb, ab)), stable=True)
        single_mask = torch.cat((text_mask, audio_mask), dim=1)
        ji = single_mask.flatten().nonzero().flatten()
        joint_width = text_width + audio_width
        single_batch = ji // joint_width
        local = ji % joint_width - text_width - prompt_width
        target_rows = (local >= 0).nonzero().flatten()
        target_batch = single_batch.index_select(0, target_rows)
        target_indices = target_batch * target_width + local.index_select(
            0, target_rows
        )
        lengths = single_mask.sum(1, dtype=torch.int32)
        cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
        return cls(
            ai,
            ci,
            ab,
            cb,
            ji,
            double_order,
            torch.argsort(double_order),
            single_order,
            single_batch,
            target_rows,
            target_indices,
            target_batch,
            cu,
            int(lengths.max().item()),
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
