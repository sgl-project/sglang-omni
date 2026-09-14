# SPDX-License-Identifier: Apache-2.0
"""Request-scoped layouts for AuK's non-causal packed transformer trunk."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
    freqs, scale = rope
    if freqs.ndim == 2:
        freqs = freqs.unsqueeze(0)
    freqs = freqs.expand(batch, -1, -1)
    if isinstance(scale, torch.Tensor) and scale.ndim >= 2:
        if scale.ndim == 2:
            scale = scale.unsqueeze(0)
        scale = scale.expand(batch, -1, -1)
        scale = gather_rows(scale, indices).unsqueeze(0)
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
            (
                4
                if audio_mask.is_cuda
                and torch.cuda.get_device_capability(audio_mask.device)[0] >= 10
                else 3
            ),
        )

    def unpack_target(self, x):
        output = x.new_zeros(self.batch * self.target_width, x.shape[-1])
        output.index_copy_(0, self.target_indices, x)
        return output.view(self.batch, self.target_width, x.shape[-1])
