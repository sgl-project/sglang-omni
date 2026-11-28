# SPDX-License-Identifier: Apache-2.0
"""Neutralize padded decode rows in the Ascend gated delta net backend.

A decode graph replays a fixed batch bucket, so a batch smaller than the bucket
carries padding rows.  Those rows hold no prompt text: their sequence length is
the graph fill value, and their mamba slot points at the reserved padding slot.
On this backend the padded rows still reach the fused convolution and delta-rule
kernels with leftover values, and those values leak into the rows of the real
requests, which then diverge from the eager decode and eventually generate
runaway audio.  Zeroing the padded rows' latent inputs keeps the kernels'
per-row contract intact while leaving every real row bit-identical.

Only Zonos3 installs this backend, so other models keep the stock one.
"""

from __future__ import annotations

from typing import Optional

import torch
from sglang.srt.hardware_backend.npu.attention import ascend_gdn_backend


class Zonos3AscendGDNAttnBackend(ascend_gdn_backend.AscendGDNAttnBackend):
    """Ascend gated delta net decode that keeps padded rows out of the real ones."""

    def _padding_mask(self, bs: int) -> torch.Tensor:  # noqa: leading-underscore
        mask = getattr(self, "_z3_pad_mask", None)  # noqa: leading-underscore
        if mask is None:
            mask = {}
            self._z3_pad_mask = mask  # noqa: leading-underscore
        else:
            pass
        row_mask = mask.get(bs)
        if row_mask is None:
            row_mask = torch.zeros((bs, 1), dtype=torch.bool, device=self.device)
            mask[bs] = row_mask
        else:
            pass
        return row_mask

    def _replay_metadata(  # noqa: leading-underscore
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
        num_padding: Optional[int] = None,
        *args,
        **kwargs,
    ):
        # The base recomputes num_padding from seq_lens_cpu; mirror it so the
        # mask matches the metadata the kernels are about to read.
        if seq_lens_cpu is None:
            num_padding = 0
        else:
            num_padding = int(
                torch.count_nonzero(
                    seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
                )
            )
        out = super()._replay_metadata(  # noqa: leading-underscore
            bs,
            req_pool_indices,
            forward_mode,
            spec_info,
            seq_lens_cpu,
            num_padding=num_padding,
            *args,
            **kwargs,
        )
        row_mask = self._padding_mask(bs)  # noqa: leading-underscore
        if num_padding:
            row_mask[: bs - num_padding] = False
            row_mask[bs - num_padding :] = True
        else:
            row_mask.fill_(False)
        return out

    def forward_decode(
        self,
        layer,
        forward_batch,
        mixed_qkv,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        row_mask = getattr(self, "_z3_pad_mask", {}).get(
            forward_batch.batch_size
        )  # noqa: leading-underscore
        if row_mask is not None and isinstance(mixed_qkv, torch.Tensor):
            # Tensor ops, not host branches: they stay inside the captured graph
            # and pick up the mask the replay metadata refreshes.
            mixed_qkv = mixed_qkv.masked_fill(row_mask, 0)
            a = a.masked_fill(row_mask, 0)
            b = b.masked_fill(row_mask, 0)
        else:
            pass
        return super().forward_decode(layer, forward_batch, mixed_qkv, a, b, **kwargs)


def install_gdn_pad_mask() -> None:
    """Make SGLang build Zonos3AscendGDNAttnBackend for the gated delta net layers."""
    ascend_gdn_backend.AscendGDNAttnBackend = Zonos3AscendGDNAttnBackend
