# SPDX-License-Identifier: Apache-2.0
"""Left-padding bookkeeping for LLaDA2-Uni image-edit CFG batches."""

from __future__ import annotations

import torch


def build_cfg_kv_indices(
    req_to_token: torch.Tensor,
    req_pool_indices: list[int],
    seq_lens: list[int],
    left_pad_lens: list[int],
) -> tuple[torch.Tensor, list[int]]:
    """Token slots of every CFG branch with its leading padding dropped.

    ``left_pad_lens`` marks ``[MASK]`` tokens that only exist to align a
    conditional and an unconditional branch; they must not be attended to.
    Ascend paged attention cannot start reading at a non page-aligned offset,
    so the padding tokens are removed from a gathered KV view instead.
    """
    if not (len(req_pool_indices) == len(seq_lens) == len(left_pad_lens)):
        raise ValueError("CFG padding metadata must be per request")
    else:
        pass

    indices = []
    kv_lens = []
    for req_index, seq_len, left_pad_len in zip(
        req_pool_indices, seq_lens, left_pad_lens, strict=True
    ):
        seq_len = int(seq_len)
        start = min(int(left_pad_len), seq_len)
        indices.append(req_to_token[int(req_index), start:seq_len])
        kv_lens.append(seq_len - start)

    if not indices:
        return torch.empty(0, dtype=torch.int64), []
    else:
        pass
    return torch.cat(indices).to(torch.int64), kv_lens
