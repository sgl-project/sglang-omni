# SPDX-License-Identifier: Apache-2.0
"""CFG left-padding KV bookkeeping for LLaDA2-Uni."""

from __future__ import annotations

from itertools import accumulate

import pytest
import torch

from sglang_omni.models.llada2_uni.cfg_padding import build_cfg_kv_indices


def test_build_cfg_kv_indices_drops_left_padding() -> None:
    req_to_token = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
        ],
        dtype=torch.int32,
    )

    indices, kv_lens = build_cfg_kv_indices(
        req_to_token,
        req_pool_indices=[0, 1, 2],
        seq_lens=[5, 5, 5],
        left_pad_lens=[2, 0, 5],
    )

    assert kv_lens == [3, 5, 0]
    assert list(accumulate(kv_lens)) == [3, 8, 8]
    assert indices.tolist() == [2, 3, 4, 10, 11, 12, 13, 14]
    assert indices.dtype == torch.int64


def test_build_cfg_kv_indices_clamps_padding_to_sequence_length() -> None:
    req_to_token = torch.tensor([[7, 8, 9]], dtype=torch.int32)

    indices, kv_lens = build_cfg_kv_indices(
        req_to_token,
        req_pool_indices=[0],
        seq_lens=[3],
        left_pad_lens=[9],
    )

    assert kv_lens == [0]
    assert indices.numel() == 0


def test_build_cfg_kv_indices_requires_per_request_metadata() -> None:
    with pytest.raises(ValueError, match="per request"):
        build_cfg_kv_indices(
            torch.zeros((2, 4), dtype=torch.int32),
            req_pool_indices=[0, 1],
            seq_lens=[4],
            left_pad_lens=[1, 2],
        )
