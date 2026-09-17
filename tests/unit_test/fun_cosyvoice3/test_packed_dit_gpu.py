# SPDX-License-Identifier: Apache-2.0
"""GPU contract: the ragged FA3 read matches padded SDPA on packed rows."""

from __future__ import annotations

import pytest
import torch
from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    RaggedRowAttention,
    RowAttention,
    pack_rows,
)

pytestmark = pytest.mark.accelerator

LENGTHS = (11, 4, 19, 7)
CHUNK = 4
HEADS = 2
HEAD_DIM = 64


@pytest.mark.parametrize("chunk_size", [CHUNK, None])
def test_the_ragged_read_matches_the_padded_read(chunk_size: int | None) -> None:
    device = torch.device("cuda")
    if not _is_fa3_supported():
        pytest.skip("FA3 is unavailable on this device")
    torch.manual_seed(2)
    rows = pack_rows(LENGTHS, device)
    query, key, value = (
        torch.randn(
            1, rows.total, HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        for _ in range(3)
    )

    ragged = RaggedRowAttention(
        rows, chunk_size=chunk_size, heads=HEADS, head_dim=HEAD_DIM
    )(query, key, value)
    padded = RowAttention(rows, chunk_size=chunk_size, heads=HEADS)(query, key, value)

    torch.testing.assert_close(ragged, padded, rtol=2e-2, atol=2e-2)
