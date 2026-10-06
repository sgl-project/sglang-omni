# SPDX-License-Identifier: Apache-2.0
"""Cached block attention matches masked SDPA and stores the promoted rows in place."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.dots_tts.tail import batched_causal_update_mask
from sglang_omni.models.dots_tts.tail_attention import cached_block_attention
from tests.unit_test.fixtures.accelerator import require_cuda

HEADS = 4
HEAD_DIM = 64


@pytest.mark.accelerator
@pytest.mark.parametrize(("query_rows", "previous_rows"), [(10, 5), (2, 2)])
@torch.no_grad()
def test_cached_block_attention_matches_masked_sdpa(
    query_rows: int, previous_rows: int
) -> None:
    require_cuda()
    generator = torch.Generator(device="cuda").manual_seed(3)
    slots_total, tokens = 6, 200
    dtype = torch.bfloat16

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    key_pool, value_pool = randn(slots_total, HEADS, tokens, HEAD_DIM), randn(
        slots_total, HEADS, tokens, HEAD_DIM
    )
    slots = torch.tensor([4, 1, 5, slots_total], device="cuda")
    valid = torch.tensor([0, 37, 130, 0], device="cuda")
    query, key, value = (randn(4, HEADS, query_rows, HEAD_DIM) for _ in range(3))
    original_keys, original_values = key_pool.clone(), value_pool.clone()

    output = cached_block_attention(
        query, key, value, key_pool, value_pool, slots, valid, previous_rows
    )

    capacity = 160
    gathered_keys = torch.zeros(
        4, HEADS, capacity + query_rows, HEAD_DIM, device="cuda"
    )
    gathered_values = torch.zeros_like(gathered_keys)
    for row, slot in enumerate(slots.tolist()):
        if slot < slots_total:
            gathered_keys[row, :, :capacity] = original_keys[slot, :, :capacity].float()
            gathered_values[row, :, :capacity] = original_values[
                slot, :, :capacity
            ].float()
        else:
            pass
    gathered_keys[:, :, capacity:] = key.float()
    gathered_values[:, :, capacity:] = value.float()
    mask = batched_causal_update_mask(
        capacity_tokens=capacity,
        valid_persistent=valid,
        prev_len=previous_rows,
        current_len=query_rows - previous_rows,
    )
    expected = F.scaled_dot_product_attention(
        query.float(), gathered_keys, gathered_values, attn_mask=mask
    ).transpose(1, 2)
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)

    for row, slot in enumerate(slots.tolist()):
        if slot < slots_total:
            start = int(valid[row])
            stored = slice(start, start + previous_rows)
            torch.testing.assert_close(
                key_pool[slot, :, stored], key[row, :, :previous_rows]
            )
            torch.testing.assert_close(
                value_pool[slot, :, stored], value[row, :, :previous_rows]
            )
            original_keys[slot, :, stored] = key[row, :, :previous_rows]
            original_values[slot, :, stored] = value[row, :, :previous_rows]
        else:
            pass
    assert torch.equal(key_pool, original_keys)
    assert torch.equal(value_pool, original_values)
