# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.mark.accelerator
@pytest.mark.parametrize(("num_slots", "tokens"), [(8, 0), (12, 5), (24, 13)])
def test_kv_copies_preserve_live_rows_and_skip_dummy_storage(
    num_slots: int, tokens: int
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton KV copies")

    from sglang_omni.models.dots_tts.tail_kv import gather_kv, scatter_kv

    torch.manual_seed(0)
    pool_shape = (2, num_slots, 3, 23, 11)
    pool_backing = [
        torch.randn(pool_shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    pools = [backing[..., 1:20, 1:8] for backing in pool_backing]
    out_shape = (2, 6, 3, tokens + 2, 9)
    out_backing = [
        torch.full(out_shape, value, device="cuda", dtype=torch.bfloat16)
        for value in (123, -123)
    ]
    outputs = [backing[..., 1 : 1 + tokens, 1:8] for backing in out_backing]
    expected_outputs = [backing.clone() for backing in out_backing]
    expected_pools = [backing.clone() for backing in pool_backing]
    slot_ids = [num_slots - 1, num_slots, 1, num_slots, 0, num_slots // 2]
    start_ids = [0, 10_000, 3, -10_000, 4, 5]
    slots = torch.tensor(slot_ids, device="cuda", dtype=torch.long)
    starts = torch.tensor(start_ids, device="cuda", dtype=torch.long)

    # note (0xtoward): Strided views and repeated dummy IDs exercise masked
    # accesses; checking the backing tensors also catches writes outside views.
    gather_kv(*pools, slots, *outputs)
    for pool, expected in zip(pools, expected_outputs):
        view = expected[..., 1 : 1 + tokens, 1:8]
        for row, slot in enumerate(slot_ids):
            if slot < num_slots:
                view[:, row].copy_(pool[:, slot, :, :tokens])
            else:
                view[:, row].zero_()
    for actual, expected in zip(out_backing, expected_outputs):
        assert torch.equal(actual, expected)

    scatter_kv(*outputs, slots, starts, *pools)
    for output, expected in zip(outputs, expected_pools):
        view = expected[..., 1:20, 1:8]
        for row, (slot, start) in enumerate(zip(slot_ids, start_ids)):
            if slot < num_slots:
                view[:, slot, :, start : start + tokens].copy_(output[:, row])
    for actual, expected in zip(pool_backing, expected_pools):
        assert torch.equal(actual, expected)
