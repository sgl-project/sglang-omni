# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.codec_state_pool import (
    CodecStatePool,
    CodecStateSpec,
    ring_addresses,
)


def make_pool(num_slots: int, capture_slots: int) -> CodecStatePool:
    spec = CodecStateSpec(
        conv_histories=(("a", 3, 6), ("b", 2, 0)),
        transconv_overlaps=(("c", 4, 2),),
        num_layers=2,
        num_key_value_heads=2,
        head_dim=4,
        retained_context=5,
    )
    return CodecStatePool(
        spec,
        num_slots=num_slots,
        capture_slots=capture_slots,
        ring_frames=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_a_full_pool_hands_out_no_slot_until_one_is_released() -> None:
    pool = make_pool(num_slots=2, capture_slots=1)

    taken = {pool.acquire(), pool.acquire()}
    assert taken == {0, 1}
    assert pool.acquire() is None
    pool.release(1)
    assert pool.acquire() == 1


def test_zeroing_capture_slots_leaves_every_other_slot() -> None:
    pool = make_pool(num_slots=3, capture_slots=2)
    tensors = [
        *pool.histories.values(),
        *pool.overlaps.values(),
        *pool.keys,
        *pool.values,
        pool.frame_positions,
    ]
    for tensor in tensors:
        tensor.fill_(1)

    pool.zero_slots(torch.tensor(pool.scratch_slots))

    assert set(pool.histories) == {"a"}
    assert pool.scratch_slots == (3, 4)
    for tensor in [*pool.histories.values(), *pool.overlaps.values()]:
        assert not tensor[3:].any() and tensor[:3].all()
    for ring in [*pool.keys, *pool.values]:
        assert not ring[3 * 8 :].any() and ring[: 3 * 8].all()
    assert pool.frame_positions.tolist() == [1, 1, 1, 0, 0]


@pytest.mark.parametrize("start_position", [0, 3, 71, 75, 81, 200])
def test_ring_addresses_list_each_window_in_position_order_on_distinct_rows(
    start_position: int,
) -> None:
    ring_frames, retained_frames, new_frames = 81, 71, 10
    slot_index = torch.tensor([2, 5])
    addresses = ring_addresses(
        slot_index,
        torch.tensor([start_position, 0]),
        new_frames,
        ring_frames,
        retained_frames,
    )

    for row, start in enumerate((start_position, 0)):
        first = max(0, start - retained_frames)
        positions = list(range(first, start + new_frames))
        expected = [
            int(slot_index[row]) * ring_frames + position % ring_frames
            for position in positions
        ]
        assert int(addresses.window_lengths[row]) == len(positions)
        window = addresses.window_rows[row, : len(positions)].tolist()
        assert window == expected
        assert len(set(window)) == len(window)
        assert (
            addresses.write_rows[row * new_frames : (row + 1) * new_frames].tolist()
            == expected[-new_frames:]
        )
        assert addresses.frame_positions[
            row * new_frames : (row + 1) * new_frames
        ].tolist() == list(range(start, start + new_frames))
