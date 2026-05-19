# SPDX-License-Identifier: Apache-2.0
"""Stage 1 acceptance tests for the CUDA Graph migration.

Locks in the behavior that the new ``HiggsBatchedSamplerState`` pool is
a drop-in for the old per-request ``HiggsSamplerState`` dict:

- view_row / write_row round-trip preserves all fields.
- Multiple rows evolve independently.
- A reset row matches a freshly constructed state.

Stage 2 will add bit-identical-parity tests between the per-row
``step`` function and the new batched step.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.sampler import (
    HiggsBatchedSamplerState,
    HiggsSamplerState,
)


@pytest.fixture
def pool() -> HiggsBatchedSamplerState:
    if not torch.cuda.is_available():
        return HiggsBatchedSamplerState(
            max_batch_size=4, num_codebooks=8, device="cpu"
        )
    return HiggsBatchedSamplerState(
        max_batch_size=4, num_codebooks=8, device="cuda"
    )


def test_fresh_pool_view_matches_fresh_state(pool):
    view = pool.view_row(0)
    assert view.num_codebooks == 8
    assert view.delay_count == 0
    assert view.eoc_countdown is None
    assert view.generation_done is False
    # last_codes is None *only* until the first sample lands; the pool
    # exposes the tensor when delay_count > 0. At delay_count == 0 the
    # view should report None so the model_runner falls back to text embed.
    assert view.last_codes is None


def test_write_row_persists_to_pool(pool):
    state = HiggsSamplerState(num_codebooks=8)
    state.delay_count = 5
    state.eoc_countdown = 3
    state.generation_done = False
    state.last_codes = torch.arange(8, dtype=torch.long, device=pool.device)
    pool.write_row(1, state)

    assert int(pool.delay_count[1].item()) == 5
    assert int(pool.eoc_countdown[1].item()) == 3
    assert bool(pool.generation_done[1].item()) is False
    assert torch.equal(
        pool.last_codes[1], torch.arange(8, dtype=torch.long, device=pool.device)
    )


def test_view_then_write_round_trip(pool):
    pool.delay_count[2] = 9
    pool.eoc_countdown[2] = 4
    pool.generation_done[2] = True
    pool.last_codes[2].copy_(torch.arange(8, 16, dtype=torch.long, device=pool.device))

    snapshot = pool.view_row(2)
    assert snapshot.delay_count == 9
    assert snapshot.eoc_countdown == 4
    assert snapshot.generation_done is True
    assert torch.equal(
        snapshot.last_codes,
        torch.arange(8, 16, dtype=torch.long, device=pool.device),
    )

    # Mutating the snapshot doesn't accidentally write to the pool (the
    # snapshot fields are Python scalars, the tensor *is* aliased — that's
    # fine; we explicitly writeback below).
    snapshot.delay_count = 10
    snapshot.eoc_countdown = None
    snapshot.generation_done = False
    pool.write_row(2, snapshot)

    assert int(pool.delay_count[2].item()) == 10
    assert int(pool.eoc_countdown[2].item()) == -1  # None → -1 sentinel
    assert bool(pool.generation_done[2].item()) is False


def test_reset_row_clears_all_fields(pool):
    pool.delay_count[3] = 7
    pool.eoc_countdown[3] = 2
    pool.generation_done[3] = True
    pool.last_codes[3].copy_(torch.arange(8, dtype=torch.long, device=pool.device))

    pool.reset_row(3)

    assert int(pool.delay_count[3].item()) == 0
    assert int(pool.eoc_countdown[3].item()) == -1
    assert bool(pool.generation_done[3].item()) is False
    assert torch.equal(
        pool.last_codes[3], torch.zeros(8, dtype=torch.long, device=pool.device)
    )


def test_rows_are_independent(pool):
    """Writing to one row must not bleed into another."""
    s0 = HiggsSamplerState(num_codebooks=8, delay_count=3)
    s1 = HiggsSamplerState(num_codebooks=8, delay_count=5, eoc_countdown=2)
    pool.write_row(0, s0)
    pool.write_row(1, s1)

    assert int(pool.delay_count[0].item()) == 3
    assert int(pool.delay_count[1].item()) == 5
    assert int(pool.eoc_countdown[0].item()) == -1
    assert int(pool.eoc_countdown[1].item()) == 2


def test_eoc_countdown_none_round_trips_through_minus_one(pool):
    """``eoc_countdown=None`` stores as -1 in tensor and reads back as None."""
    state = HiggsSamplerState(num_codebooks=8, eoc_countdown=None)
    pool.write_row(0, state)
    assert int(pool.eoc_countdown[0].item()) == -1
    snap = pool.view_row(0)
    assert snap.eoc_countdown is None
