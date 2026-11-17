# SPDX-License-Identifier: Apache-2.0
"""CPU checks for AuK SeaCache filtering and trajectory decisions."""

import pytest
import torch

from sglang_omni.models.auk.seacache import (
    SeaCacheConfig,
    SeaCacheState,
    filter_audio,
    relative_l1,
)


@pytest.mark.parametrize("frames", [7, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_filter_preserves_valid_audio(frames, dtype):
    hidden = torch.randn(2, frames, 4, dtype=dtype)
    lengths = torch.tensor([frames, frames - 2])
    filtered = filter_audio(hidden, torch.tensor(0.25), lengths)
    assert filtered.shape == hidden.shape
    assert filtered.dtype == torch.float32
    assert torch.isfinite(filtered).all()
    hidden[1, frames - 2 :] = 1000
    changed = filter_audio(hidden, torch.tensor(0.25), lengths)
    torch.testing.assert_close(filtered, changed)


def test_masked_distance_and_cfg_batch_decision():
    previous = torch.ones(4, 8, 2)
    current = previous.clone()
    current[0, 5:] = 100
    current[3, :5] = 2
    lengths = torch.tensor([5, 8, 8, 5])
    torch.testing.assert_close(
        relative_l1(current, previous, lengths), torch.tensor([0.0, 0.0, 0.0, 1.0])
    )

    state = SeaCacheState(config=SeaCacheConfig(threshold=0.1), total_steps=3)
    assert state.decide(previous, torch.tensor(0.0), lengths)[0]
    state.previous_residual = torch.zeros(4, 10, 2)
    should_compute, reason = state.decide(current, torch.tensor(0.5), lengths)
    assert should_compute and reason == "threshold"
    assert state.decide(current, torch.tensor(1.0), lengths)[0]
    assert (state.computed_steps, state.cached_steps) == (3, 0)


def test_skip_limit_boundaries_and_trajectory_isolation():
    config = SeaCacheConfig(threshold=100.0, max_skip_steps=1)
    hidden = torch.ones(2, 5, 4)
    lengths = torch.tensor([5, 5])
    first = SeaCacheState(config=config, total_steps=5)
    second = SeaCacheState(config=config, total_steps=5)
    for step in range(5):
        compute, reason = first.decide(hidden, torch.tensor(step / 5), lengths)
        if compute:
            first.previous_residual = torch.ones(2, 6, 4)
        assert (
            reason
            == ["boundary", "cache_hit", "max_skip", "cache_hit", "boundary"][step]
        )
    assert (first.computed_steps, first.cached_steps) == (3, 2)
    assert second.step == 0 and second.previous_residual is None


def test_zero_threshold_computes_every_step():
    state = SeaCacheState(config=SeaCacheConfig(threshold=0), total_steps=4)
    for step in range(4):
        assert state.decide(
            torch.ones(1, 5, 3), torch.tensor(step / 4), torch.tensor([5])
        )[0]
    assert (state.computed_steps, state.cached_steps) == (4, 0)


def test_flow_time_endpoints():
    # AuK integrates from pure noise at t=0 toward data at t=1.
    data, noise = torch.tensor([7.0]), torch.tensor([-3.0])
    for time, expected in [(0.0, noise), (1.0, data)]:
        mixed = time * data + (1 - time) * noise
        torch.testing.assert_close(mixed, expected)
