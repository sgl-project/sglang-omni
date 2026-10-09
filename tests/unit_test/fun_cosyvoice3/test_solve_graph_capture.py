# SPDX-License-Identifier: Apache-2.0
"""A step's rows replayed in consecutive groups that fit the largest graph tier."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import replay_in_groups


@pytest.mark.parametrize(
    ("lengths", "max_frames", "max_rows", "groups"),
    [
        ((300, 200), 512, 4, [(slice(0, 2), slice(0, 500))]),
        ((512,), 512, 4, [(slice(0, 1), slice(0, 512))]),
        (
            (300, 300, 100),
            512,
            4,
            [(slice(0, 1), slice(0, 300)), (slice(1, 3), slice(300, 700))],
        ),
        (
            (10, 10, 10, 10, 10),
            512,
            2,
            [
                (slice(0, 2), slice(0, 20)),
                (slice(2, 4), slice(20, 40)),
                (slice(4, 5), slice(40, 50)),
            ],
        ),
        (
            (400, 112, 1),
            512,
            4,
            [(slice(0, 2), slice(0, 512)), (slice(2, 3), slice(512, 513))],
        ),
    ],
)
def test_replay_in_groups_keeps_row_order_within_the_frame_and_row_limits(
    lengths: tuple[int, ...],
    max_frames: int,
    max_rows: int,
    groups: list[tuple[slice, slice]],
) -> None:
    replayed: list[tuple[slice, slice]] = []

    def replay(rows: slice, frames: slice) -> torch.Tensor:
        replayed.append((rows, frames))
        return torch.arange(frames.start, frames.stop).reshape(1, -1, 1)

    generated = replay_in_groups(
        replay, lengths, max_frames=max_frames, max_rows=max_rows
    )

    assert replayed == groups
    assert torch.equal(generated, torch.arange(sum(lengths)).reshape(1, -1, 1))
