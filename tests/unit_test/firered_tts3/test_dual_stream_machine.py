# SPDX-License-Identifier: Apache-2.0
"""Dual-stream token order, patch boundaries, and stop timing on real CPU flow."""

from __future__ import annotations

import pytest
import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from tests.unit_test.fixtures.firered_tts3 import (
    SmallFlowModel,
    make_flow_request,
    make_flow_result,
    make_flow_runner,
)


@pytest.mark.parametrize(
    "ratio,expected_tokens,patch_counts",
    [
        ((2, 1), [101, 102, 0, 103, 151671, 0, 151671], [0, 0, 1, 1, 1, 2, 2]),
        (
            (2, 3),
            [101, 102, 0, 0, 0, 103, 151671, 0, 0, 0, 151671],
            [0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 6],
        ),
    ],
)
def test_group_schedule_and_text_feeds(
    ratio: tuple[int, int], expected_tokens: list[int], patch_counts: list[int]
) -> None:
    runner = make_flow_runner(SmallFlowModel())
    request = make_flow_request(dual_stream=True, ratio=ratio)
    result = make_flow_result(torch.zeros(3, 8), [123])
    runner.post_prefill(result, None, None, [request])
    actual_tokens = [int(result.next_token_ids[0])]
    actual_counts = [len(request.data.latent_patches)]
    for _ in expected_tokens[1:]:
        result = make_flow_result(torch.ones(1, 8))
        runner.post_decode(result, None, None, [request])
        actual_tokens.append(int(result.next_token_ids[0]))
        actual_counts.append(len(request.data.latent_patches))
    assert actual_tokens == expected_tokens
    assert actual_counts == patch_counts
    assert len(request.data.pending_feedback_queue) == patch_counts[-1]
    assert request.data.req.finished_reason is None


def test_stop_fires_after_the_patch_feed() -> None:
    """The stopping patch must be emitted before the group's final feed decides to stop."""
    runner = make_flow_runner(SmallFlowModel())
    request = make_flow_request(dual_stream=True)
    request.data.state.min_patches = 2
    request.data.state.stop_threshold = 0.0
    runner.post_prefill(
        make_flow_result(torch.zeros(3, 8), [123]), None, None, [request]
    )
    patch_counts = []
    for _ in range(5):
        runner.post_decode(make_flow_result(torch.ones(1, 8)), None, None, [request])
        patch_counts.append(len(request.data.latent_patches))
        assert request.data.req.finished_reason is None
    runner.post_decode(make_flow_result(torch.ones(1, 8)), None, None, [request])
    assert patch_counts == [0, 1, 1, 1, 2]
    assert isinstance(request.data.req.finished_reason, FINISH_MATCHED_TOKEN)
    assert len(request.data.latent_patches) == 2
