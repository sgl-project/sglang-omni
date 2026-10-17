# SPDX-License-Identifier: Apache-2.0
"""Prompt capture coverage and ownership across prefill chunks."""

import pytest
import torch

from sglang_omni.models.qwen3_omni.prompt_hidden import PromptHiddenCapture


def test_prompt_capture_collects_absolute_positions_across_chunks() -> None:
    capture = PromptHiddenCapture(positions=torch.tensor([3, 7]))
    rows = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    for start, end in [(0, 3), (3, 4), (4, 7), (7, 8)]:
        capture.add_extend(prefix_length=start, hidden_states=rows[start:end])
    result = capture.take_complete()
    torch.testing.assert_close(result.hidden_states, rows[[3, 7]])
    assert result.positions.tolist() == [3, 7]
    assert not capture.needs_capture
    assert capture.take_complete() is None


def test_prompt_capture_owns_rows_and_ignores_replayed_chunks() -> None:
    capture = PromptHiddenCapture(positions=torch.tensor([1, 3]))
    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    capture.add_extend(prefix_length=0, hidden_states=first)
    first.fill_(-1)
    capture.add_extend(prefix_length=0, hidden_states=first)
    last = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    capture.add_extend(prefix_length=2, hidden_states=last)
    result = capture.take_complete()
    last.fill_(-2)
    capture.add_extend(prefix_length=0, hidden_states=torch.zeros((4, 2)))
    torch.testing.assert_close(
        result.hidden_states, torch.tensor([[3.0, 4.0], [7.0, 8.0]])
    )


def test_prompt_capture_keeps_single_row_rank_and_accepts_empty_chunks() -> None:
    capture = PromptHiddenCapture(positions=torch.tensor([2]))
    capture.add_extend(prefix_length=0, hidden_states=torch.empty((0, 4)))
    assert capture.needs_capture
    capture.add_extend(
        prefix_length=2, hidden_states=torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    )
    result = capture.take_complete()
    assert result.hidden_states.shape == (1, 4)
    torch.testing.assert_close(
        result.hidden_states, torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    )


@pytest.mark.parametrize(
    ("prefix_length", "hidden_states", "message"),
    [
        (-1, torch.ones((1, 2)), "nonnegative prefix"),
        (3, torch.ones(2), "2D states"),
        (3, torch.ones((1, 1)), "width and dtype"),
        (3, torch.ones((1, 2), dtype=torch.float64), "width and dtype"),
    ],
    ids=["negative-prefix", "rank", "width", "dtype"],
)
def test_prompt_capture_rejects_invalid_chunks_without_consuming_rows(
    prefix_length: int, hidden_states: torch.Tensor, message: str
) -> None:
    capture = PromptHiddenCapture(positions=torch.tensor([1, 3]))
    capture.add_extend(prefix_length=1, hidden_states=torch.ones((1, 2)))
    with pytest.raises(ValueError, match=message):
        capture.add_extend(prefix_length=prefix_length, hidden_states=hidden_states)
    assert capture.needs_capture
    capture.add_extend(prefix_length=3, hidden_states=torch.full((1, 2), 2.0))
    torch.testing.assert_close(
        capture.take_complete().hidden_states, torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    )


def test_prompt_captures_do_not_share_request_storage() -> None:
    positions = torch.tensor([1])
    first = PromptHiddenCapture(positions=positions)
    second = PromptHiddenCapture(positions=positions)
    positions.fill_(0)
    first.add_extend(prefix_length=1, hidden_states=torch.ones((1, 2)))
    with pytest.raises(RuntimeError, match="Missing multimodal prompt"):
        second.take_complete()
    second.add_extend(prefix_length=1, hidden_states=torch.full((1, 2), 2.0))
    first_result = first.take_complete()
    second_result = second.take_complete()
    torch.testing.assert_close(first_result.hidden_states, torch.ones((1, 2)))
    torch.testing.assert_close(second_result.hidden_states, torch.full((1, 2), 2.0))
    assert first_result.positions.tolist() == [1]


def test_handoff_retains_immutable_capture_for_later_cache_insertion() -> None:
    capture = PromptHiddenCapture(positions=torch.tensor([1]))
    capture.add_extend(prefix_length=1, hidden_states=torch.tensor([[2.0, 3.0]]))
    result = capture.take_complete()
    result.positions.fill_(0)
    result.hidden_states.fill_(-1)
    assert capture.positions.tolist() == [1]
    torch.testing.assert_close(capture.hidden_states, torch.tensor([[2.0, 3.0]]))
    assert capture.take_complete() is None


def test_sparse_restore_copies_only_required_missing_rows() -> None:
    capture = PromptHiddenCapture(positions=torch.tensor([1, 3]))
    capture.add_extend(prefix_length=1, hidden_states=torch.tensor([[2.0, 3.0]]))
    rows = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    capture.add_rows(positions=torch.arange(5), hidden_states=rows)
    rows.fill_(-1)
    torch.testing.assert_close(
        capture.take_complete().hidden_states, torch.tensor([[2.0, 3.0], [6.0, 7.0]])
    )
