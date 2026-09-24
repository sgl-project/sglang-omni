# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.utils.graph_padding import select_padded_graph


@pytest.mark.parametrize(
    ("rows", "expected"),
    [(2, ("8x16", 6)), (5, ("8x16", 3)), (8, ("16x16", 8)), (16, (None, 0))],
)
def test_select_padded_graph_uses_smallest_batch_then_capacity(
    rows: int, expected: tuple[str | None, int]
) -> None:
    graphs = {(8, 32): "8x32", (8, 16): "8x16", (16, 16): "16x16"}

    assert select_padded_graph(graphs, rows, 12) == expected


@pytest.mark.parametrize(
    ("rows", "extra", "expected"),
    [
        (4, None, (None, 0)),
        (3, {(8, 16): "gather"}, (None, 0)),
        (4, {(8, 16): "gather"}, ("gather", 4)),
        (5, {(8, 16): "gather"}, ("gather", 3)),
    ],
)
def test_select_padded_graph_replaces_positional_capture_with_bounded_gather(
    rows: int,
    extra: dict[tuple[int, int], str] | None,
    expected: tuple[str | None, int],
) -> None:
    graphs = {(8, 16): "positional"}

    assert (
        select_padded_graph(
            graphs, rows, 12, skip_batch=8, extra=extra, max_batch_ratio=2
        )
        == expected
    )


@pytest.mark.parametrize(
    ("rows", "capacity", "expected"),
    [
        (1, 12, (None, 0)),
        (2, 12, ("4x16", 2)),
        (3, 12, ("4x16", 1)),
        (4, 12, ("8x16", 4)),
        (5, 12, ("8x16", 3)),
        (4, 24, (None, 0)),
        (7, 24, (None, 0)),
        (8, 24, ("16x32", 8)),
        (8, 33, (None, 0)),
    ],
)
def test_select_padded_graph_bounds_total_batch_expansion(
    rows: int, capacity: int, expected: tuple[str | None, int]
) -> None:
    """The ratio cap covers both exact-boundary and context-driven expansion."""
    graphs = {(4, 16): "4x16", (8, 16): "8x16", (16, 32): "16x32"}

    assert select_padded_graph(graphs, rows, capacity, max_batch_ratio=2) == expected
