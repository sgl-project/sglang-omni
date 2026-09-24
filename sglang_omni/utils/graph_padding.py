# SPDX-License-Identifier: Apache-2.0
"""Select padded auxiliary graphs; callers isolate filler state and trim outputs."""

from __future__ import annotations

from typing import TypeVar

GraphT = TypeVar("GraphT")


def select_padded_graph(
    graphs: dict[tuple[int, int], GraphT],
    rows: int,
    capacity: int,
    *,
    skip_batch: int | None = None,
    extra: dict[tuple[int, int], GraphT] | None = None,
    max_batch_ratio: float | None = None,
) -> tuple[GraphT | None, int]:
    """Return the smallest compatible graph and filler count, or (None, 0).

    Skip positional captures at skip_batch; extra supplies gather-mode twins.
    """
    pool = [
        (batch_size, bucket_capacity, graphs)
        for batch_size, bucket_capacity in graphs
        if batch_size > rows
        and bucket_capacity >= capacity
        and batch_size != skip_batch
    ]
    if extra:
        pool += [
            (batch_size, bucket_capacity, extra)
            for batch_size, bucket_capacity in extra
            if batch_size > rows and bucket_capacity >= capacity
        ]
    if not pool:
        return None, 0
    batch_size, bucket_capacity, source = min(pool, key=lambda item: (item[0], item[1]))
    if max_batch_ratio is not None and batch_size > rows * max_batch_ratio:
        return None, 0
    return source[(batch_size, bucket_capacity)], batch_size - rows
