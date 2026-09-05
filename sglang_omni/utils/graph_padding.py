# SPDX-License-Identifier: Apache-2.0
"""Select padded aux graphs and extend their per-row inputs.

Callers must isolate filler writes from live state and slice outputs back to
the real row count. Positional captures require gather-mode replacements.
"""

from __future__ import annotations

from typing import TypeVar

import torch

GraphT = TypeVar("GraphT")


def select_padded_graph(
    graphs: dict[tuple[int, int], GraphT],
    rows: int,
    capacity: int,
    *,
    skip_batch: int | None = None,
    extra: dict[tuple[int, int], GraphT] | None = None,
) -> tuple[GraphT | None, int]:
    """Pick the smallest captured graph a ``rows``-row batch can pad up to.

    Candidates need ``batch_size > rows`` and ``bucket_capacity >= capacity``;
    ties resolve to the smallest batch then the smallest capacity. Entries in
    ``graphs`` whose batch equals ``skip_batch`` are ignored (captures that
    read state positionally instead of via the slot-index input buffer);
    ``extra`` supplies gather-mode replacements for such batches. Returns
    ``(graph, filler_row_count)`` or ``(None, 0)``.
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
    return source[(batch_size, bucket_capacity)], batch_size - rows


def pad_rows(
    tensor: torch.Tensor,
    pad: int,
    *,
    fill_value: int | float | None = None,
) -> torch.Tensor:
    """Append ``pad`` filler rows to a per-row input tensor.

    Filler rows are zeros unless ``fill_value`` is given (e.g. the sacrificial
    slot index for the slot-id tensor). The result matches the captured input
    buffer's batch dimension, so a plain ``copy_`` stages it for replay.
    """
    if pad <= 0:
        return tensor
    shape = (pad, *tensor.shape[1:])
    filler = (
        tensor.new_zeros(shape)
        if fill_value is None
        else tensor.new_full(shape, fill_value)
    )
    return torch.cat([tensor, filler])
