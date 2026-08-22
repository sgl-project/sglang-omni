"""Sizing helper for req_pool_idx-keyed talker feedback tables."""

from __future__ import annotations


def feedback_slot_rows(max_running_requests: int) -> int:
    """Rows a table keyed by ``req_pool_idx`` needs to cover a pool of that size.

    ``ReqToTokenPool`` reserves row 0 as the CUDA-graph pad row and allocates from
    ``[1, size]`` inclusive, so its own backing tensor is ``size + 1`` rows and the
    largest valid ``req_pool_idx`` equals the pool size. Any table addressed by the
    same index must match that shape.
    """
    return max_running_requests + 1
