# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for Diffusion-LLM scheduler token-id handling.

Kept free of any ``sglang`` import so it can be unit-tested in a CPU-only,
sglang-free environment.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def split_token_ids_for_batch(reqs: list[Any], token_ids: Any) -> list[list[int]]:
    """Normalize a model's ``next_token_ids`` into a per-request list of lists.

    Diffusion-LLM stages may return token ids in one of two shapes depending
    on whether the batch was scheduled with a single request or concurrently:

    * a **flat** ``list[int]`` (or ``Tensor`` already .tolist()'d) — the model
      collapsed all requests into one stream. This is the only correct shape
      when ``len(reqs) == 1`` and is wrapped as ``[token_ids]``.
    * a **nested** ``list[list[int]]`` — one inner list per request, the
      standard sglang batch shape when ``prefill_max_requests > 1``. Returned
      as-is.

    Safety: if ``len(reqs) > 1`` but a flat ``list[int]`` is returned (an
    unexpected shape for concurrent scheduling), we refuse to silently pair
    each request with a single int (which would corrupt every output). Instead
    we fall back to splitting the flat list round-robin across requests and log
    a warning, so the scheduler keeps running instead of producing garbage.
    """
    n = len(reqs)
    # Single-request batch: flat list is the expected, documented shape.
    if n == 1:
        if token_ids and isinstance(token_ids[0], list):
            # Already one inner list for the single request; keep it.
            return [list(token_ids[0])]
        return [list(token_ids)]
    # Multi-request batch: expect one inner list per request.
    if token_ids and not isinstance(token_ids[0], int):
        return [list(t) for t in token_ids]
    # Unexpected flat shape under concurrency: degrade safely (round-robin).
    logger.warning(
        "DllmScheduler: got a flat token-id list for a %d-request batch; "
        "splitting round-robin instead of pairing each request with one int. "
        "If token ids are misaligned, raise prefill_max_requests only when the "
        "model runner emits one inner list per request.",
        n,
    )
    out: list[list[int]] = [[] for _ in range(n)]
    for i, tid in enumerate(token_ids):
        out[i % n].append(int(tid))
    return out
