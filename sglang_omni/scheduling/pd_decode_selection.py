# SPDX-License-Identifier: Apache-2.0
"""Choosing which Decode half receives one request's KV.

This is a module-level function taking only values every Prefill rank sees
identically, and that shape is the point. See `select_decode_stage`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def select_decode_stage(targets: Sequence[str], request_id: str) -> str:
    """Return the Decode stage that receives ``request_id``'s KV.

    **Every Prefill rank runs this independently and they must agree.** The KV
    send is rank-addressed: rank *i* sends its shard to the chosen stage's rank
    *i* (`rank_endpoints[to_stage][tp_rank]` in `comm/engine.py`). At tp_size 1
    there is one rank and agreement is trivial. Above that, two ranks choosing
    differently scatter one request's shards across different Decode halves,
    each of which then holds part of the KV and none of which holds all of it.

    Nothing catches that. Each half admits on partial KV and attends over
    whatever occupies the missing heads. The output is wrong and nothing says
    so.

    So the choice must be a pure function of values every rank sees
    identically. ``request_id`` is one. Queue depth, in-flight counts, local
    counters and arrival order are not: they are the natural ingredients of a
    load-balancing policy and every one of them differs per rank.

    That is why this takes ``targets`` and ``request_id`` rather than a
    scheduler. Adding rank-local state means changing this signature, which is
    the point at which someone has to think about the paragraph above.

    A policy that needs global state -- least-loaded, or affinity to a cached
    prefix -- has to be decided in one place and carried to the ranks, not
    computed on each. The coordinator's admission binding for a replicated
    process is exactly that shape and is the natural home for one.
    """
    if not targets:
        raise ValueError("no decode targets to select from")
    if len(targets) == 1:
        return targets[0]
    digest = hashlib.blake2b(request_id.encode("utf-8"), digest_size=8).digest()
    return targets[int.from_bytes(digest, "big") % len(targets)]
