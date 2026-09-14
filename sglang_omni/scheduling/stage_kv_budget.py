# SPDX-License-Identifier: Apache-2.0
"""Deliver a stage's declared KV byte budget to its SGLang engine bootstrap.

A stage's ``engine.kv_cache_bytes`` must reach the KV-cache
configurator without threading a new keyword argument through every model's
factory and engine-builder signature. The stage worker scopes the budget
around the single factory invocation choke point, and
``create_sglang_infrastructure`` consumes it at the single engine construction
choke point, so model files need no per-model plumbing and a newly added model
is covered automatically.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class _StageKvBudget:
    stage_name: str
    kv_cache_bytes: int
    consumed: bool = False


_local = threading.local()


def _current() -> _StageKvBudget | None:
    return getattr(_local, "budget", None)


@contextmanager
def stage_kv_cache_budget(stage_name: str, kv_cache_bytes: int):
    """Scope a declared KV byte budget around one stage factory invocation.

    Raises on exit when the budget was never consumed: the stage declared
    ``engine.kv_cache_bytes`` but its factory did not construct an
    SGLang engine, so honoring the budget is impossible and silently ignoring
    it would fake a guarantee the deployment relies on.
    """
    active = _current()
    if active is not None:
        raise RuntimeError(
            f"stage_kv_cache_budget for stage {stage_name!r} cannot nest inside "
            f"the active scope for stage {active.stage_name!r}"
        )
    budget = _StageKvBudget(stage_name=stage_name, kv_cache_bytes=kv_cache_bytes)
    _local.budget = budget
    try:
        yield
    finally:
        _local.budget = None
    if not budget.consumed:
        raise RuntimeError(
            f"Stage {stage_name!r} declares engine.kv_cache_bytes but its "
            "factory did not build an SGLang engine that consumes a KV byte "
            "budget; remove the setting or place it on a stage with a KV cache"
        )


def consume_stage_kv_cache_bytes() -> int | None:
    """Return the scoped budget for engine construction, marking it consumed.

    A second consumption in the same scope raises: two engines each sized to
    the full stage budget would silently commit twice the declared bytes.
    """
    budget = _current()
    if budget is None:
        return None
    if budget.consumed:
        raise RuntimeError(
            f"Stage {budget.stage_name!r} declares one "
            "engine.kv_cache_bytes budget but its factory constructed "
            "a second SGLang engine; a stage byte budget covers exactly one "
            "engine's KV pool"
        )
    budget.consumed = True
    return budget.kv_cache_bytes


def peek_stage_kv_cache_bytes() -> int | None:
    """Return the scoped budget without consuming it."""
    budget = _current()
    if budget is None:
        return None
    return budget.kv_cache_bytes
