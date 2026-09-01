# SPDX-License-Identifier: Apache-2.0
"""Explicit factory opt-in for compiler-owned PD scheduler construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from sglang_omni.config.schema import StageConfig

PD_CAPABLE_MARKER = "__sglang_pd_disaggregation_capable__"


def pd_disaggregation_capable(factory: Callable[..., Any]) -> Callable[..., Any]:
    setattr(factory, PD_CAPABLE_MARKER, True)
    return factory


# Note (Audrey Zheng): what a PD half cannot do yet. The scheduler enforces
# these in its constructor (`_validate_pd_runtime`), which is the last
# possible moment -- the model has loaded and the pipeline is half up before
# anyone learns the config was never going to work. Check them where the
# config is read instead, and name the stage.
#
# These are temporary. Each one names what has to be built, so that lifting
# it is a deletion here rather than a search.
_PD_UNSUPPORTED_ENGINE_ARGS: tuple[tuple[str, Any, str], ...] = (
    (
        "page_size",
        1,
        "the KV handoff addresses one page per token; see request_page_indices",
    ),
    (
        "disable_radix_cache",
        True,
        "a shared prefix makes the handed-off pages not solely this "
        "request's, and DecodeContinuation carries neither extra_key nor "
        "cache_salt, so the Decode half would insert under a key that "
        "collides with a different request holding the same token ids",
    ),
)


def validate_pd_engine_args(stages: Iterable[StageConfig]) -> None:
    """Reject a PD config the runtime is going to refuse anyway.

    Applying these silently instead would be worse: an operator who asked for
    `page_size=32` would get 1 without being told, and would keep getting it
    after the restriction is lifted.
    """
    for stage in stages:
        if stage.pd_execution is None or stage.engine is None:
            continue
        for name, required, because in _PD_UNSUPPORTED_ENGINE_ARGS:
            declared = getattr(stage.engine, name, None)
            if declared is None or declared == required:
                continue
            raise ValueError(
                f"Stage {stage.name!r} is PD-disaggregated and declares "
                f"{name}={declared!r}, but PD currently requires {required!r}: "
                f"{because}"
            )
