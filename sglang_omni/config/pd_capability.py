# SPDX-License-Identifier: Apache-2.0
"""Generic PD (prefill/decode) capability declaration and validation.

A factory must explicitly opt in via :func:`pd_disaggregation_capable`; the
compiler validates the marker in the parent process before spawning workers so a
mis-configured pipeline fails fast.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from sglang_omni.config.schema import StageConfig
from sglang_omni.utils.imports import import_string

# Note (Yue Yin): A private dunder avoids collisions with factory-owned metadata.
_PD_CAPABLE_ATTR = "__sglang_pd_disaggregation_capable__"


def pd_disaggregation_capable(factory: Callable[..., Any]) -> Callable[..., Any]:
    """Mark *factory* as able to run as a PD prefill/decode half."""
    setattr(factory, _PD_CAPABLE_ATTR, True)
    return factory


def factory_supports_pd(factory: Callable[..., Any]) -> bool:
    """Return whether *factory* declared PD-disaggregation capability."""
    return bool(getattr(factory, _PD_CAPABLE_ATTR, False))


def validate_pd_capabilities(stages: Iterable[StageConfig]) -> None:
    """Reject PD-enabled stages whose factory is not PD-capable.

    Runs in the parent process before workers are spawned. Only PD-expanded
    stages are checked, so ordinary non-PD pipelines do not import factories
    here.
    """
    for stage in stages:
        if stage.pd_execution is None:
            continue
        factory = import_string(stage.factory_path)
        if not factory_supports_pd(factory):
            raise ValueError(
                f"Stage {stage.name!r} is PD-disaggregated (role="
                f"{stage.pd_execution.role!r}) but its factory "
                f"{stage.factory_path!r} "
                "is not PD-capable; decorate the factory with "
                "@pd_disaggregation_capable to opt in"
            )
