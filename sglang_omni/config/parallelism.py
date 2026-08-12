# SPDX-License-Identifier: Apache-2.0
"""Resolve a stage's configured parallel execution mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sglang_omni.config.schema import StageConfig

ParallelKind = Literal["tp", "sp"]


@dataclass(frozen=True)
class ResolvedStageParallelism:
    """Internal, mutually exclusive TP/SP execution metadata."""

    kind: ParallelKind | None
    size: int

    @property
    def tp_size(self) -> int:
        return self.size if self.kind == "tp" else 1

    @property
    def sp_size(self) -> int:
        return self.size if self.kind == "sp" else 1

    @property
    def is_parallel(self) -> bool:
        return self.kind is not None


def resolve_stage_parallelism(stage: StageConfig) -> ResolvedStageParallelism:
    """Return the one parallel mode configured for ``stage``."""
    if stage.tp_size != stage.parallelism.tp:
        raise ValueError(
            f"Stage {stage.name!r}: tp_size={stage.tp_size} conflicts with "
            f"parallelism.tp={stage.parallelism.tp}"
        )
    if stage.parallelism.sp > 1 and stage.tp_size > 1:
        raise ValueError(
            f"Stage {stage.name!r} cannot enable TP and SP at the same time"
        )
    if stage.parallelism.sp > 1:
        return ResolvedStageParallelism(kind="sp", size=stage.parallelism.sp)
    if stage.tp_size > 1:
        return ResolvedStageParallelism(kind="tp", size=stage.tp_size)
    return ResolvedStageParallelism(kind=None, size=1)
