# SPDX-License-Identifier: Apache-2.0
"""Per-stage host CPU cost declarations.

A stage declares how it consumes host CPU, not how many CPUs it wants:

- ``serial-loop``: a dispatch loop bound by single-thread cycles. It needs a
  small number of exclusive physical cores (SMT siblings reserved with them),
  not width.
- ``gpu-bound``: negligible host cost; stays in the shared pool.

Undeclared stages default to ``gpu-bound``, so a model without declarations
keeps today's behavior even when the allocator is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

HOST_CLASSES = ("serial-loop", "gpu-bound")


@dataclass(frozen=True)
class StageCpuCost:
    host_class: str
    exclusive_cores: int = 1

    def __post_init__(self) -> None:
        if self.host_class not in HOST_CLASSES:
            raise ValueError(
                f"host_class must be one of {HOST_CLASSES}, got {self.host_class!r}"
            )
        if self.host_class == "serial-loop" and self.exclusive_cores < 1:
            raise ValueError(
                f"serial-loop requires exclusive_cores >= 1, "
                f"got {self.exclusive_cores}"
            )


def resolve_stage_cpu_costs(config) -> dict[str, StageCpuCost]:
    """Resolve validated stage costs from the pipeline config class.

    Reads the ``stage_cpu_costs()`` classmethod (raw dicts, mirroring
    ``process_edge_resources()``) and validates every entry against the
    stages that actually exist in the config.
    """
    raw = type(config).stage_cpu_costs()
    if not raw:
        return {}
    stage_names = {stage.name for stage in config.stages}
    costs: dict[str, StageCpuCost] = {}
    for stage_name, entry in raw.items():
        if stage_name not in stage_names:
            raise ValueError(
                f"stage_cpu_costs() names unknown stage {stage_name!r}; "
                f"known stages: {sorted(stage_names)}"
            )
        if not isinstance(entry, dict) or "host_class" not in entry:
            raise ValueError(
                f"stage_cpu_costs()[{stage_name!r}] must be a dict with a "
                f"'host_class' key, got {entry!r}"
            )
        unknown = set(entry) - {"host_class", "exclusive_cores"}
        if unknown:
            raise ValueError(
                f"stage_cpu_costs()[{stage_name!r}] has unknown keys {sorted(unknown)}"
            )
        costs[stage_name] = StageCpuCost(
            host_class=entry["host_class"],
            exclusive_cores=int(entry.get("exclusive_cores", 1)),
        )
    return costs
