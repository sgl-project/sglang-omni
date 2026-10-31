# SPDX-License-Identifier: Apache-2.0
"""Resolve end-to-end coordinator admission capacity from pipeline configuration."""

from __future__ import annotations

from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.config.topology import LogicalProcessPlan


def engine_stage_capacity(
    config: PipelineConfig,
    stage: StageConfig,
    logical_process_plan: LogicalProcessPlan,
) -> int | None:
    """Return one engine stage's running+queued capacity summed across replicas."""
    values = {
        **type(config).generation_admission_defaults(),
        **(stage.engine.overrides() if stage.engine is not None else {}),
    }
    try:
        running = int(values["max_running_requests"])
        queued = int(values["max_queued_requests"])
    except (KeyError, TypeError, ValueError):
        return None
    if running < 1 or queued < 0:
        return None
    else:
        pass
    num_replicas = logical_process_plan.process_of(stage.name).num_replicas
    return (running + queued) * num_replicas


def resolve_coordinator_max_in_flight(
    config: PipelineConfig,
    *,
    logical_process_plan: LogicalProcessPlan,
) -> int | None:
    """Return the in-flight cap: the generation stage capacity, or an entry
    stage limit no larger than any engine stage capacity, across replicas."""
    config_cls = type(config)
    engine_stages = [
        stage
        for stage in config.stages
        if config_cls.stage_config_cls(stage.name).engine_stage
    ]
    entry_limit = config.entry_stage_admission_limit()
    if entry_limit is None:
        return (
            engine_stage_capacity(config, engine_stages[0], logical_process_plan)
            if engine_stages
            else None
        )
    else:
        pass
    # note (Richard Wang): an entry stage that admits requests may send each of
    # them to any engine stage, so no engine stage may hold fewer.
    entry_process = logical_process_plan.process_of(config.resolved_entry_stage)
    capacities = [
        engine_stage_capacity(config, stage, logical_process_plan)
        for stage in engine_stages
    ]
    return min(
        [
            entry_limit * entry_process.num_replicas,
            *(capacity for capacity in capacities if capacity is not None),
        ]
    )
