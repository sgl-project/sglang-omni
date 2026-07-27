# SPDX-License-Identifier: Apache-2.0
"""Explicit process-placement overrides for pipeline stages."""

from __future__ import annotations

from sglang_omni.config.schema import PipelineConfig


def apply_stage_process_overrides(
    pipeline_config: PipelineConfig,
    *,
    isolate_stages: list[str] | None = None,
) -> PipelineConfig:
    """Return a config with selected process-safe stages in their own process."""
    if not isolate_stages:
        return pipeline_config

    config = pipeline_config.model_copy(deep=True)
    stages = {stage.name: stage for stage in config.stages}
    role_map = type(config).isolation_role_to_stage()
    resource_contracts = type(config).isolation_stage_resources()
    isolated_stage_names: list[str] = []

    for requested_name in isolate_stages:
        stage = stages.get(requested_name)
        if stage is None:
            resolved_name = role_map.get(requested_name)
            stage = stages.get(resolved_name) if resolved_name is not None else None
        if stage is None:
            raise ValueError(f"Unknown stage or isolation role: {requested_name}")
        if stage.tp_size > 1:
            raise ValueError(
                f"Stage {stage.name!r} already uses one process per TP rank"
            )
        if stage.name not in resource_contracts:
            raise ValueError(f"Stage {stage.name!r} does not support process isolation")
        for resource_stage_name, memory_fraction in resource_contracts[
            stage.name
        ].items():
            resource_stage = stages.get(resource_stage_name)
            if resource_stage is None:
                raise ValueError(
                    f"Isolation resources for stage {stage.name!r} reference "
                    f"unknown stage {resource_stage_name!r}"
                )
            resources = resource_stage.runtime.resources
            if resources.total_gpu_memory_fraction is None:
                resources.total_gpu_memory_fraction = memory_fraction
        stage.process = stage.name
        isolated_stage_names.append(stage.name)

    from sglang_omni.config.placement import build_stage_placement_plan
    from sglang_omni.config.topology import build_process_topology_plan

    topology = build_process_topology_plan(
        config,
        build_stage_placement_plan(config),
    )
    groups = {group.name: group for group in topology.groups}
    for stage_name in isolated_stage_names:
        process_name = topology.stage_to_process[stage_name]
        stage_names = groups[process_name].stage_names
        if stage_names != (stage_name,):
            raise ValueError(
                f"Stage {stage_name!r} cannot be isolated because process group "
                f"{process_name!r} also contains stages {list(stage_names)}"
            )

    return config
