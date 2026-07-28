# SPDX-License-Identifier: Apache-2.0
"""Explicit process-placement overrides for pipeline stages."""

from __future__ import annotations

from sglang_omni.config.schema import PipelineConfig, StageResourceConfig


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
    process_safe_stages = type(config).process_isolation_stages()
    resource_contracts = type(config).isolation_stage_resources()
    baseline_groups = _stage_process_groups(config)
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
        # Note (Akazaakane): a stage the model already places alone is the requested
        # topology, so accept it unchanged instead of demanding a resource contract
        # that would only re-declare fractions the config already carries.
        if baseline_groups[stage.name][1] == (stage.name,):
            continue
        if stage.name not in process_safe_stages:
            raise ValueError(f"Stage {stage.name!r} does not support process isolation")
        for resource_stage_name, memory_fraction in resource_contracts.get(
            stage.name, {}
        ).items():
            resource_stage = stages.get(resource_stage_name)
            if resource_stage is None:
                raise ValueError(
                    f"Isolation resources for stage {stage.name!r} reference "
                    f"unknown stage {resource_stage_name!r}"
                )
            resources = resource_stage.runtime.resources
            if resources.total_gpu_memory_fraction is None:
                # Note (Akazaakane): plain attribute assignment skips the field
                # validator, so an out-of-range contract value would reach placement
                # accounting unchecked. Re-validate the whole resource block.
                resource_stage.runtime.resources = StageResourceConfig.model_validate(
                    {
                        **resources.model_dump(),
                        "total_gpu_memory_fraction": memory_fraction,
                    }
                )
        stage.process = stage.name
        isolated_stage_names.append(stage.name)

    if not isolated_stage_names:
        return config

    resolved_groups = _stage_process_groups(config)
    for stage_name in isolated_stage_names:
        process_name, group_stage_names = resolved_groups[stage_name]
        if group_stage_names != (stage_name,):
            raise ValueError(
                f"Stage {stage_name!r} cannot be isolated because process group "
                f"{process_name!r} also contains stages {list(group_stage_names)}"
            )

    return config


def _stage_process_groups(
    config: PipelineConfig,
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Map each non-TP stage to its process name and that process's members."""
    from sglang_omni.config.placement import build_stage_placement_plan
    from sglang_omni.config.topology import build_process_topology_plan

    # Note (Akazaakane): apply_policy=False because the placement policy is a
    # user-supplied hook that the CLI chain runs later, after the remaining
    # overrides. Running it here would execute it against a half-built config and
    # a second time in prepare_pipeline_runtime. Only resolved GPU ids are needed.
    topology = build_process_topology_plan(
        config,
        build_stage_placement_plan(config, apply_policy=False),
    )
    groups = {group.name: group for group in topology.groups}
    return {
        stage_name: (process_name, groups[process_name].stage_names)
        for stage_name, process_name in topology.stage_to_process.items()
    }
