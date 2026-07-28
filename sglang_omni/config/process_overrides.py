# SPDX-License-Identifier: Apache-2.0
"""Explicit process-placement overrides for pipeline stages."""

from __future__ import annotations

import logging

from sglang_omni.config.schema import PipelineConfig, StageConfig, StageResourceConfig

logger = logging.getLogger(__name__)


def parse_stage_process_assignment(value: str) -> tuple[str, str]:
    """Parse one ``STAGE=PROCESS`` placement assignment."""
    stage_name, separator, process_name = value.partition("=")

    stage_name = stage_name.strip()
    process_name = process_name.strip()

    if not separator or not stage_name or not process_name:
        raise ValueError(
            f"Invalid stage process assignment {value!r}; expected STAGE=PROCESS"
        )

    return stage_name, process_name


def apply_stage_process_overrides(
    pipeline_config: PipelineConfig,
    *,
    isolate_stages: list[str] | None = None,
    stage_processes: list[str] | None = None,
) -> PipelineConfig:
    """Return a config with stages moved into the requested processes.

    ``isolate_stages`` is singleton shorthand: ``--isolate-stage X`` requests the
    same placement as ``--stage-process X=X``. ``stage_processes`` additionally
    expresses grouping, because repeating one process name colocates stages.
    """
    if not isolate_stages and not stage_processes:
        return pipeline_config

    config = pipeline_config.model_copy(deep=True)
    stages = {stage.name: stage for stage in config.stages}
    role_map = type(config).isolation_role_to_stage()
    resource_contracts = type(config).process_edge_resources()
    baseline_processes = _declared_process_map(config)

    requested: dict[str, str] = {}
    for assignment in stage_processes or []:
        requested_name, process_name = parse_stage_process_assignment(assignment)
        stage = _resolve_stage(stages, role_map, requested_name)
        _reject_tp_stage(stage)
        if stage.name in requested:
            raise ValueError(
                f"Stage {stage.name!r} has multiple --stage-process assignments: "
                f"{requested[stage.name]!r} and {process_name!r}"
            )
        requested[stage.name] = process_name

    singleton_stage_names: list[str] = []
    for requested_name in isolate_stages or []:
        stage = _resolve_stage(stages, role_map, requested_name)
        _reject_tp_stage(stage)
        if stage.name in requested:
            raise ValueError(
                f"Stage {stage.name!r} cannot use both --isolate-stage and "
                "--stage-process"
            )
        # Note (Akazaakane): a stage the model already runs alone is the requested
        # topology, so accept it unchanged instead of demanding a resource contract
        # that would only re-declare fractions the config already carries.
        if _runs_alone(baseline_processes, stage.name):
            continue
        requested[stage.name] = stage.name
        singleton_stage_names.append(stage.name)

    if not requested:
        return config

    for stage_name, process_name in requested.items():
        stages[stage_name].process = process_name

    new_cross_process_edges = _new_cross_process_edges(config, baseline_processes)

    # Capability first: report an unsupported handoff before a missing fraction,
    # because declaring fractions would not make that split correct.
    _validate_process_safe_edges(config, new_cross_process_edges)
    _apply_process_edge_resources(
        stages,
        new_cross_process_edges,
        resource_contracts,
    )

    resolved_groups = _stage_process_groups(config)
    for stage_name in singleton_stage_names:
        process_name, group_stage_names = resolved_groups[stage_name]
        if group_stage_names != (stage_name,):
            raise ValueError(
                f"Stage {stage_name!r} cannot be isolated because process group "
                f"{process_name!r} also contains stages {list(group_stage_names)}"
            )
    for stage_name, process_name in requested.items():
        resolved_name = resolved_groups[stage_name][0]
        if resolved_name != process_name:
            raise ValueError(
                f"Stage {stage_name!r} was assigned to process {process_name!r} "
                f"but resolved to process group {resolved_name!r}"
            )

    return config


def _resolve_stage(
    stages: dict[str, StageConfig],
    role_map: dict[str, str],
    requested_name: str,
) -> StageConfig:
    stage = stages.get(requested_name)
    if stage is None:
        resolved_name = role_map.get(requested_name)
        stage = stages.get(resolved_name) if resolved_name is not None else None
    if stage is None:
        raise ValueError(f"Unknown stage or isolation role: {requested_name}")
    return stage


def _reject_tp_stage(stage: StageConfig) -> None:
    if stage.tp_size > 1:
        raise ValueError(f"Stage {stage.name!r} already uses one process per TP rank")


def _runs_alone(process_map: dict[str, str], stage_name: str) -> bool:
    process_name = process_map.get(stage_name)
    if process_name is None:
        return False
    return sum(1 for name in process_map.values() if name == process_name) == 1


def _apply_process_edge_resources(
    stages: dict[str, StageConfig],
    new_cross_process_edges: set[tuple[str, str]],
    resource_contracts: dict[tuple[str, str], dict[str, float]],
) -> None:
    recommendations: dict[str, tuple[float, tuple[str, str]]] = {}
    for edge in sorted(new_cross_process_edges):
        for resource_stage_name, memory_fraction in resource_contracts.get(
            edge, {}
        ).items():
            resource_stage = stages.get(resource_stage_name)
            if resource_stage is None:
                raise ValueError(
                    f"Process-edge resources for {edge!r} reference unknown stage "
                    f"{resource_stage_name!r}"
                )
            validated_fraction = StageResourceConfig.model_validate(
                {"total_gpu_memory_fraction": memory_fraction}
            ).total_gpu_memory_fraction
            assert validated_fraction is not None
            previous = recommendations.get(resource_stage_name)
            if previous is not None and previous[0] != validated_fraction:
                raise ValueError(
                    f"Conflicting process-edge resources for stage "
                    f"{resource_stage_name!r}: edge {previous[1]!r} recommends "
                    f"{previous[0]} but edge {edge!r} recommends "
                    f"{validated_fraction}"
                )
            recommendations[resource_stage_name] = (validated_fraction, edge)

    for resource_stage_name, (memory_fraction, edge) in sorted(recommendations.items()):
        resource_stage = stages[resource_stage_name]
        resources = resource_stage.runtime.resources
        explicit_fraction = resources.total_gpu_memory_fraction
        if explicit_fraction is None:
            # Note (Akazaakane): plain attribute assignment skips the field
            # validator, so re-validate the whole resource block.
            resource_stage.runtime.resources = StageResourceConfig.model_validate(
                {
                    **resources.model_dump(),
                    "total_gpu_memory_fraction": memory_fraction,
                }
            )
        elif explicit_fraction < memory_fraction:
            logger.warning(
                "Stage %r declares total_gpu_memory_fraction=%s below "
                "process-edge resource contract %s for edge %r; preserving "
                "the explicit value",
                resource_stage_name,
                explicit_fraction,
                memory_fraction,
                edge,
            )


def _new_cross_process_edges(
    config: PipelineConfig,
    baseline_processes: dict[str, str],
) -> set[tuple[str, str]]:
    requested_processes = _declared_process_map(config)
    return {
        (source, destination)
        for source, destination in _pipeline_edges(config)
        if not _is_cross_process(baseline_processes, source, destination)
        and _is_cross_process(requested_processes, source, destination)
    }


def _validate_process_safe_edges(
    config: PipelineConfig,
    new_cross_process_edges: set[tuple[str, str]],
) -> None:
    """Reject handoffs the model has not declared safe to split."""
    safe_edges = type(config).process_safe_edges()

    for source, destination in sorted(new_cross_process_edges):
        if (source, destination) not in safe_edges:
            raise ValueError(
                f"Process split across {source!r} -> {destination!r} is not "
                "supported by this model"
            )


def _is_cross_process(
    process_map: dict[str, str],
    source: str,
    destination: str,
) -> bool:
    # Note (Akazaakane): a TP stage is absent from this map because it already runs
    # one process per rank, so any edge touching it is crossed either way.
    if source not in process_map or destination not in process_map:
        return True
    return process_map[source] != process_map[destination]


def _declared_process_map(config: PipelineConfig) -> dict[str, str]:
    """Process name per non-TP stage from declarations alone.

    Deliberately independent of GPU placement so an unsupported split is
    reported before placement accounting rejects a missing memory fraction.
    """
    process_by_stage = {
        stage.name: stage.process
        for stage in config.stages
        if stage.tp_size == 1 and stage.process
    }
    for group in config.fused_stages or []:
        members = [name for name in group if name in process_by_stage]
        if not members:
            continue
        shared = process_by_stage[members[0]]
        for name in members:
            process_by_stage[name] = shared
    return process_by_stage


def _pipeline_edges(config: PipelineConfig) -> set[tuple[str, str]]:
    """Every statically declared handoff between two stages."""
    edges: set[tuple[str, str]] = set()
    for stage in config.stages:
        targets = stage.next
        if isinstance(targets, str):
            targets = [targets]
        for target in list(targets or []) + list(stage.stream_to):
            edges.add((stage.name, target))
        for source in stage.wait_for or []:
            edges.add((source, stage.name))
    return edges


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
