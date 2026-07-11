# SPDX-License-Identifier: Apache-2.0
"""Process topology resolution for pipeline stages.

This module does not decide GPU placement. It consumes the existing resolved
GPU placement from the colocation planner, then answers one question: which
non-TP stages should run in the same OS process?
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass

from sglang_omni.config.placement import StagePlacementPlan, resolve_stage_gpu_ids
from sglang_omni.config.schema import PipelineConfig, StageConfig


@dataclass(frozen=True)
class ProcessGroupPlacement:
    """Resolved non-TP stage process group."""

    name: str
    stage_names: tuple[str, ...]
    gpu_id: int | None


@dataclass(frozen=True)
class ProcessTopologyPlan:
    """Resolved process topology derived from the pipeline config."""

    groups: tuple[ProcessGroupPlacement, ...]
    stage_to_process: dict[str, str]
    tp_stage_to_processes: dict[str, tuple[str, ...]]


def build_process_topology_plan(
    config: PipelineConfig,
    gpu_placement: StagePlacementPlan,
    *,
    stages_cfg: list[StageConfig] | None = None,
) -> ProcessTopologyPlan:
    stages = stages_cfg if stages_cfg is not None else config.stages
    groups = _build_process_groups(config, stages, gpu_placement)
    tp_stage_to_processes = _build_tp_process_names(stages)

    plan = ProcessTopologyPlan(
        groups=tuple(groups),
        stage_to_process={
            stage_name: group.name
            for group in groups
            for stage_name in group.stage_names
        },
        tp_stage_to_processes=tp_stage_to_processes,
    )
    _validate_process_name_uniqueness(plan)
    _validate_gpu_process_colocation(config, gpu_placement, stages, plan)
    return plan


def _build_process_groups(
    config: PipelineConfig,
    stages: list[StageConfig],
    gpu_placement: StagePlacementPlan,
) -> list[ProcessGroupPlacement]:
    non_tp_stages = [stage for stage in stages if stage.tp_size == 1]
    _validate_non_tp_processes(non_tp_stages)

    components = _resolve_non_tp_process_components(config, non_tp_stages)
    used_names: set[str] = set()
    groups: list[ProcessGroupPlacement] = []
    for component in components.values():
        group_name = _component_process_name(component, used_names)
        groups.append(
            ProcessGroupPlacement(
                name=group_name,
                stage_names=tuple(stage.name for stage in component),
                gpu_id=_resolve_group_gpu_id(group_name, component, gpu_placement),
            )
        )
    return groups


def _validate_non_tp_processes(stages: list[StageConfig]) -> None:
    for stage in stages:
        if stage.process is None:
            raise ValueError(
                f"Stage {stage.name!r} must declare process; non-TP stage "
                "process groups are explicit"
            )


def _resolve_non_tp_process_components(
    config: PipelineConfig,
    stages: list[StageConfig],
) -> OrderedDict[str, list[StageConfig]]:
    parent = {stage.name: stage.name for stage in stages}
    stage_by_name = {stage.name: stage for stage in stages}

    def find(name: str) -> str:
        root = name
        while parent[root] != root:
            root = parent[root]
        while parent[name] != name:
            next_name = parent[name]
            parent[name] = root
            name = next_name
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_process: OrderedDict[str, list[str]] = OrderedDict()
    for stage in stages:
        by_process.setdefault(stage.process or "", []).append(stage.name)
    for stage_names in by_process.values():
        first = stage_names[0]
        for stage_name in stage_names[1:]:
            union(first, stage_name)

    for group in config.fused_stages or []:
        local_stage_names = [
            stage_name
            for stage_name in group
            if stage_name in stage_by_name and stage_by_name[stage_name].tp_size == 1
        ]
        if not local_stage_names:
            continue
        first = local_stage_names[0]
        for stage_name in local_stage_names[1:]:
            union(first, stage_name)

    components: OrderedDict[str, list[StageConfig]] = OrderedDict()
    for stage in stages:
        components.setdefault(find(stage.name), []).append(stage)
    return components


def _component_process_name(
    stages: list[StageConfig],
    used_names: set[str],
) -> str:
    explicit_names = {stage.process for stage in stages if stage.process}
    if len(explicit_names) == 1:
        base_name = next(iter(explicit_names))
    else:
        base_name = "fused_" + "_".join(stage.name for stage in stages)

    name = base_name
    suffix = 1
    while name in used_names:
        suffix += 1
        name = f"{base_name}_{suffix}"
    used_names.add(name)
    return name


def _build_tp_process_names(stages: list[StageConfig]) -> dict[str, tuple[str, ...]]:
    return {
        stage.name: tuple(
            _tp_process_name(stage, tp_rank) for tp_rank in range(stage.tp_size)
        )
        for stage in stages
        if stage.tp_size > 1
    }


def _tp_process_name(stage: StageConfig, tp_rank: int) -> str:
    process_base = stage.process or stage.name
    return f"{process_base}_tp{tp_rank}"


def _validate_process_name_uniqueness(plan: ProcessTopologyPlan) -> None:
    non_tp_processes = set(plan.stage_to_process.values())
    tp_processes = [
        process_name
        for process_names in plan.tp_stage_to_processes.values()
        for process_name in process_names
    ]
    duplicate_tp_processes = sorted(
        process_name
        for process_name, count in Counter(tp_processes).items()
        if count > 1
    )
    if duplicate_tp_processes:
        raise ValueError(f"Duplicate TP process names: {duplicate_tp_processes}")

    collisions = sorted(non_tp_processes.intersection(tp_processes))
    if collisions:
        raise ValueError(
            "TP-derived process names collide with non-TP process groups: "
            f"{collisions}"
        )


def _resolve_group_gpu_id(
    group_name: str,
    stages: list[StageConfig],
    gpu_placement: StagePlacementPlan,
) -> int | None:
    gpu_ids = {
        gpu_id
        for stage in stages
        for gpu_id in _stage_gpu_ids(gpu_placement, stage)
        if gpu_id is not None
    }
    if len(gpu_ids) > 1:
        stage_names = ", ".join(stage.name for stage in stages)
        raise ValueError(
            f"Process group {group_name!r} spans multiple GPUs "
            f"{sorted(gpu_ids)} through stages: {stage_names}"
        )
    return next(iter(gpu_ids), None)


def _stage_gpu_ids(
    gpu_placement: StagePlacementPlan,
    stage: StageConfig,
) -> list[int | None]:
    """Return resolved per-rank GPU ids for a stage.

    This is the only fact topology needs from the GPU-placement side.
    """
    return resolve_stage_gpu_ids(gpu_placement, stage)


def _validate_gpu_process_colocation(
    config: PipelineConfig,
    gpu_placement: StagePlacementPlan,
    stages: list[StageConfig],
    topology_plan: ProcessTopologyPlan,
) -> None:
    stage_by_name = {stage.name: stage for stage in stages}
    gpu_processes: dict[int, set[str]] = defaultdict(set)
    missing_fraction: dict[int, set[str]] = defaultdict(set)

    for group in topology_plan.groups:
        for stage_name in group.stage_names:
            stage = stage_by_name[stage_name]
            for gpu_id in _stage_gpu_ids(gpu_placement, stage):
                if gpu_id is None:
                    continue
                gpu_processes[gpu_id].add(group.name)
                if stage.runtime.resources.total_gpu_memory_fraction is None:
                    missing_fraction[gpu_id].add(stage.name)

    for stage in stages:
        if stage.tp_size <= 1:
            continue
        for rank, gpu_id in enumerate(_stage_gpu_ids(gpu_placement, stage)):
            if gpu_id is None:
                continue
            gpu_processes[gpu_id].add(
                topology_plan.tp_stage_to_processes[stage.name][rank]
            )
            if stage.runtime.resources.total_gpu_memory_fraction is None:
                missing_fraction[gpu_id].add(stage.name)

    require = config.placement.require_memory_fraction_for_colocation
    limit = config.placement.max_total_gpu_memory_fraction_per_gpu
    for gpu_id, process_names in gpu_processes.items():
        if len(process_names) <= 1:
            continue
        missing = sorted(missing_fraction.get(gpu_id, set()))
        if require and missing:
            raise ValueError(
                f"GPU {gpu_id} is shared by multiple process groups without "
                "runtime.resources.total_gpu_memory_fraction: "
                f"{', '.join(missing)}"
            )
        total = gpu_placement.gpus[gpu_id].total_gpu_memory_fraction
        if total > limit + 1e-9:
            raise ValueError(
                f"GPU {gpu_id} total_gpu_memory_fraction={total:.3f} exceeds "
                f"placement limit {limit:.3f}"
            )
