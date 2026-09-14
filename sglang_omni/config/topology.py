# SPDX-License-Identifier: Apache-2.0
"""Process topology resolution for pipeline stages.

Two plans live here. ``LogicalProcessPlan`` is compiled from the config alone
and is the single place where Process membership is decided: it groups stages
by ``StageConfig.process``, attaches the sparse ``PipelineConfig.processes``
replica policy, and validates every cross-process edge of the final topology.

``ProcessTopologyPlan`` is built after replica expansion and GPU placement. It
does not decide placement either; it answers which expanded non-TP stages share
one OS process, and which OS processes a TP stage spans.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass

from sglang_omni.config.placement import StagePlacementPlan, resolve_stage_gpu_ids
from sglang_omni.config.schema import (
    PipelineConfig,
    ProcessConfig,
    StageConfig,
    stage_process_name,
)

_DEFAULT_PROCESS_CONFIG = ProcessConfig()


@dataclass(frozen=True)
class LogicalProcess:
    """One logical Process: the unit a replica copies as a whole."""

    name: str
    stage_names: tuple[str, ...]
    tp_size: int
    num_replicas: int
    # Note (kaige): one inner tuple per replica, each holding tp_size device
    # ids. None when the process declares no replica_devices.
    replica_devices: tuple[tuple[int, ...], ...] | None

    @property
    def is_replicated(self) -> bool:
        return self.num_replicas > 1

    @property
    def is_tensor_parallel(self) -> bool:
        return self.tp_size > 1


@dataclass(frozen=True)
class LogicalProcessPlan:
    """Validated logical Process topology.

    Membership is final once this plan exists: replica expansion copies whole
    processes, and no later step re-derives membership from stage names.
    """

    processes: tuple[LogicalProcess, ...]
    stage_to_process: dict[str, str]

    def get(self, process_name: str) -> LogicalProcess:
        for process in self.processes:
            if process.name == process_name:
                return process
        raise KeyError(f"Unknown process name: {process_name!r}")

    def process_of(self, stage_name: str) -> LogicalProcess:
        return self.get(self.stage_to_process[stage_name])

    def num_replicas(self, process_name: str) -> int:
        return self.get(process_name).num_replicas

    def replicated_process_names(self) -> tuple[str, ...]:
        return tuple(
            process.name for process in self.processes if process.is_replicated
        )

    def has_replicas(self) -> bool:
        return any(process.is_replicated for process in self.processes)


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


def compile_logical_processes(
    config: PipelineConfig,
) -> tuple[LogicalProcessPlan, list[StageConfig]]:
    """Compile the logical Process topology from ``config``.

    This is the final configuration-validation layer before worker startup;
    downstream runtime code consumes the resulting plan without revalidating it.

    Returns the plan plus stage-config copies. The input config is left
    untouched.
    """
    stages = [stage.model_copy(deep=True) for stage in config.stages]
    members = _group_stages_by_process(stages)
    processes = tuple(
        _build_logical_process(name, member_stages, config.processes.get(name))
        for name, member_stages in members.items()
    )
    plan = LogicalProcessPlan(
        processes=processes,
        stage_to_process={
            stage_name: process.name
            for process in processes
            for stage_name in process.stage_names
        },
    )

    cross_process_edges = _cross_process_edges(stages, plan)
    _validate_process_local_edges(config, cross_process_edges)
    return plan, stages


def _group_stages_by_process(
    stages: list[StageConfig],
) -> OrderedDict[str, list[StageConfig]]:
    """Group stages by declared Process Name, preserving config order.

    Config order is load order inside one OS process, and
    ``process_total_gpu_memory_fraction`` accumulates along it.
    """
    members: OrderedDict[str, list[StageConfig]] = OrderedDict()
    for stage in stages:
        members.setdefault(stage_process_name(stage), []).append(stage)
    return members


def _build_logical_process(
    name: str,
    stages: list[StageConfig],
    process_cfg: ProcessConfig | None,
) -> LogicalProcess:
    policy = process_cfg if process_cfg is not None else _DEFAULT_PROCESS_CONFIG
    # Note (kaige): a TP stage owns its process outright, so the max is that
    # stage's tp_size; a shared process only ever holds non-TP stages.
    tp_size = max(stage.tp_size for stage in stages)
    return LogicalProcess(
        name=name,
        stage_names=tuple(stage.name for stage in stages),
        tp_size=tp_size,
        num_replicas=policy.num_replicas,
        replica_devices=_resolve_replica_devices(
            name,
            stages,
            policy,
            tp_size=tp_size,
        ),
    )


def _resolve_replica_devices(
    process_name: str,
    stages: list[StageConfig],
    policy: ProcessConfig,
    *,
    tp_size: int,
) -> tuple[tuple[int, ...], ...] | None:
    device_ids = policy.replica_devices
    has_gpu_stage = any(stage.gpu is not None for stage in stages)

    if device_ids is None:
        if policy.num_replicas > 1 and has_gpu_stage:
            raise ValueError(
                f"Process {process_name!r}: num_replicas={policy.num_replicas} "
                "on a process with GPU stage(s) requires replica_devices with "
                f"{policy.num_replicas * tp_size} device id(s)"
            )
        return None

    if not has_gpu_stage:
        raise ValueError(
            f"Process {process_name!r} has no GPU stage and must not declare "
            "replica_devices"
        )

    expected = policy.num_replicas * tp_size
    if len(device_ids) != expected:
        raise ValueError(
            f"Process {process_name!r}: replica_devices has {len(device_ids)} "
            f"id(s); expected {expected} (num_replicas={policy.num_replicas} x "
            f"tp_size={tp_size})"
        )
    replica_devices = tuple(
        tuple(device_ids[index * tp_size : (index + 1) * tp_size])
        for index in range(policy.num_replicas)
    )
    for replica_id, devices in enumerate(replica_devices):
        if len(set(devices)) != len(devices):
            raise ValueError(
                f"Process {process_name!r}: replica_devices for replica "
                f"{replica_id} must contain unique GPU ids, got {list(devices)}"
            )
    return replica_devices


def _pipeline_edges(stages: list[StageConfig]) -> set[tuple[str, str]]:
    """Every statically declared handoff between two stages."""
    edges: set[tuple[str, str]] = set()
    for stage in stages:
        targets = stage.next
        if isinstance(targets, str):
            targets = [targets]
        for target in list(targets or []) + list(stage.stream_to):
            edges.add((stage.name, target))
        for source in stage.wait_for or []:
            edges.add((source, stage.name))
    return edges


def _cross_process_edges(
    stages: list[StageConfig],
    plan: LogicalProcessPlan,
) -> set[tuple[str, str]]:
    return {
        (source, destination)
        for source, destination in _pipeline_edges(stages)
        if plan.stage_to_process[source] != plan.stage_to_process[destination]
    }


def _validate_process_local_edges(
    config: PipelineConfig,
    cross_process_edges: set[tuple[str, str]],
) -> None:
    """Reject handoffs that require source and destination to share a process."""
    invalid_edges = sorted(cross_process_edges & type(config).process_local_edges())
    if invalid_edges:
        rendered = ", ".join(
            f"{source!r} -> {target!r}" for source, target in invalid_edges
        )
        raise ValueError(
            f"Cross-process edge(s) require source and destination to share a "
            f"process: {rendered}. Give the stages the same StageConfig.process"
        )


def build_process_topology_plan(
    config: PipelineConfig,
    gpu_placement: StagePlacementPlan,
    *,
    stages_cfg: list[StageConfig],
) -> ProcessTopologyPlan:
    groups = _build_process_groups(stages_cfg, gpu_placement)
    tp_stage_to_processes = _build_tp_process_names(stages_cfg)

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
    _validate_gpu_process_colocation(
        config,
        gpu_placement,
        stages_cfg,
        plan,
    )
    return plan


def _build_process_groups(
    stages: list[StageConfig],
    gpu_placement: StagePlacementPlan,
) -> list[ProcessGroupPlacement]:
    non_tp_stages = [stage for stage in stages if stage.tp_size == 1]

    components: OrderedDict[str, list[StageConfig]] = OrderedDict()
    for stage in non_tp_stages:
        components.setdefault(stage_process_name(stage), []).append(stage)

    return [
        ProcessGroupPlacement(
            name=group_name,
            stage_names=tuple(stage.name for stage in component),
            gpu_id=_resolve_group_gpu_id(group_name, component, gpu_placement),
        )
        for group_name, component in components.items()
    ]


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
            f"TP-derived process names collide with non-TP process groups: {collisions}"
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


def _render_missing_fraction_stages(
    stage_names: set[str],
    gpu_placement: StagePlacementPlan,
) -> str:
    """Name the stages an operator must edit, not just the expanded instances.

    Note (kaige): fractions are declared on the logical stage in YAML, so an
    error that only names talker_ar@r0 does not point at anything editable.
    """
    instance_to_logical = {
        instance: logical
        for logical, instances in gpu_placement.replica_instances.items()
        for instance in instances
    }
    rendered = []
    for name in sorted(stage_names):
        logical = instance_to_logical.get(name)
        rendered.append(name if logical is None else f"{logical} (instance {name})")
    return ", ".join(rendered)


def _stage_lacks_declared_memory_budget(stage: StageConfig) -> bool:
    """True when a stage does not declare its total GPU footprint.

    Colocation needs each stage on a shared GPU to declare how much of the card
    it intends to use in total. ``gpu_memory_fraction`` states that directly;
    on the byte path only ``total_reserve_bytes`` does, because
    ``engine.kv_cache_bytes`` covers the KV pool alone and says nothing about
    weights, activations, or graphs.
    """
    return stage.gpu_memory_fraction is None and stage.total_reserve_bytes is None


def _validate_gpu_process_colocation(
    config: PipelineConfig,
    gpu_placement: StagePlacementPlan,
    stages: list[StageConfig],
    topology_plan: ProcessTopologyPlan,
) -> None:
    stage_by_name = {stage.name: stage for stage in stages}
    gpu_processes: dict[int, set[str]] = defaultdict(set)
    missing_fraction: dict[int, set[str]] = defaultdict(set)
    logical_stage_by_name = {stage.name: stage for stage in config.stages}
    instance_to_logical = {
        instance: logical
        for logical, instances in gpu_placement.replica_instances.items()
        for instance in instances
    }
    replica_device_stage_names = set()
    for stage in stages:
        logical_name = instance_to_logical.get(stage.name, stage.name)
        logical_stage = logical_stage_by_name.get(logical_name)
        if logical_stage is None:
            continue
        process_name = stage_process_name(logical_stage)
        process_config = config.processes.get(process_name)
        if process_config is not None and process_config.replica_devices is not None:
            replica_device_stage_names.add(stage.name)
    replica_gpus: set[int] = set()

    def record(gpu_id: int, process_name: str, stage: StageConfig) -> None:
        gpu_processes[gpu_id].add(process_name)
        if stage.name in replica_device_stage_names:
            replica_gpus.add(gpu_id)
        if _stage_lacks_declared_memory_budget(stage):
            missing_fraction[gpu_id].add(stage.name)

    for group in topology_plan.groups:
        for stage_name in group.stage_names:
            stage = stage_by_name[stage_name]
            for gpu_id in _stage_gpu_ids(gpu_placement, stage):
                if gpu_id is None:
                    continue
                record(gpu_id, group.name, stage)

    for stage in stages:
        if stage.tp_size <= 1:
            continue
        for rank, gpu_id in enumerate(_stage_gpu_ids(gpu_placement, stage)):
            if gpu_id is None:
                continue
            record(gpu_id, topology_plan.tp_stage_to_processes[stage.name][rank], stage)

    require = config.placement.require_memory_fraction_for_colocation
    limit = config.placement.max_total_gpu_memory_fraction_per_gpu
    for gpu_id, process_names in gpu_processes.items():
        if len(process_names) <= 1:
            continue
        missing = missing_fraction.get(gpu_id, set())
        if (require or gpu_id in replica_gpus) and missing:
            sharing = (
                "has replica-induced GPU sharing"
                if gpu_id in replica_gpus
                else "is shared by multiple process groups"
            )
            raise ValueError(
                f"GPU {gpu_id} {sharing} without a declared total "
                "footprint (gpu_memory_fraction or total_reserve_bytes): "
                f"{_render_missing_fraction_stages(missing, gpu_placement)}"
            )
        total = gpu_placement.gpus[gpu_id].total_gpu_memory_fraction
        if total > limit + 1e-9:
            raise ValueError(
                f"GPU {gpu_id} total_gpu_memory_fraction={total:.3f} exceeds "
                f"placement limit {limit:.3f}"
            )
