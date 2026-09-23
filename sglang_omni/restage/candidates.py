"""Materialize a residency shape through the serving config schema."""

from collections.abc import Mapping, Sequence

from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.config.schema import PipelineConfig, ProcessConfig
from sglang_omni.config.topology import (
    build_process_topology_plan,
    compile_logical_processes,
)
from sglang_omni.pipeline.replicas import expand_replica_stages


def materialize_candidate(
    config: PipelineConfig,
    assignments: Mapping[str, Sequence[Sequence[int]]],
    devices: Sequence[int],
) -> PipelineConfig:
    """Copy a config with one TP-rank device tuple per process replica.

    Device IDs are in the server's visible-device namespace. Assign every
    GPU process; CPU-only processes retain their existing policy. This checks
    declared runtime constraints, not measured memory use or performance.
    """
    logical, stages = compile_logical_processes(config)
    gpu_stages = {stage.name for stage in stages if stage.gpu is not None}
    gpu_processes = {
        process.name: process
        for process in logical.processes
        if gpu_stages.intersection(process.stage_names)
    }
    if set(assignments) != set(gpu_processes):
        raise ValueError(
            f"Assignments must cover exactly the GPU processes: {sorted(gpu_processes)}"
        )
    budget = set(devices)
    result = config.model_copy(deep=True)
    for name, replicas in assignments.items():
        process = gpu_processes[name]
        if not replicas:
            raise ValueError(f"Process {name!r} needs at least one replica")
        for ranks in replicas:
            if len(ranks) != process.tp_size:
                raise ValueError(
                    f"Process {name!r} needs tp_size={process.tp_size} ranks"
                )
            if not set(ranks) <= budget:
                raise ValueError(
                    f"Process {name!r} uses devices outside the GPU budget"
                )
        result.processes[name] = ProcessConfig(
            num_replicas=len(replicas),
            replica_devices=(
                [device for ranks in replicas for device in ranks]
                if len(replicas) > 1
                else None
            ),
        )
        for stage in result.stages:
            if stage.name in process.stage_names and stage.gpu is not None:
                stage.gpu = list(replicas[0]) if stage.tp_size > 1 else replicas[0][0]
    result = type(config).model_validate(result.model_dump())
    # Note (Jiaxin Deng): expand before placement checks so every replica contributes
    # to the same memory and process constraints used by serving startup.
    plan, stage_copies = compile_logical_processes(result)
    expanded, topology = expand_replica_stages(stage_copies, plan)
    placement = build_stage_placement_plan(
        result, stages_cfg=expanded, replica_instances=topology.replicas
    )
    build_process_topology_plan(result, placement, stages_cfg=expanded)
    return result
