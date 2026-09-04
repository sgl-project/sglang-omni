# SPDX-License-Identifier: Apache-2.0
"""Stage placement planning and validation for Omni pipelines."""

from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.utils.imports import import_string


@dataclass(frozen=True)
class StagePlacement:
    stage_name: str
    gpu_ids: tuple[int, ...]
    tp_size: int
    total_gpu_memory_fraction: float | None
    kv_cache_bytes: int | None = None
    total_reserve_bytes: int | None = None


@dataclass(frozen=True)
class GpuPlacement:
    gpu_id: int
    stage_names: tuple[str, ...]
    total_gpu_memory_fraction: float
    has_memory_fraction: bool
    missing_fraction_stage_names: tuple[str, ...]
    total_kv_cache_bytes: int
    total_reserve_bytes: int


@dataclass(frozen=True)
class StagePlacementPlan:
    stages: dict[str, StagePlacement]
    gpus: dict[int, GpuPlacement]
    replica_instances: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def instances_of(self, logical_name: str) -> list[StagePlacement]:
        """Placements of every replica instance behind *logical_name*.

        Unreplicated stages resolve to their own placement; CPU-only stages
        (absent from the plan) resolve to an empty list.
        """
        names = self.replica_instances.get(logical_name, (logical_name,))
        return [self.stages[name] for name in names if name in self.stages]


class PlacementPolicy(Protocol):
    def validate(self, config: PipelineConfig, plan: StagePlacementPlan) -> None: ...


class StagePlacementPlanner:
    """Build a model-agnostic placement plan from pipeline stage config."""

    def __init__(self, config: PipelineConfig):
        self._config = config

    def build(
        self,
        *,
        stages_cfg: list[StageConfig] | None = None,
        apply_policy: bool = True,
        replica_instances: dict[str, tuple[str, ...]] | None = None,
    ) -> StagePlacementPlan:
        stages = stages_cfg if stages_cfg is not None else self._config.stages
        placements: dict[str, StagePlacement] = {}
        gpu_entries: dict[int, list[StagePlacement]] = defaultdict(list)

        for stage in stages:
            gpu_ids = _resolve_stage_gpu_ids(stage)
            if not gpu_ids:
                continue

            kv_cache_bytes = (
                stage.engine.kv_cache_bytes if stage.engine is not None else None
            )
            placement = StagePlacement(
                stage_name=stage.name,
                gpu_ids=gpu_ids,
                tp_size=stage.tp_size,
                total_gpu_memory_fraction=stage.gpu_memory_fraction,
                kv_cache_bytes=kv_cache_bytes,
                total_reserve_bytes=stage.total_reserve_bytes,
            )
            placements[stage.name] = placement
            for gpu_id in gpu_ids:
                gpu_entries[gpu_id].append(placement)

        gpu_plans = {
            gpu_id: _build_gpu_placement(gpu_id, entries)
            for gpu_id, entries in gpu_entries.items()
        }
        plan = StagePlacementPlan(
            stages=placements,
            gpus=gpu_plans,
            replica_instances=dict(replica_instances or {}),
        )
        self._validate_memory_budgets(plan)
        if apply_policy:
            _apply_placement_policy(self._config, plan)
        return plan

    def _validate_memory_budgets(self, plan: StagePlacementPlan) -> None:
        limit = self._config.placement.max_total_gpu_memory_fraction_per_gpu
        for gpu in plan.gpus.values():
            if gpu.total_gpu_memory_fraction > limit + 1e-9:
                raise ValueError(
                    f"GPU {gpu.gpu_id} total_gpu_memory_fraction="
                    f"{gpu.total_gpu_memory_fraction:.3f} exceeds placement limit "
                    f"{limit:.3f}"
                )


def validate_gpu_capacity(plan: StagePlacementPlan) -> None:
    """Fail before spawn when declared budgets exceed a GPU's physical memory.

    Fractions and byte reserves are combined in the byte domain, so mixed
    fraction/byte colocation cannot overcommit a card while each domain's own
    sum still looks fine. The KV pools summed over every stage instance on a
    card, replica instances included, are a hard lower bound on their own, so
    they are checked even when no stage declares a total reserve. Skipped when
    device metadata is unavailable.
    """

    from sglang_omni.utils.gpu_memory import format_bytes_gib, get_gpu_device_info

    for gpu_id, gpu in sorted(plan.gpus.items()):
        if gpu.total_reserve_bytes <= 0 and gpu.total_kv_cache_bytes <= 0:
            continue
        info = get_gpu_device_info(gpu_id)
        if info.total_memory_bytes is None:
            continue
        declared = (
            gpu.total_gpu_memory_fraction * info.total_memory_bytes
            + gpu.total_reserve_bytes
        )
        if gpu.total_reserve_bytes > 0 and declared > info.total_memory_bytes:
            raise ValueError(
                f"GPU {gpu_id} declared budgets exceed physical memory: "
                f"fraction {gpu.total_gpu_memory_fraction:.3f} of "
                f"{format_bytes_gib(info.total_memory_bytes)} plus "
                f"total_reserve_bytes {format_bytes_gib(gpu.total_reserve_bytes)} "
                f"= {format_bytes_gib(int(declared))}. Lower the stage budgets "
                "or move a stage to another GPU."
            )
        if gpu.total_kv_cache_bytes > info.total_memory_bytes:
            kv_stages = sorted(
                placement.stage_name
                for placement in plan.stages.values()
                if placement.kv_cache_bytes and gpu_id in placement.gpu_ids
            )
            raise ValueError(
                f"GPU {gpu_id} declared KV pools alone exceed physical memory: "
                f"{format_bytes_gib(info.total_memory_bytes)} of VRAM against "
                f"{format_bytes_gib(gpu.total_kv_cache_bytes)} summed over "
                f"engine.kv_cache_bytes of {', '.join(kv_stages)}. Lower "
                "engine.kv_cache_bytes or reduce the replica count."
            )


def build_stage_placement_plan(
    config: PipelineConfig,
    *,
    stages_cfg: list[StageConfig] | None = None,
    apply_policy: bool = True,
    replica_instances: dict[str, tuple[str, ...]] | None = None,
) -> StagePlacementPlan:
    return StagePlacementPlanner(config).build(
        stages_cfg=stages_cfg,
        apply_policy=apply_policy,
        replica_instances=replica_instances,
    )


def resolve_stage_gpu_ids(
    plan: StagePlacementPlan,
    stage_cfg: StageConfig,
) -> list[int | None]:
    placement = plan.stages.get(stage_cfg.name)
    if placement is None:
        return [None] * stage_cfg.tp_size
    return list(placement.gpu_ids)


def resolve_gpu_stage_names(plan: StagePlacementPlan) -> set[str]:
    """Names of all GPU-resident stages.

    The placement planner only records stages that resolve to a GPU (CPU-only
    stages are skipped), so the plan's stage keys are exactly the GPU stages.
    The transport router uses this to decide CUDA-IPC vs SHM per edge.
    """
    return set(plan.stages.keys())


def _resolve_stage_gpu_ids(stage: StageConfig) -> tuple[int, ...]:
    # Shape rules (scalar vs list, length == tp_size, unique ids) are
    # enforced by StageConfig validation, but launcher helpers mutate
    # tp_size and gpu after construction, so re-check the TP shape here
    # rather than expanding a scalar into duplicate ranks.
    gpu = stage.gpu
    if gpu is None:
        return ()
    if isinstance(gpu, int):
        if stage.tp_size > 1:
            raise ValueError(
                f"Stage {stage.name!r}: TP placement requires a list of "
                f"{stage.tp_size} unique GPU ids, got scalar gpu={gpu}"
            )
        return (gpu,)
    gpu_ids = tuple(int(gpu_id) for gpu_id in gpu)
    if len(gpu_ids) != stage.tp_size or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(
            f"Stage {stage.name!r}: TP placement requires a list of "
            f"{stage.tp_size} unique GPU ids, got gpu={list(gpu)}"
        )
    return gpu_ids


def _build_gpu_placement(
    gpu_id: int,
    entries: list[StagePlacement],
) -> GpuPlacement:
    total = 0.0
    total_kv_cache_bytes = 0
    total_reserve_bytes = 0
    has_memory_fraction = False
    missing: set[str] = set()
    stage_names: list[str] = []
    for entry in entries:
        stage_names.append(entry.stage_name)
        if entry.total_gpu_memory_fraction is None:
            missing.add(entry.stage_name)
        else:
            has_memory_fraction = True
            total += entry.total_gpu_memory_fraction

        if entry.kv_cache_bytes is not None:
            total_kv_cache_bytes += entry.kv_cache_bytes
        if entry.total_reserve_bytes is not None:
            total_reserve_bytes += entry.total_reserve_bytes
    return GpuPlacement(
        gpu_id=gpu_id,
        stage_names=tuple(stage_names),
        total_gpu_memory_fraction=total,
        has_memory_fraction=has_memory_fraction,
        missing_fraction_stage_names=tuple(sorted(missing)),
        total_kv_cache_bytes=total_kv_cache_bytes,
        total_reserve_bytes=total_reserve_bytes,
    )


def _apply_placement_policy(
    config: PipelineConfig,
    plan: StagePlacementPlan,
) -> None:
    if config.placement_policy is None:
        return
    policy = import_string(config.placement_policy)
    if inspect.isclass(policy):
        policy = policy()
    if hasattr(policy, "validate"):
        policy.validate(config, plan)
        return
    if callable(policy):
        policy(config, plan)
        return
    raise TypeError(
        f"placement_policy {config.placement_policy!r} must be callable or expose "
        "validate(config, plan)"
    )
