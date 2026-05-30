# SPDX-License-Identifier: Apache-2.0
"""Stage placement planning and validation for Omni pipelines."""

from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from sglang_omni.config.runtime import reject_untyped_total_gpu_memory_fraction
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.utils.gpu_memory import format_bytes_gib, get_gpu_device_info
from sglang_omni.utils.imports import import_string

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StagePlacement:
    stage_name: str
    gpu_ids: tuple[int, ...]
    tp_size: int
    total_gpu_memory_fraction: float | None


@dataclass(frozen=True)
class GpuPlacement:
    gpu_id: int
    stage_names: tuple[str, ...]
    total_gpu_memory_fraction: float
    has_memory_fraction: bool
    missing_fraction_stage_names: tuple[str, ...]


@dataclass(frozen=True)
class StagePlacementPlan:
    stages: dict[str, StagePlacement]
    gpus: dict[int, GpuPlacement]
    same_gpu_stream_targets: dict[str, frozenset[str]]


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
    ) -> StagePlacementPlan:
        stages = stages_cfg if stages_cfg is not None else self._config.stages
        placements: dict[str, StagePlacement] = {}
        gpu_entries: dict[int, list[tuple[str, float | None]]] = defaultdict(list)

        for stage in stages:
            reject_untyped_total_gpu_memory_fraction(
                stage.name,
                stage.factory_args,
                self._config.runtime_overrides.get(stage.name, {}),
            )
            gpu_ids = _resolve_stage_gpu_ids(stage)
            if not gpu_ids:
                continue

            fraction = stage.runtime.resources.total_gpu_memory_fraction
            placements[stage.name] = StagePlacement(
                stage_name=stage.name,
                gpu_ids=gpu_ids,
                tp_size=stage.tp_size,
                total_gpu_memory_fraction=fraction,
            )
            for gpu_id in gpu_ids:
                gpu_entries[gpu_id].append((stage.name, fraction))

        gpu_plans = {
            gpu_id: _build_gpu_placement(gpu_id, entries)
            for gpu_id, entries in gpu_entries.items()
        }
        plan = StagePlacementPlan(
            stages=placements,
            gpus=gpu_plans,
            same_gpu_stream_targets=_build_same_gpu_stream_targets(
                stages,
                placements,
            ),
        )
        self._validate_memory_budgets(plan)
        self._validate_dynamic_headroom(stages, plan)
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

    def _validate_dynamic_headroom(
        self,
        stages: list[StageConfig],
        plan: StagePlacementPlan,
    ) -> None:
        dynamic_entries: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for stage in stages:
            budget = stage.runtime.resources.encoder_activation_budget_bytes
            if budget is None:
                continue
            for gpu_id in _resolve_stage_gpu_ids(stage):
                dynamic_entries[gpu_id].append((stage.name, int(budget)))

        if not dynamic_entries:
            return

        limit = self._config.placement.max_total_gpu_memory_fraction_per_gpu
        for gpu_id, entries in dynamic_entries.items():
            gpu = plan.gpus.get(gpu_id)
            if gpu is None:
                continue
            total_memory = get_gpu_device_info(gpu_id).total_memory_bytes
            if total_memory is None:
                logger.info(
                    "Skipping dynamic GPU headroom validation for GPU %s: "
                    "total memory is unavailable",
                    gpu_id,
                )
                continue

            resident_bytes = int(total_memory * gpu.total_gpu_memory_fraction)
            dynamic_bytes = sum(value for _, value in entries)
            limit_bytes = int(total_memory * limit)
            if resident_bytes + dynamic_bytes <= limit_bytes:
                logger.info(
                    "GPU %s memory budget validated: resident=%s dynamic=%s "
                    "limit=%s",
                    gpu_id,
                    format_bytes_gib(resident_bytes),
                    format_bytes_gib(dynamic_bytes),
                    format_bytes_gib(limit_bytes),
                )
                continue

            detail = ", ".join(
                f"{stage_name}={format_bytes_gib(value)}"
                for stage_name, value in entries
            )
            raise ValueError(
                f"GPU {gpu_id} resident plus dynamic memory budgets exceed "
                "placement limit: "
                f"resident={format_bytes_gib(resident_bytes)} "
                f"dynamic={format_bytes_gib(dynamic_bytes)} ({detail}) "
                f"limit={format_bytes_gib(limit_bytes)}. Lower "
                "runtime.resources.total_gpu_memory_fraction or "
                "runtime.resources.encoder_activation_budget_bytes."
            )


def build_stage_placement_plan(
    config: PipelineConfig,
    *,
    stages_cfg: list[StageConfig] | None = None,
    apply_policy: bool = True,
) -> StagePlacementPlan:
    return StagePlacementPlanner(config).build(
        stages_cfg=stages_cfg,
        apply_policy=apply_policy,
    )


def resolve_stage_gpu_ids(
    plan: StagePlacementPlan,
    stage_cfg: StageConfig,
) -> list[int | None]:
    placement = plan.stages.get(stage_cfg.name)
    if placement is None:
        return [None] * stage_cfg.tp_size
    return list(placement.gpu_ids)


def resolve_same_gpu_stream_targets(
    plan: StagePlacementPlan,
    stage_cfg: StageConfig,
) -> set[str]:
    return set(plan.same_gpu_stream_targets.get(stage_cfg.name, frozenset()))


def _resolve_stage_gpu_ids(stage: StageConfig) -> tuple[int, ...]:
    gpu = stage.gpu
    if gpu is None:
        return ()
    if isinstance(gpu, int):
        if stage.tp_size > 1:
            raise ValueError(
                f"Stage {stage.name!r}: TP placement requires a list of "
                f"{stage.tp_size} unique GPU ids, got scalar gpu={gpu}"
            )
        return tuple(gpu for _ in range(stage.tp_size))
    if len(gpu) != stage.tp_size:
        raise ValueError(
            f"Stage {stage.name!r}: gpu has {len(gpu)} entries "
            f"but tp_size={stage.tp_size}"
        )
    gpu_ids = tuple(int(gpu_id) for gpu_id in gpu)
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(
            f"Stage {stage.name!r}: TP placement requires unique GPU ids, "
            f"got {list(gpu_ids)}"
        )
    return gpu_ids


def _build_same_gpu_stream_targets(
    stages: list[StageConfig],
    placements: dict[str, StagePlacement],
) -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    for stage in stages:
        if not stage.stream_to:
            continue
        sender_gpu = _primary_gpu(stage.name, placements)
        if sender_gpu is None:
            continue
        same_gpu_targets = {
            target_name
            for target_name in stage.stream_to
            if _primary_gpu(target_name, placements) == sender_gpu
        }
        if same_gpu_targets:
            out[stage.name] = frozenset(same_gpu_targets)
    return out


def _primary_gpu(
    stage_name: str,
    placements: dict[str, StagePlacement],
) -> int | None:
    placement = placements.get(stage_name)
    if placement is None or not placement.gpu_ids:
        return None
    return placement.gpu_ids[0]


def _build_gpu_placement(
    gpu_id: int,
    entries: list[tuple[str, float | None]],
) -> GpuPlacement:
    total = 0.0
    has_memory_fraction = False
    missing: set[str] = set()
    stage_names: list[str] = []
    for stage_name, fraction in entries:
        stage_names.append(stage_name)
        if fraction is None:
            missing.add(stage_name)
            continue
        has_memory_fraction = True
        total += fraction
    return GpuPlacement(
        gpu_id=gpu_id,
        stage_names=tuple(stage_names),
        total_gpu_memory_fraction=total,
        has_memory_fraction=has_memory_fraction,
        missing_fraction_stage_names=tuple(sorted(missing)),
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
