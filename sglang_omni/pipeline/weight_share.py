# SPDX-License-Identifier: Apache-2.0
"""Runtime-owned leader/follower assignment for CUDA IPC weight sharing.

Replicas of one logical Process that land on the same physical GPU are a
sharing group: the lowest replica index loads the checkpoint and publishes
CUDA-IPC handles, the others alias them. This module only plans; the mechanism
itself lives in :mod:`sglang_omni.utils.ipc_weights` and is driven by the same
environment contract an external supervisor would use.

Planning happens in the parent, before the coordinator binds and before any
child is spawned, so an unusable configuration fails in milliseconds instead of
after a leader has loaded a full checkpoint.
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from sglang_omni.config.runtime import (
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.config.schema import PipelineConfig, replica_instance_name
from sglang_omni.config.topology import LogicalProcess, LogicalProcessPlan
from sglang_omni.mps.decision import process_gpu_ids
from sglang_omni.utils.ipc_weights import (
    ENV_WEIGHT_SHARE,
    ENV_WEIGHT_SHARE_COMPAT,
    ENV_WEIGHT_SHARE_RUN_ID,
)

logger = logging.getLogger(__name__)

__all__ = [
    "WeightShareError",
    "WeightShareGroup",
    "WeightSharePlan",
    "plan_weight_share",
]


class WeightShareError(ValueError):
    """Weight sharing was requested but cannot be planned for this pipeline."""


@dataclass(frozen=True)
class WeightShareGroup:
    """One leader plus its followers, all on one physical GPU."""

    logical_process: str
    gpu_id: int
    leader: str
    followers: tuple[str, ...]
    store_dir: Path


@dataclass(frozen=True)
class WeightSharePlan:
    run_id: str
    groups: tuple[WeightShareGroup, ...]
    env_by_process: dict[str, dict[str, str]]
    follower_process_names: frozenset[str]


def plan_weight_share(
    config: PipelineConfig,
    *,
    logical_process_plan: LogicalProcessPlan,
    process_specs,
    runtime_dir: Path,
) -> WeightSharePlan | None:
    """Assign weight-share roles, or return None when sharing is off."""

    if config.weight_share == "off":
        return None

    if os.name != "posix":
        raise WeightShareError(
            "weight_share=on requires a POSIX host: the handle store relies on "
            "flock leases and owner-only directory permissions"
        )

    _reject_external_env(process_specs)

    gpu_ids_by_process = {
        spec.process_name: process_gpu_ids(spec) for spec in process_specs
    }
    candidates = _collect_candidate_groups(logical_process_plan, gpu_ids_by_process)
    if not candidates:
        raise WeightShareError(
            "weight_share=on but no logical Process places two or more replicas "
            "on one GPU; declare processes.<name>.num_replicas with repeated "
            "replica_devices entries, or use weight_share=off"
        )

    for logical_process, _, _ in candidates:
        _validate_sharing_process(config, logical_process)

    run_id = secrets.token_hex(8)
    groups: list[WeightShareGroup] = []
    env_by_process: dict[str, dict[str, str]] = {}
    for logical_process, gpu_id, replica_ids in candidates:
        store_dir = _create_store_dir(runtime_dir, logical_process.name, gpu_id)
        members = [
            replica_instance_name(logical_process.name, replica_id)
            for replica_id in replica_ids
        ]
        leader, *followers = members
        for index, process_name in enumerate(members):
            role = "leader" if index == 0 else "follower"
            env_by_process[process_name] = {
                ENV_WEIGHT_SHARE: f"{role}:{store_dir}",
                ENV_WEIGHT_SHARE_RUN_ID: run_id,
            }
        groups.append(
            WeightShareGroup(
                logical_process=logical_process.name,
                gpu_id=gpu_id,
                leader=leader,
                followers=tuple(followers),
                store_dir=store_dir,
            )
        )
        logger.info(
            "Weight sharing on GPU %d for process %r: leader=%s followers=%s",
            gpu_id,
            logical_process.name,
            leader,
            list(followers),
        )

    # Note (Jiaxin Deng): a stage that never shares still unpickles CUDA tensors
    # reduced by a sharing engine, and the reduction carries a device UUID only
    # the patched path understands, so the compat flag covers every process.
    for spec in process_specs:
        env_by_process.setdefault(spec.process_name, {})[ENV_WEIGHT_SHARE_COMPAT] = "1"

    return WeightSharePlan(
        run_id=run_id,
        groups=tuple(groups),
        env_by_process=env_by_process,
        follower_process_names=frozenset(
            follower for group in groups for follower in group.followers
        ),
    )


def _reject_external_env(process_specs) -> None:
    """Refuse to plan on top of a supervisor that already assigned roles."""

    external = (os.environ.get(ENV_WEIGHT_SHARE) or "").strip()
    if external:
        raise WeightShareError(
            f"weight_share=on but the parent environment already sets "
            f"{ENV_WEIGHT_SHARE}={external!r}; the runtime assigns roles itself, "
            "so unset it or use weight_share=off with the external supervisor"
        )
    for spec in process_specs:
        for stage_spec in spec.stage_specs:
            if (stage_spec.env_defaults or {}).get(ENV_WEIGHT_SHARE):
                raise WeightShareError(
                    f"stage {stage_spec.stage_name!r} sets {ENV_WEIGHT_SHARE} in "
                    "its environment defaults; the runtime assigns weight-share "
                    "roles itself, so remove it"
                )


def _collect_candidate_groups(
    logical_process_plan: LogicalProcessPlan,
    gpu_ids_by_process: dict[str, set[int]],
) -> list[tuple[LogicalProcess, int, tuple[int, ...]]]:
    """Find replica sets of one Process that resolve to a single shared GPU.

    Replica identity comes from the logical plan, never from the OS process
    name: a replicated TP process is named ``P@rN_tp<rank>``, which no longer
    parses as a replica instance.
    """
    candidates: list[tuple[LogicalProcess, int, tuple[int, ...]]] = []
    for process in logical_process_plan.processes:
        if not process.is_replicated:
            continue
        if process.is_tensor_parallel:
            logger.info(
                "Weight sharing skips tensor-parallel process %r: CUDA IPC "
                "handles are not rank qualified",
                process.name,
            )
            continue
        by_gpu: dict[int, list[int]] = {}
        for replica_id in range(process.num_replicas):
            process_name = replica_instance_name(process.name, replica_id)
            gpu_ids = gpu_ids_by_process.get(process_name, set())
            if len(gpu_ids) != 1:
                continue
            by_gpu.setdefault(next(iter(gpu_ids)), []).append(replica_id)
        for gpu_id, replica_ids in sorted(by_gpu.items()):
            if len(replica_ids) < 2:
                continue
            candidates.append((process, gpu_id, tuple(sorted(replica_ids))))
        if not by_gpu:
            logger.info(
                "Weight sharing skips process %r: no replica resolves to a "
                "single GPU",
                process.name,
            )
    return candidates


def _validate_sharing_process(
    config: PipelineConfig,
    process: LogicalProcess,
) -> None:
    config_cls = type(config)
    engine_stages = [
        stage_name
        for stage_name in process.stage_names
        if config_cls.stage_config_cls(stage_name).engine_stage
    ]
    if len(engine_stages) != 1:
        raise WeightShareError(
            f"weight sharing needs exactly one SGLang engine stage in process "
            f"{process.name!r}, found {sorted(engine_stages)}"
        )
    stage_name = engine_stages[0]
    stage = next(stage for stage in config.stages if stage.name == stage_name)
    # Note (Jiaxin Deng): a follower frees its dummy weights before KV
    # profiling, so an underived cap would over-budget KV. The child enforces
    # this too, but only after the leader has loaded a whole checkpoint.
    if _resolved_max_total_tokens(config, stage) is None:
        raise WeightShareError(
            f"weight sharing requires an explicit max_total_tokens on engine "
            f"stage {stage_name!r} (set stages.{stage_name}.engine."
            "max_total_tokens): a follower attaches after its dummy weights are "
            "freed, so memory profiling cannot derive a stable KV budget"
        )


def _resolved_max_total_tokens(config: PipelineConfig, stage) -> int | None:
    # Both kwarg channels can carry server args; the stage's own engine block
    # outranks stage_factory_kwargs, matching how the worker overlays them.
    overrides = dict(
        resolve_stage_factory_kwargs(stage, config).get("server_args_overrides") or {}
    )
    overrides.update(
        resolve_stage_typed_kwargs(stage).get("server_args_overrides") or {}
    )
    value = overrides.get("max_total_tokens")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WeightShareError(
            f"stage {stage.name!r} must define a positive integer "
            f"max_total_tokens for weight sharing, got {value!r}"
        )
    return value


def _create_store_dir(runtime_dir: Path, process_name: str, gpu_id: int) -> Path:
    store_dir = Path(runtime_dir) / "weights" / process_name / f"gpu{gpu_id}"
    store_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    return store_dir
