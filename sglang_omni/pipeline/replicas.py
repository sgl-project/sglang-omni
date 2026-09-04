# SPDX-License-Identifier: Apache-2.0
"""Process-level replicas.

A logical Process with ``num_replicas = N`` is copied whole: every member stage
gets N instances, and instances with the same index form one complete Process
replica. Expansion runs after the logical Process plan is compiled, so it never
changes Process membership.

Model code, route hooks, and placement policies keep referring to the logical
stage name; the runtime resolves ``(logical name, request binding) -> instance``
at every send site. Bindings are chosen once per replicated Process at
admission, projected onto that Process's member stages, and propagated on the
message envelope.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from sglang_omni.config.schema import (
    REPLICA_SEPARATOR,
    StageConfig,
    parse_replica_instance_name,
    replica_instance_name,
)
from sglang_omni.config.topology import LogicalProcessPlan

logger = logging.getLogger(__name__)

__all__ = [
    "REPLICA_SEPARATOR",
    "replica_instance_name",
    "parse_replica_instance_name",
    "ReplicaTopology",
    "expand_replica_stages",
    "BindingPolicy",
    "RoundRobinBindingPolicy",
    "assign_replica_bindings",
]


@dataclass(frozen=True)
class ReplicaTopology:

    replicas: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.replicas)

    def is_replicated(self, logical_name: str) -> bool:
        return logical_name in self.replicas

    def num_replicas(self, logical_name: str) -> int:
        instances = self.replicas.get(logical_name)
        return len(instances) if instances is not None else 1

    def instances(self, logical_name: str) -> tuple[str, ...]:
        instances = self.replicas.get(logical_name)
        if instances is None:
            return (logical_name,)
        return instances

    def resolve(self, logical_name: str, replica_id: int) -> str:
        instances = self.replicas.get(logical_name)
        if instances is None:
            if replica_id != 0:
                raise ValueError(
                    f"Stage {logical_name!r} is not replicated; got replica_id="
                    f"{replica_id}"
                )
            return logical_name
        if not 0 <= replica_id < len(instances):
            raise ValueError(
                f"Stage {logical_name!r} has {len(instances)} replicas; got "
                f"replica_id={replica_id}"
            )
        return instances[replica_id]

    def logical_name(self, name: str) -> str:
        logical, replica_id = parse_replica_instance_name(name)
        if replica_id is None:
            return name
        instances = self.replicas.get(logical)
        if instances is None or name not in instances:
            return name
        return logical

    def replicated_logical_names(self) -> list[str]:
        return sorted(self.replicas)

    def to_dict(self) -> dict[str, list[str]]:
        return {name: list(instances) for name, instances in self.replicas.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ReplicaTopology":
        if not data:
            return cls()
        return cls(
            replicas={
                name: tuple(str(i) for i in instances)
                for name, instances in data.items()
            }
        )


def expand_replica_stages(
    stages_cfg: list[StageConfig],
    plan: LogicalProcessPlan,
) -> tuple[list[StageConfig], ReplicaTopology]:
    """Expand replicated logical Processes into per-replica instance stages.

    Every member stage of a Process with N replicas is copied N times, and
    copies sharing an index form one complete Process replica ``P@rN``. Instance
    stages keep logical names in their wiring (``next``, ``stream_to``,
    ``wait_for``, ``project_payload``); resolution to a concrete instance
    happens at send time from the request's bindings.
    """
    replicas: dict[str, tuple[str, ...]] = {}
    expanded_by_stage: dict[str, list[StageConfig]] = {}

    for process in plan.processes:
        if not process.is_replicated:
            continue
        for stage_name in process.stage_names:
            replicas[stage_name] = tuple(
                replica_instance_name(stage_name, replica_id)
                for replica_id in range(process.num_replicas)
            )

    stage_by_name = {stage.name: stage for stage in stages_cfg}
    for process in plan.processes:
        if not process.is_replicated:
            # Note (kaige): replica_devices covers index 0 too, so a single
            # replica still overrides StageConfig.gpu. Names stay unsuffixed
            # and the process produces no replica bindings.
            if process.replica_devices is not None:
                for stage_name in process.stage_names:
                    stage_cfg = stage_by_name[stage_name]
                    expanded_by_stage[stage_name] = [
                        stage_cfg.model_copy(
                            update={
                                "gpu": _replica_stage_gpu(
                                    stage_cfg, process.replica_devices[0]
                                )
                            }
                        )
                    ]
            continue
        for replica_id in range(process.num_replicas):
            devices = (
                None
                if process.replica_devices is None
                else process.replica_devices[replica_id]
            )
            for stage_name in process.stage_names:
                stage_cfg = stage_by_name[stage_name]
                expanded_by_stage.setdefault(stage_name, []).append(
                    stage_cfg.model_copy(
                        update={
                            "name": replicas[stage_name][replica_id],
                            "gpu": _replica_stage_gpu(stage_cfg, devices),
                            "process": replica_instance_name(process.name, replica_id),
                        }
                    )
                )
        logger.info(
            "Replicated process %r x%d -> %s (replica_devices=%s)",
            process.name,
            process.num_replicas,
            [
                replica_instance_name(process.name, replica_id)
                for replica_id in range(process.num_replicas)
            ],
            process.replica_devices,
        )

    # Note (kaige): keep config order so stages that share an OS process are
    # still constructed in declaration order within each replica.
    expanded: list[StageConfig] = []
    for stage_cfg in stages_cfg:
        expanded.extend(expanded_by_stage.get(stage_cfg.name, [stage_cfg]))

    return expanded, ReplicaTopology(replicas=replicas)


def _replica_stage_gpu(
    stage_cfg: StageConfig,
    devices: tuple[int, ...] | None,
) -> int | list[int] | None:
    """Return the device placement one replica of *stage_cfg* runs on.

    ``replica_devices`` only covers GPU stages; a CPU stage in a mixed process
    stays on the host.
    """
    if devices is None or stage_cfg.gpu is None:
        return stage_cfg.gpu
    if stage_cfg.tp_size == 1:
        return devices[0]
    return list(devices)


def validate_device_assignment(
    stages_cfg: list[StageConfig],
    *,
    device_count: int | None,
) -> None:
    """Fail fast on GPU ids outside this host's visible devices.

    Ids are indices into the launcher's visible devices, so the caller must
    pass ``device_count`` from the launcher process. ``None`` skips the check
    when CUDA is unavailable or the device count is unknown.
    """
    if device_count is None:
        return

    for stage_cfg in stages_cfg:
        gpu = stage_cfg.gpu
        if gpu is None:
            continue
        ids = [gpu] if isinstance(gpu, int) else gpu
        for gpu_id in ids:
            if gpu_id >= device_count:
                raise ValueError(
                    f"Stage {stage_cfg.name!r}: GPU id {gpu_id} out of range; "
                    f"only {device_count} visible device(s)"
                )


class BindingPolicy(Protocol):
    """Selects a replica for one (logical Process, request) at admission."""

    def bind(self, process_name: str, num_replicas: int, request_id: str) -> int: ...


class RoundRobinBindingPolicy:
    """Per-process round-robin selection; thread-safe."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    def bind(self, process_name: str, num_replicas: int, request_id: str) -> int:
        del request_id
        with self._lock:
            index = self._counters.get(process_name, 0)
            self._counters[process_name] = index + 1
        return index % num_replicas


def assign_replica_bindings(
    plan: LogicalProcessPlan,
    policy: BindingPolicy,
    request_id: str,
) -> dict[str, int] | None:
    """Return stage-keyed replica bindings for one request.

    The policy is consulted once per replicated Process; the chosen index is
    then projected onto every member stage of that Process.
    """
    if not plan.has_replicas():
        return None

    bindings: dict[str, int] = {}
    for process in plan.processes:
        if not process.is_replicated:
            continue
        index = policy.bind(process.name, process.num_replicas, request_id)
        if not 0 <= index < process.num_replicas:
            raise ValueError(
                f"Binding policy selected replica {index} for process "
                f"{process.name!r} with {process.num_replicas} replica(s)"
            )
        for stage_name in process.stage_names:
            bindings[stage_name] = index
    return bindings
