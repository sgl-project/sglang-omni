# SPDX-License-Identifier: Apache-2.0
"""Runtime-level stage replicas.

A logical stage with ``num_replicas = N`` is expanded into N replica instance
stages before placement/endpoint allocation. Model code, route hooks, and
placement policies keep referring to the logical stage name; the runtime
resolves ``(logical name, request binding) -> instance`` at every send site.
Bindings are assigned once per request by the coordinator at admission and
propagate on the message envelope.
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

logger = logging.getLogger(__name__)

__all__ = [
    "REPLICA_SEPARATOR",
    "replica_instance_name",
    "parse_replica_instance_name",
    "ReplicaTopology",
    "split_replica_devices",
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

    def resolve_bound(
        self,
        logical_name: str,
        bindings: dict[str, int] | None,
    ) -> str:
        if not self.is_replicated(logical_name):
            return logical_name
        replica_id = None if bindings is None else bindings.get(logical_name)
        if replica_id is None:
            raise ValueError(
                f"no replica binding for replicated stage {logical_name!r}"
            )
        return self.resolve(logical_name, replica_id)

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


def split_replica_devices(
    replica_devices: str | list[int] | int | None,
    *,
    stage_name: str,
    num_replicas: int,
    tp_size: int,
) -> list[list[int] | None]:
    if replica_devices is None:
        return [None] * num_replicas

    if isinstance(replica_devices, str):
        ids = [int(part) for part in replica_devices.split(",") if part.strip()]
    elif isinstance(replica_devices, int):
        ids = [replica_devices]
    else:
        ids = [int(part) for part in replica_devices]

    if len(ids) == num_replicas * tp_size:
        return [ids[r * tp_size : (r + 1) * tp_size] for r in range(num_replicas)]
    raise ValueError(
        f"Stage {stage_name!r}: replica_devices has {len(ids)} id(s); expected "
        f"{num_replicas * tp_size} (one per replica x tp rank) for "
        f"num_replicas={num_replicas}, tp_size={tp_size}"
    )


def expand_replica_stages(
    stages_cfg: list[StageConfig],
) -> tuple[list[StageConfig], ReplicaTopology]:
    """Expand replicated logical stages into per-replica instance stages.

    Instance stages keep logical names in their wiring (``next``,
    ``stream_to``, ``wait_for``, ``project_payload``); resolution to a
    concrete instance happens at send time from the request's bindings.
    """
    replicas: dict[str, tuple[str, ...]] = {}
    expanded: list[StageConfig] = []

    for stage_cfg in stages_cfg:
        if stage_cfg.num_replicas <= 1:
            expanded.append(stage_cfg)
            continue

        if stage_cfg.replica_devices is None and stage_cfg.gpu is not None:
            raise ValueError(
                f"Stage {stage_cfg.name!r}: num_replicas="
                f"{stage_cfg.num_replicas} on a GPU stage requires explicit "
                f"replica_devices with "
                f"{stage_cfg.num_replicas * stage_cfg.tp_size} GPU id(s)"
            )
        per_replica_gpus = split_replica_devices(
            stage_cfg.replica_devices,
            stage_name=stage_cfg.name,
            num_replicas=stage_cfg.num_replicas,
            tp_size=stage_cfg.tp_size,
        )

        instance_names = []
        for replica_id in range(stage_cfg.num_replicas):
            instance = replica_instance_name(stage_cfg.name, replica_id)
            instance_names.append(instance)
            gpu_ids = per_replica_gpus[replica_id]
            gpu: int | list[int] | None
            if gpu_ids is None:
                gpu = None
            elif stage_cfg.tp_size == 1:
                gpu = gpu_ids[0]
            else:
                gpu = list(gpu_ids)
            process = stage_cfg.process
            if process is not None:
                process = f"{process}{REPLICA_SEPARATOR}{replica_id}"
            expanded.append(
                stage_cfg.model_copy(
                    update={
                        "name": instance,
                        "gpu": gpu,
                        "process": process,
                        "num_replicas": 1,
                        "replica_devices": None,
                    }
                )
            )
        replicas[stage_cfg.name] = tuple(instance_names)
        logger.info(
            "Replicated stage %r -> %s (replica_devices=%s)",
            stage_cfg.name,
            {
                instance: gpus
                for instance, gpus in zip(instance_names, per_replica_gpus)
            },
            stage_cfg.replica_devices,
        )

    return expanded, ReplicaTopology(replicas=replicas)


def validate_device_assignment(
    stages_cfg: list[StageConfig],
    *,
    device_count: int | None,
) -> None:
    """Fail fast on GPU ids that cannot exist on this host.

    Ids are indices into the launcher's visible devices, so the caller must
    pass ``device_count`` from the launcher process. ``None`` (CUDA
    unavailable or count unknown) skips the range check but still rejects
    negative and duplicate ids.
    """
    for stage_cfg in stages_cfg:
        gpu = stage_cfg.gpu
        if gpu is None:
            continue
        ids = [gpu] if isinstance(gpu, int) else [int(part) for part in gpu]
        for gpu_id in ids:
            if gpu_id < 0:
                raise ValueError(f"Stage {stage_cfg.name!r}: GPU id {gpu_id} is negative")
            if device_count is not None and gpu_id >= device_count:
                raise ValueError(
                    f"Stage {stage_cfg.name!r}: GPU id {gpu_id} out of range; "
                    f"only {device_count} visible device(s)"
                )
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"Stage {stage_cfg.name!r}: duplicate GPU id in tp group {ids}"
            )


class BindingPolicy(Protocol):
    """Selects a replica for one (logical stage, request) at admission."""

    def bind(self, logical_name: str, num_replicas: int, request_id: str) -> int: ...


class RoundRobinBindingPolicy:
    """Per-stage round-robin selection; thread-safe."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    def bind(self, logical_name: str, num_replicas: int, request_id: str) -> int:
        del request_id
        with self._lock:
            index = self._counters.get(logical_name, 0)
            self._counters[logical_name] = index + 1
        return index % num_replicas


def assign_replica_bindings(
    topology: ReplicaTopology,
    policy: BindingPolicy,
    request_id: str,
) -> dict[str, int] | None:
    """Return per-logical-stage replica bindings for one request."""
    if not topology:
        return None
    return {
        logical_name: policy.bind(
            logical_name, topology.num_replicas(logical_name), request_id
        )
        for logical_name in topology.replicated_logical_names()
    }
