"""Stage capacity arithmetic for residency planning.

A residency plan places every GPU process of a pipeline on a GPU budget. Its
utility is the aggregate flow F its busiest GPU still sustains:

    sum_{instances i on g}  F / (n_i * T_i)  <=  C_g      for every GPU g
    F  <=  n_p * T_p * min(1, pool_p / kappa_p)           for KV-holding p

with n_i replicas of the instance's process, T_i the throughput one instance
sustains inside the SLO regime, and C_g = 1 for a GPU serving one pipeline
flow or k * d(mode) for k competing flows under the measured sharing
discount. The pool factor scales down a process whose KV pool cannot hold
the kappa requests the SLO admits; kappa is the water level of

    c / T + (c + z99 * sqrt(c)) * delta / D = 1

where delta is the per-arrival stall and D the audio seconds per request.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Z99 = 2.326

# Note (Jiaxin Deng): measured on sglang-omni Qwen3-Omni tails (sgl-dm 20057386,
# matched c8); every other model reuses them as a prior until measured.
SHARING_DISCOUNT = {"dedicated": 1.0, "timeslice": 0.58, "mps": 0.85}
RUNTIME_MARGIN_GIB = 2.0


@dataclass(frozen=True)
class Workload:
    context_tokens: int
    audio_seconds: float
    slo_rtf: float = 1.0


@dataclass(frozen=True)
class StageCapacity:
    """One GPU process: throughput of one instance plus its memory geometry."""

    process: str
    throughput: float
    provenance: str
    weights_gib: float
    kv_bytes_per_token: int | None = None
    delta_s: float = 0.0
    tp_efficiency: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.throughput) or self.throughput <= 0:
            raise ValueError(f"Process {self.process!r} needs a positive throughput")
        if self.weights_gib < 0 or self.delta_s < 0:
            raise ValueError(f"Process {self.process!r} has negative geometry")
        if not 0 < self.tp_efficiency <= 1:
            raise ValueError("tp_efficiency must be in (0, 1]")


@dataclass(frozen=True)
class GpuLoad:
    device: int
    instances: tuple[str, ...]
    flows: int
    capacity: float
    bound: float


@dataclass(frozen=True)
class PlanUtility:
    utility: float
    binding: str
    gpus: tuple[GpuLoad, ...]
    pool_bounds: dict[str, float]
    provenance: str


def kappa_lower(throughput: float, delta_s: float, audio_seconds: float) -> float:
    """Smallest concurrency at which the SLO water level is reached."""
    c = 1.0
    while c < 256:
        if c / throughput + (c + Z99 * math.sqrt(c)) * delta_s / audio_seconds >= 1.0:
            return c
        c += 0.1
    return c


def pool_requests(
    capacity: StageCapacity,
    workload: Workload,
    fraction: float,
    gpu_mem_gib: float,
    *,
    tp_size: int = 1,
) -> float | None:
    """Requests of L tokens the KV pool holds at a memory fraction, or None."""
    if capacity.kv_bytes_per_token is None:
        return None
    pool_gib = fraction * gpu_mem_gib - capacity.weights_gib / tp_size
    if pool_gib <= 0:
        return 0.0
    return (
        pool_gib * (1 << 30) / (capacity.kv_bytes_per_token * workload.context_tokens)
    )


def footprint_gib(capacity: StageCapacity, workload: Workload) -> float:
    """Weights, runtime margin and a kappa-sized KV pool for one instance."""
    kv = 0.0
    if capacity.kv_bytes_per_token is not None:
        kappa = kappa_lower(
            capacity.throughput, capacity.delta_s, workload.audio_seconds
        )
        kv = kappa * workload.context_tokens * capacity.kv_bytes_per_token / (1 << 30)
    return capacity.weights_gib + RUNTIME_MARGIN_GIB + kv


def plan_utility(
    assignments: Mapping[str, Sequence[Sequence[int]]],
    flows: Mapping[str, Sequence[int]],
    capacities: Mapping[str, StageCapacity],
    workload: Workload,
    *,
    gpu_mem_gib: float,
    fractions: Mapping[str, float],
    sharing_mode: str = "timeslice",
) -> PlanUtility:
    """Aggregate throughput a placement sustains, bounded by its busiest GPU.

    A process with n instances spreads the flow F over them, so an instance on
    a GPU costs F / (n * T) of that GPU. Instances of one pipeline flow share a
    GPU serially (capacity 1); instances of k different flows retain k * d(mode)
    of a GPU, the measured sharing discount. A KV-holding process is further
    bounded by the requests its pool can hold against the kappa the SLO admits.
    ``fractions`` is the per-device memory fraction of each process instance.
    """
    if sharing_mode not in SHARING_DISCOUNT:
        raise ValueError(f"Unknown sharing mode {sharing_mode!r}")
    if set(assignments) != set(capacities) or set(flows) != set(assignments):
        raise ValueError(
            "Assignments, flows and capacities must name the same processes"
        )
    per_device: dict[int, list[tuple[str, int, int]]] = {}
    predicted = False
    for name, replicas in assignments.items():
        if not replicas or len(flows[name]) != len(replicas):
            raise ValueError(f"Process {name!r} needs one flow id per replica")
        capacity = capacities[name]
        if "PREDICTED" in capacity.provenance or "PRIOR" in capacity.provenance:
            predicted = True
        for ranks, flow in zip(replicas, flows[name]):
            if len(ranks) > 1:
                predicted = True
            for device in ranks:
                per_device.setdefault(device, []).append((name, flow, len(ranks)))
    gpus = []
    for device, instances in sorted(per_device.items()):
        distinct = len({flow for _, flow, _ in instances})
        capacity_units = (
            1.0 if distinct == 1 else distinct * SHARING_DISCOUNT[sharing_mode]
        )
        load = 0.0
        for name, _, tp_size in instances:
            capacity = capacities[name]
            throughput = capacity.throughput * (
                capacity.tp_efficiency if tp_size > 1 else 1.0
            )
            load += 1.0 / (len(assignments[name]) * throughput)
        gpus.append(
            GpuLoad(
                device=device,
                instances=tuple(name for name, _, _ in instances),
                flows=distinct,
                capacity=capacity_units,
                bound=capacity_units / load,
            )
        )
    pool_bounds = {}
    for name, replicas in assignments.items():
        capacity = capacities[name]
        pool = pool_requests(
            capacity, workload, fractions[name], gpu_mem_gib, tp_size=len(replicas[0])
        )
        if pool is None:
            continue
        kappa = kappa_lower(
            capacity.throughput, capacity.delta_s, workload.audio_seconds
        )
        pool_bounds[name] = len(replicas) * capacity.throughput * min(1.0, pool / kappa)
    binding_gpu = min(gpus, key=lambda row: row.bound)
    utility, binding = binding_gpu.bound, f"gpu{binding_gpu.device}"
    for name, bound in pool_bounds.items():
        if bound < utility:
            utility, binding = bound, f"pool:{name}"
    return PlanUtility(
        utility=utility,
        binding=binding,
        gpus=tuple(gpus),
        pool_bounds=pool_bounds,
        provenance="PREDICTED" if predicted else "MEASURED",
    )
