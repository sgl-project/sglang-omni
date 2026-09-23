"""Residency shapes: the placement families a planner ranks.

The backbone process holds the KV pool; every other GPU process is a tail.
Four families cover the measured wins so far: replicate the whole pipeline,
give the backbone its own GPUs and replicate the tails on dedicated GPUs,
consolidate the tails on one shared GPU, and tensor-parallel a backbone whose
weights do not fit one GPU. Each shape names replica device tuples per process
and the sharing mode of any GPU hosting more than one instance.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import count

USABLE_FRACTION = 0.9
SHARED_MODES = ("timeslice", "mps")


@dataclass(frozen=True)
class ProcessShape:
    name: str
    footprint_gib: float
    backbone: bool = False
    tp_size: int = 1


@dataclass(frozen=True)
class Shape:
    """Replica device tuples per process, plus the pipeline flow each serves.

    ``flows`` parallels ``assignments``: instances of different processes with
    the same flow id belong to one pipeline copy and share a GPU serially,
    while instances of different flows on one GPU compete under
    ``sharing_mode``.
    """

    name: str
    assignments: dict[str, tuple[tuple[int, ...], ...]]
    flows: dict[str, tuple[int, ...]]
    sharing_mode: str
    idle_devices: tuple[int, ...]

    @property
    def replicas(self) -> dict[str, int]:
        return {name: len(replicas) for name, replicas in self.assignments.items()}


def _fits(footprint, gpu_mem_gib):
    return footprint <= USABLE_FRACTION * gpu_mem_gib


def residency_shapes(
    processes: Sequence[ProcessShape],
    devices: Sequence[int],
    *,
    gpu_mem_gib: float,
    max_colocated: int = 4,
) -> list[Shape]:
    """Enumerate the shape families over a device budget, deduplicated.

    Idle GPUs are allowed; the ranking decides whether they pay. Shapes that
    cannot fit their instances by declared footprint are not generated.
    """
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Devices must be distinct")
    if gpu_mem_gib <= 0:
        raise ValueError("gpu_mem_gib must be positive")
    backbones = [p for p in processes if p.backbone]
    if len(backbones) != 1:
        raise ValueError("Exactly one backbone process is required")
    backbone = backbones[0]
    tails = [p for p in processes if not p.backbone]
    if any(p.tp_size > 1 for p in tails):
        raise ValueError("Tail processes must not be tensor parallel")
    devices = tuple(devices)
    budget = len(devices)
    tails_footprint = sum(p.footprint_gib for p in tails)
    pipeline_footprint = backbone.footprint_gib + tails_footprint
    shapes: list[Shape] = []

    def emit(name, parts, mode, used):
        assignments: dict[str, tuple[tuple[int, ...], ...]] = {}
        flows: dict[str, tuple[int, ...]] = {}
        for flow, part in enumerate(parts):
            for proc, replicas in part.items():
                assignments[proc] = assignments.get(proc, ()) + tuple(replicas)
                flows[proc] = flows.get(proc, ()) + (flow,) * len(replicas)
        idle = tuple(d for d in devices if d not in used)
        shape = Shape(name, assignments, flows, mode, idle)
        if all(
            s.assignments != shape.assignments or s.sharing_mode != mode for s in shapes
        ):
            shapes.append(shape)

    def pipeline_on(dev):
        return {p.name: ((dev,),) for p in processes}

    tp_forced = backbone.tp_size > 1 or not _fits(backbone.footprint_gib, gpu_mem_gib)
    if tp_forced:
        tp = backbone.tp_size
        if tp == 1:
            tp = next(
                (
                    t
                    for t in (2, 4, 8)
                    if _fits(backbone.footprint_gib / t, gpu_mem_gib)
                ),
                None,
            )
        if tp is None or tp > budget:
            raise ValueError("Backbone weights exceed the GPU budget even with TP")
        if tails and not _fits(tails_footprint, gpu_mem_gib):
            raise ValueError("Tail processes do not fit one GPU")
        per_pipeline = tp + (1 if tails else 0)
        for k in range(1, budget // per_pipeline + 1):
            spare = budget - k * tp
            tail_counts = range(1, spare // k + 1) if tails else (0,)
            for n in tail_counts:
                cursor = count()
                parts = []
                used = set()
                for _ in range(k):
                    ranks = tuple(devices[next(cursor)] for _ in range(tp))
                    used.update(ranks)
                    parts.append({backbone.name: (ranks,)})
                    for _ in range(n):
                        dev = devices[next(cursor)]
                        used.add(dev)
                        parts.append({p.name: ((dev,),) for p in tails})
                suffix = f"_tails{n}" if tails else ""
                emit(f"tp{tp}_x{k}{suffix}", parts, "dedicated", used)
        return shapes

    if _fits(pipeline_footprint, gpu_mem_gib):
        for n in range(1, budget + 1):
            parts = [pipeline_on(devices[i]) for i in range(n)]
            emit(f"replicate_x{n}", parts, "dedicated", set(devices[:n]))
        for k in range(2, max_colocated + 1):
            if not _fits(k * pipeline_footprint, gpu_mem_gib):
                break
            parts = [pipeline_on(d) for d in devices for _ in range(k)]
            for mode in SHARED_MODES:
                emit(f"colocate_x{k}_{mode}", parts, mode, set(devices))
    elif tails and _fits(tails_footprint, gpu_mem_gib):
        for n in range(1, budget // 2 + 1):
            parts = [
                {
                    backbone.name: ((devices[2 * i],),),
                    **{p.name: ((devices[2 * i + 1],),) for p in tails},
                }
                for i in range(n)
            ]
            emit(f"replicate_pairs_x{n}", parts, "dedicated", set(devices[: 2 * n]))
    if not tails or not _fits(tails_footprint, gpu_mem_gib):
        return shapes
    # Note (Jiaxin Deng): tails fed by several backbones are their own flows, so
    # tails sharing a GPU compete like replicas rather than serialize.
    for b in range(1, budget):
        backbone_parts = [{backbone.name: ((devices[i],),)} for i in range(b)]
        tail_devices = devices[b:]
        tail_parts = [{p.name: ((d,),) for p in tails} for d in tail_devices]
        emit(
            f"split_backbone{b}_tails{len(tail_devices)}",
            backbone_parts + tail_parts,
            "dedicated",
            set(devices),
        )
        if b >= 2 and _fits(b * tails_footprint, gpu_mem_gib):
            shared = devices[b]
            tail_parts = [{p.name: ((shared,),) for p in tails} for _ in range(b)]
            for mode in SHARED_MODES:
                emit(
                    f"consolidate_backbone{b}_tails{b}_{mode}",
                    backbone_parts + tail_parts,
                    mode,
                    set(devices[: b + 1]),
                )
    return shapes


def instance_fractions(
    shape: Shape, processes: Mapping[str, ProcessShape], *, gpu_mem_gib: float
) -> dict[str, float]:
    """Memory fraction each process instance declares, filling its GPU.

    Instances sharing a GPU split the usable fraction by footprint; a lone
    instance takes the whole usable fraction so its KV pool is as large as
    the card allows. Replicas of one process share one fraction, since the
    serving schema declares it per stage.
    """
    load: dict[int, float] = {}
    for name, replicas in shape.assignments.items():
        for ranks in replicas:
            for device in ranks:
                load[device] = load.get(device, 0.0) + processes[
                    name
                ].footprint_gib / len(ranks)
    fractions = {}
    for name, replicas in shape.assignments.items():
        shares = []
        for ranks in replicas:
            for device in ranks:
                own = processes[name].footprint_gib / len(ranks)
                shares.append(USABLE_FRACTION * own / load[device])
        fractions[name] = round(min(shares), 3)
    return fractions
