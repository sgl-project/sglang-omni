"""Calibrated constants: what two probes on one GPU tell the planner.

A saturation probe of the shipped pipeline on one GPU measures its throughput;
a single-request probe measures the arrival stall and the engine's share of a
request's service time. Stage throughputs follow from the share: stages that
serialize on one GPU add busy time, so 1/T_pipeline = sum_s 1/T_s. Every
derived number is PREDICTED until a measured cell replaces it.
"""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.config.schema import StageConfig
from sglang_omni.config.topology import LogicalProcess
from sglang_omni.restage.capacity import StageCapacity, Workload


class StageConstants(BaseModel):
    model_config = ConfigDict(extra="forbid")

    throughput: float | None = Field(default=None, gt=0)
    provenance: str
    weights_gib: float = Field(ge=0)
    kv_bytes_per_token: int | None = Field(default=None, gt=0)
    delta_s: float = Field(default=0.0, ge=0)


class Constants(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str
    gpu_name: str
    gpu_mem_gib: float = Field(gt=0)
    context_tokens: int = Field(gt=0)
    audio_seconds: float = Field(gt=0)
    slo_rtf: float = Field(default=1.0, gt=0)
    pipeline_throughput: float | None = Field(default=None, gt=0)
    pipeline_provenance: str = ""
    stages: dict[str, StageConstants]

    @property
    def workload(self) -> Workload:
        return Workload(self.context_tokens, self.audio_seconds, self.slo_rtf)

    @classmethod
    def load(cls, path: Path) -> "Constants":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def gpu_stages(stages: Sequence[StageConfig]) -> list[StageConfig]:
    return [stage for stage in stages if stage.gpu is not None]


def kv_stage(
    stages: Sequence[StageConfig], constants: "Constants"
) -> StageConfig | None:
    """The stage whose constants declare a KV pool: the plan sizes its memory."""
    holders = [
        stage
        for stage in stages
        if stage.name in constants.stages
        and constants.stages[stage.name].kv_bytes_per_token is not None
    ]
    if len(holders) > 1:
        raise ValueError("Constants declare a KV pool on more than one stage")
    return holders[0] if holders else None


def process_capacity(
    process: LogicalProcess,
    stages: Sequence[StageConfig],
    constants: Constants,
) -> StageCapacity:
    """Capacity of one instance of a logical process under these constants.

    A process holding every GPU stage is the calibrated pipeline itself and
    keeps its measured throughput; any other membership is predicted from the
    stage shares. A stage without a throughput is assumed non-binding.
    """
    all_gpu = gpu_stages(stages)
    members = [stage for stage in all_gpu if stage.name in process.stage_names]
    if not members:
        raise ValueError(f"Process {process.name!r} holds no GPU stage")
    missing = [stage.name for stage in members if stage.name not in constants.stages]
    if missing:
        raise ValueError(f"Constants lack stages {missing} of process {process.name!r}")
    rows = [constants.stages[stage.name] for stage in members]
    engine = kv_stage(members, constants)
    kv = constants.stages[engine.name].kv_bytes_per_token if engine else None
    delta = constants.stages[engine.name].delta_s if engine else 0.0
    weights = sum(row.weights_gib for row in rows)
    if constants.pipeline_throughput is not None and {
        stage.name for stage in members
    } == {stage.name for stage in all_gpu}:
        return StageCapacity(
            process=process.name,
            throughput=constants.pipeline_throughput,
            provenance=constants.pipeline_provenance or "MEASURED pipeline probe",
            weights_gib=weights,
            kv_bytes_per_token=kv,
            delta_s=delta,
        )
    known = [row for row in rows if row.throughput is not None]
    if not known:
        raise ValueError(
            f"Process {process.name!r} has no stage with a calibrated throughput"
        )
    throughput = 1.0 / sum(1.0 / row.throughput for row in known)
    provenance = "PREDICTED from stage shares: " + "; ".join(
        f"{stage.name}={row.provenance}" for stage, row in zip(members, rows)
    )
    return StageCapacity(
        process=process.name,
        throughput=throughput,
        provenance=provenance,
        weights_gib=weights,
        kv_bytes_per_token=kv,
        delta_s=delta,
    )


def stage_footprints(
    process: LogicalProcess,
    stages: Sequence[StageConfig],
    constants: Constants,
    total_footprint_gib: float,
) -> dict[str, float]:
    """Split a process instance's footprint over its GPU stages by weights.

    The KV pool and runtime margin ride with the engine stage, so the engine
    receives whatever the weight split leaves of the total.
    """
    members = [
        stage for stage in gpu_stages(stages) if stage.name in process.stage_names
    ]
    engine = kv_stage(members, constants)
    shares = {stage.name: constants.stages[stage.name].weights_gib for stage in members}
    if engine is None:
        total_weights = sum(shares.values()) or 1.0
        return {
            name: total_footprint_gib * weight / total_weights
            for name, weight in shares.items()
        }
    footprints = {
        name: weight for name, weight in shares.items() if name != engine.name
    }
    footprints[engine.name] = total_footprint_gib - sum(footprints.values())
    return footprints


def backbone_process(
    processes: Sequence[LogicalProcess],
    stages: Sequence[StageConfig],
    constants: Constants,
) -> LogicalProcess:
    """The process holding the KV pool, whose replicas the shapes revolve around."""
    engine = kv_stage(gpu_stages(stages), constants)
    if engine is None:
        raise ValueError("Constants declare no KV-holding stage")
    return next(p for p in processes if engine.name in p.stage_names)
