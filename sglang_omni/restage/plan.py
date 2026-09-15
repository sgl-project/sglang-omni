"""Rank residency shapes for a pipeline and export the candidates worth measuring."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.sources import dump_user_config
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.restage.calibration import (
    Constants,
    backbone_process,
    gpu_stages,
    process_capacity,
    stage_footprints,
)
from sglang_omni.restage.candidates import materialize_candidate
from sglang_omni.restage.capacity import PlanUtility, footprint_gib, plan_utility
from sglang_omni.restage.configurations import enumerate_configurations
from sglang_omni.restage.shapes import (
    ProcessShape,
    instance_fractions,
    residency_shapes,
)

RESERVED_DIMENSION_KEYS = ("mps", "processes.")
RESERVED_DIMENSION_SUFFIXES = (
    ".gpu",
    ".gpu_memory_fraction",
    ".engine.mem_fraction_static",
    ".tp_size",
)


class SearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    devices: list[int] = Field(min_length=1)
    max_colocated: int = Field(default=4, ge=1)
    dimensions: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    def validate_dimensions(self) -> None:
        for name, choices in self.dimensions.items():
            for choice in choices:
                for key in choice:
                    if key.startswith(RESERVED_DIMENSION_KEYS) or key.endswith(
                        RESERVED_DIMENSION_SUFFIXES
                    ):
                        raise ValueError(
                            f"Dimension {name!r} sets {key!r}, which the shapes own"
                        )


def _apply_fractions(config, logical, stages, constants, shape, processes):
    fractions = instance_fractions(shape, processes, gpu_mem_gib=constants.gpu_mem_gib)
    result = config.model_copy(deep=True)
    for process in logical.processes:
        if process.name not in fractions:
            continue
        total = fractions[process.name] * constants.gpu_mem_gib
        per_stage = stage_footprints(process, stages, constants, total)
        for stage in result.stages:
            if stage.name not in per_stage:
                continue
            fraction = round(per_stage[stage.name] / constants.gpu_mem_gib, 3)
            # Note (Jiaxin Deng): a model that derives its engine reserve from the
            # stage fraction declares it without mem_fraction_static; leave those.
            derives = (
                stage.gpu_memory_fraction is not None
                and stage.engine is not None
                and stage.engine.mem_fraction_static is None
            )
            stage.total_reserve_bytes = None
            stage.gpu_memory_fraction = fraction
            if stage.engine is not None and not derives:
                stage.engine.mem_fraction_static = fraction
    result.mps = "on" if shape.sharing_mode == "mps" else "off"
    return type(config).model_validate(result.model_dump()), fractions


def rank_shapes(
    config: PipelineConfig,
    space: SearchSpace,
    constants: Constants,
) -> list[dict[str, Any]]:
    """Score every shape of every configuration variant; rows carry rejections."""
    space.validate_dimensions()
    rows = []
    for variant in enumerate_configurations(config, space.dimensions):
        context = {"selections": variant.selections}
        if variant.config is None:
            rows.append(
                {**context, "status": "rejected", "rejection": variant.rejection}
            )
            continue
        try:
            logical, stages = compile_logical_processes(variant.config)
            gpu_names = {stage.name for stage in gpu_stages(stages)}
            processes = [
                p for p in logical.processes if gpu_names.intersection(p.stage_names)
            ]
            capacities = {
                p.name: process_capacity(p, stages, constants) for p in processes
            }
            backbone = backbone_process(processes, stages, constants)
            process_shapes = {
                p.name: ProcessShape(
                    p.name,
                    footprint_gib(capacities[p.name], constants.workload),
                    backbone=p.name == backbone.name,
                    tp_size=p.tp_size,
                )
                for p in processes
            }
            shapes = residency_shapes(
                list(process_shapes.values()),
                space.devices,
                gpu_mem_gib=constants.gpu_mem_gib,
                max_colocated=space.max_colocated,
            )
        except ValueError as exc:
            rows.append({**context, "status": "rejected", "rejection": str(exc)})
            continue
        for shape in shapes:
            row = {
                **context,
                "shape": shape.name,
                "assignments": shape.assignments,
                "flows": shape.flows,
                "sharing_mode": shape.sharing_mode,
                "idle_devices": list(shape.idle_devices),
            }
            try:
                sized, fractions = _apply_fractions(
                    variant.config, logical, stages, constants, shape, process_shapes
                )
                candidate = materialize_candidate(
                    sized, shape.assignments, space.devices
                )
                utility = plan_utility(
                    shape.assignments,
                    shape.flows,
                    capacities,
                    constants.workload,
                    gpu_mem_gib=constants.gpu_mem_gib,
                    fractions=fractions,
                    sharing_mode=shape.sharing_mode,
                )
            except ValueError as exc:
                rows.append({**row, "status": "rejected", "rejection": str(exc)})
                continue
            rows.append(
                {
                    **row,
                    "status": "candidate",
                    "emitted": len(rows),
                    "fractions": fractions,
                    "predicted": _utility_row(utility, constants),
                    "config": candidate,
                }
            )
    ranked = sorted(
        rows,
        key=lambda r: -(
            r["predicted"]["utility"] if r["status"] == "candidate" else -1
        ),
    )
    for rank, row in enumerate(
        (r for r in ranked if r["status"] == "candidate"), start=1
    ):
        row["rank"] = rank
    return ranked


def _utility_row(utility: PlanUtility, constants: Constants) -> dict[str, Any]:
    return {
        "utility": utility.utility,
        "unit": "audio-s/s",
        "requests_per_s": utility.utility / constants.audio_seconds,
        "binding": utility.binding,
        "provenance": utility.provenance,
        "gpus": [asdict(row) for row in utility.gpus],
        "pool_bounds": utility.pool_bounds,
    }


def write_plan(
    config: PipelineConfig,
    space: SearchSpace,
    constants: Constants,
    destination: Path,
    *,
    top: int = 3,
) -> dict[str, Any]:
    """Write ranking.json plus YAML for the baseline and the top-ranked shapes.

    The baseline is the input configuration on the first budget device. The
    ranking is a prediction; ``autotune run`` measures the exported files.
    """
    if top < 1:
        raise ValueError("top must be positive")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "search-space.json").write_text(
        space.model_dump_json(indent=2), encoding="utf-8"
    )
    constants.save(destination / "constants.json")
    rows = rank_shapes(config, space, constants)
    baseline = _baseline_row(rows)
    _write_yaml(destination / "baseline.yaml", baseline["config"])
    baseline["config_file"] = "baseline.yaml"
    exported = []
    seen = {(baseline["shape"], round(baseline["predicted"]["utility"], 6))}
    for row in rows:
        if row["status"] != "candidate":
            continue
        # Note (Jiaxin Deng): variants that only regroup stages predict the same
        # utility for the same shape; measuring both would repeat one cell.
        signature = (row["shape"], round(row["predicted"]["utility"], 6))
        if len(exported) < top and signature not in seen:
            seen.add(signature)
            filename = f"candidate-{row['rank']:03d}-{row['shape']}.yaml"
            _write_yaml(destination / filename, row["config"])
            row["config_file"] = filename
            exported.append(filename)
        del row["config"]
    (destination / "ranking.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    summary = {
        "examined": len(rows),
        "candidates": sum(row["status"] == "candidate" for row in rows),
        "rejected": sum(row["status"] == "rejected" for row in rows),
        "exported": exported,
        "baseline": "baseline.yaml",
        "performance_measured": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _baseline_row(rows):
    """The shipped layout: the first single-copy dedicated shape of the first variant."""
    candidates = [
        row
        for row in rows
        if row["status"] == "candidate"
        and row["sharing_mode"] == "dedicated"
        and all(len(r) == 1 for r in row["assignments"].values())
        and not any(row["selections"].values())
    ]
    if not candidates:
        raise ValueError("No dedicated single-copy candidate to serve as the baseline")
    return min(candidates, key=lambda row: row["emitted"])


def _write_yaml(path, config):
    path.write_text(
        yaml.safe_dump(dump_user_config(config), sort_keys=False), encoding="utf-8"
    )


def load_plan(directory: Path) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Exported configurations, baseline first, with their predicted rows."""
    directory = directory.resolve()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((directory / "ranking.json").read_text(encoding="utf-8"))
    configs = {"baseline": directory / summary["baseline"]}
    predicted = {}
    for row in rows:
        if row.get("config_file"):
            key = Path(row["config_file"]).stem
            configs[key] = directory / row["config_file"]
            predicted[key] = row["predicted"]
    configs = {"baseline": configs.pop("baseline"), **configs}
    for path in configs.values():
        if not path.is_file():
            raise ValueError(f"Plan configuration missing: {path}")
    return configs, predicted
