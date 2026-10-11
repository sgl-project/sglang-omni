# SPDX-License-Identifier: Apache-2.0
"""Record paired overlap/clean v1.5 samples as native continuous sessions."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import soundfile
from pydantic import JsonValue

import benchmarks.duplex.v15_dataset as v15_dataset
from benchmarks.duplex.artifacts import replay_run, source_fingerprint
from benchmarks.duplex.client import (
    MAX_TIMEOUT_S,
    PACKET_MS,
    SAMPLE_RATE,
    TRANSPORT,
    run_session,
)
from benchmarks.duplex.profiles import DEFAULT_PROFILE, ProfileName
from benchmarks.duplex.v10_dataset import Sample as V10Sample
from benchmarks.duplex.v15_audio import (
    PACING_TOLERANCE_S,
    normalize_audio,
    reconstruct_output,
    write_json,
)
from benchmarks.duplex.v15_dataset import Sample as V15Sample

logger = logging.getLogger(__name__)

VARIANTS = {"overlap": "input", "clean": "clean_input"}
RUN_KIND = "full-duplex-bench-v1.5-paired"
RUNNER_FILES = (
    "benchmarks/duplex/v15_audio.py",
    "benchmarks/duplex/v15_dataset.py",
    "benchmarks/duplex/v15_runner.py",
)
VARIANT_STATUSES = ("pending", "invalid", "error", "fail", "not_exercised", "pass")


class DuplexDataset(Protocol):
    def discover_samples(
        self,
        root: Path,
        sample_ids: list[str] | None = None,
        max_per_subset: int | None = None,
    ) -> list[V10Sample] | list[V15Sample]: ...

    def inventory(self, root: Path) -> dict[str, dict[str, int] | list[str]]: ...


def summarize(samples: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    statuses = [
        variant["status"]
        for sample in samples
        for variant in sample["variants"].values()
    ]
    return {
        "samples_selected": len(samples),
        "samples_invalid": sum(bool(sample["errors"]) for sample in samples),
        "variants_selected": len(statuses),
        "variants": {status: statuses.count(status) for status in VARIANT_STATUSES},
        "pairs_qualified": sum(
            all(variant["qualified"] for variant in sample["variants"].values())
            for sample in samples
        ),
        "per_subset": {
            subset: sum(sample["subset"] == subset for sample in samples)
            for subset in dict.fromkeys(sample["subset"] for sample in samples)
        },
    }


async def run_samples(
    dataset_root: Path,
    *,
    url: str,
    output: Path,
    server: dict[str, JsonValue],
    dataset_revision: str,
    timeout_s: float,
    sample_ids: list[str] | None = None,
    max_per_subset: int | None = None,
    dataset: DuplexDataset = v15_dataset,
    variants: dict[str, str] | None = None,
    kind: str = RUN_KIND,
    profile: ProfileName = DEFAULT_PROFILE,
) -> dict[str, JsonValue]:
    """Run selected sample variants, retaining failures in the selected denominator."""
    if not dataset_revision:
        raise ValueError("dataset_revision must be nonempty")
    else:
        pass
    if not 0 < timeout_s <= MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s must be positive and at most {MAX_TIMEOUT_S}")
    else:
        pass
    variants = VARIANTS if variants is None else variants
    dataset_root = dataset_root.resolve()
    samples = dataset.discover_samples(dataset_root, sample_ids, max_per_subset)
    if not samples:
        raise ValueError("no samples selected")
    else:
        pass
    output.mkdir(parents=True, exist_ok=False)

    source = source_fingerprint()
    repository_root = Path(__file__).resolve().parents[2]
    source["files_sha256"].update(
        {
            path: hashlib.sha256((repository_root / path).read_bytes()).hexdigest()
            for path in RUNNER_FILES
        }
    )
    sample_records = []
    for sample in samples:
        sample_record = {**asdict(sample), "variants": {}}
        for variant, source_key in variants.items():
            sample_record["variants"][variant] = {
                "directory": str(Path("samples") / sample.directory / variant),
                "status": "invalid" if sample.errors else "pending",
                "protocol_verdict": None,
                "qualified": False,
                "errors": list(sample.errors),
                "violations": [],
                "source": {
                    "file": sample.paths.get(source_key),
                    "sha256": sample.sha256.get(source_key),
                },
                "normalization": None,
                "input": None,
                "input_timing": None,
                "output": None,
                "files": {},
            }
        sample_records.append(sample_record)

    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "kind": kind,
            "profile": profile,
            "source": source,
            "server": server,
            "dataset": {
                "root": str(dataset_root),
                "revision": dataset_revision,
                "inventory": dataset.inventory(dataset_root),
                "selection": {
                    "sample_ids": sample_ids,
                    "max_per_subset": max_per_subset,
                },
            },
            "config": {
                "scenario": "continuous",
                "variants": variants,
                "timeout_s": timeout_s,
                "packet_ms": PACKET_MS,
                "pacing_tolerance_s": PACING_TOLERANCE_S,
                "transport": TRANSPORT,
                "response_cancel": "never sent",
            },
            "samples": [
                {
                    "id": sample_record["id"],
                    "errors": sample_record["errors"],
                    "variants": {
                        variant_name: {
                            source_key: variant_state[source_key]
                            for source_key in ("directory", "source")
                        }
                        for variant_name, variant_state in sample_record[
                            "variants"
                        ].items()
                    },
                }
                for sample_record in sample_records
            ],
        },
    )

    def save(
        status: Literal["preparing", "running", "complete"]
    ) -> dict[str, JsonValue]:
        result = {
            "schema_version": 1,
            "status": status,
            "summary": summarize(sample_records),
            "samples": sample_records,
        }
        write_json(output / "run.json", result)
        return result

    save("preparing")
    for sample, sample_record in zip(samples, sample_records):
        for variant, source_key in variants.items():
            variant_state = sample_record["variants"][variant]
            if variant_state["status"] != "pending":
                continue
            else:
                pass
            variant_dir = output / variant_state["directory"]
            try:
                pcm, normalization = normalize_audio(
                    dataset_root / sample.paths[source_key]
                )
                duration_s = len(pcm) / (SAMPLE_RATE * 2)
                variant_state["normalization"] = normalization
                variant_state["input"] = {
                    "sha256": hashlib.sha256(pcm).hexdigest(),
                    "duration_s": duration_s,
                }
                if duration_s >= timeout_s:
                    variant_state["status"] = "invalid"
                    variant_state["errors"].append(
                        f"input duration {duration_s:.3f}s is not below timeout "
                        f"{timeout_s}s"
                    )
                    continue
                else:
                    pass
                variant_dir.mkdir(parents=True)
                (variant_dir / "input.pcm").write_bytes(pcm)
                soundfile.write(
                    str(variant_dir / "input.wav"),
                    np.frombuffer(pcm, dtype="<i2"),
                    SAMPLE_RATE,
                    subtype="PCM_16",
                )
                write_json(
                    variant_dir / "manifest.json",
                    {
                        "schema_version": 1,
                        "profile": profile,
                        "source": source,
                        "server": server,
                        "config": {
                            "packet_ms": PACKET_MS,
                            "paced": True,
                            "timeout_s": timeout_s,
                            "input_duration_s": duration_s,
                            "transport": TRANSPORT,
                            "dataset_sample": sample.id,
                            "dataset_variant": variant,
                        },
                        "input": {
                            "file": "input.pcm",
                            "sha256": variant_state["input"]["sha256"],
                            "sample_rate": SAMPLE_RATE,
                        },
                        "cases": [
                            {
                                "id": "continuous",
                                "scenario": "continuous",
                                "trace_file": "continuous.jsonl",
                            }
                        ],
                    },
                )
            except Exception as exc:
                logger.exception(
                    f"variant {sample_record['id']}/{variant} preparation failed"
                )
                variant_state["status"] = "error"
                variant_state["errors"].append(
                    f"preparation failed: {type(exc).__name__}: {exc}"
                )
                save("preparing")

    save("running")
    for sample_record in sample_records:
        for variant, variant_state in sample_record["variants"].items():
            if variant_state["status"] != "pending":
                continue
            else:
                pass
            variant_dir = output / variant_state["directory"]
            try:
                await run_session(
                    url,
                    (variant_dir / "input.pcm").read_bytes(),
                    trace_path=variant_dir / "continuous.jsonl",
                    timeout_s=timeout_s,
                    profile=profile,
                )
                protocol_report = replay_run(variant_dir)
                write_json(variant_dir / "report.json", protocol_report)
                (case_report,) = protocol_report["cases"]
                playout_report = reconstruct_output(variant_dir, profile=profile)
            except Exception as exc:
                logger.exception(f"variant {sample_record['id']}/{variant} failed")
                variant_state["status"] = "error"
                variant_state["errors"].append(f"{type(exc).__name__}: {exc}")
            else:
                variant_state["protocol_verdict"] = case_report["status"]
                variant_state["violations"] = case_report["violations"]
                variant_state["errors"].extend(playout_report["errors"])
                variant_state["status"] = (
                    "fail" if playout_report["errors"] else case_report["status"]
                )
                variant_state["qualified"] = variant_state["status"] == "pass"
                variant_state["input_timing"] = playout_report["input_timing"]
                variant_state["output"] = {
                    "media_pcm_sha256": playout_report["pcm_sha256"].get(
                        "output-media.wav"
                    ),
                    "media_duration_s": playout_report["media_samples"]
                    / playout_report["sample_rate"],
                    "playout_duration_s": playout_report["playout_samples"]
                    / playout_report["sample_rate"],
                    "initial_delay_s": playout_report["initial_delay_s"],
                }
            variant_state["files"] = {
                path.name: str(path.relative_to(output))
                for path in sorted(variant_dir.iterdir())
                if not path.name.startswith(".")
            }
            save("running")
    return save("complete")
