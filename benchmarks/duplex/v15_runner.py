# SPDX-License-Identifier: Apache-2.0
"""Record paired overlap/clean v1.5 samples as native continuous sessions."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import soundfile
from pydantic import JsonValue

from benchmarks.duplex import v15_dataset
from benchmarks.duplex.artifacts import replay_run, source_fingerprint
from benchmarks.duplex.client import PACKET_MS, SAMPLE_RATE, TRANSPORT, run_session
from benchmarks.duplex.profiles import DEFAULT_PROFILE, ProfileName
from benchmarks.duplex.v15_audio import (
    PACING_TOLERANCE_S,
    normalize_audio,
    reconstruct_output,
    write_json,
)
from benchmarks.eval.benchmark_duplex import MAX_TIMEOUT_S

logger = logging.getLogger(__name__)

VARIANTS = {"overlap": "input", "clean": "clean_input"}
RUN_KIND = "full-duplex-bench-v1.5-paired"
RUNNER_FILES = (
    "benchmarks/duplex/v15_audio.py",
    "benchmarks/duplex/v15_dataset.py",
    "benchmarks/duplex/v15_runner.py",
)
VARIANT_STATUSES = ("pending", "invalid", "error", "fail", "not_exercised", "pass")


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


async def run_pairs(
    dataset_root: Path,
    *,
    url: str,
    output: Path,
    server: dict[str, JsonValue],
    dataset_revision: str,
    timeout_s: float,
    sample_ids: list[str] | None = None,
    max_per_subset: int | None = None,
    dataset: ModuleType = v15_dataset,
    variants: dict[str, str] = VARIANTS,
    kind: str = RUN_KIND,
    profile: ProfileName = DEFAULT_PROFILE,
) -> dict[str, JsonValue]:
    """Run every selected sample variant; each failure stays in the selected denominator.

    dataset supplies discover_samples and inventory; variants maps a variant name to
    the sample path key it sends.
    """
    if not dataset_revision:
        raise ValueError("dataset_revision must be nonempty")
    if not 0 < timeout_s <= MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s must be positive and at most {MAX_TIMEOUT_S}")
    dataset_root = dataset_root.resolve()
    samples = dataset.discover_samples(dataset_root, sample_ids, max_per_subset)
    if not samples:
        raise ValueError("no samples selected")
    output.mkdir(parents=True, exist_ok=False)

    source = source_fingerprint()
    repo = Path(__file__).resolve().parents[2]
    source["files_sha256"].update(
        {
            path: hashlib.sha256((repo / path).read_bytes()).hexdigest()
            for path in RUNNER_FILES
        }
    )
    entries = []
    for sample in samples:
        # Note (wenyao): run.json is rewritten per variant; omit bulky transcripts.
        entry = {**asdict(sample), "variants": {}}
        entry.pop("transcripts", None)
        for variant, key in variants.items():
            entry["variants"][variant] = {
                "directory": str(Path("samples") / sample.directory / variant),
                "status": "invalid" if sample.errors else "pending",
                "protocol_verdict": None,
                "qualified": False,
                "errors": list(sample.errors),
                "violations": [],
                "source": {
                    "file": sample.paths.get(key),
                    "sha256": sample.sha256.get(key),
                },
                "normalization": None,
                "input": None,
                "input_timing": None,
                "output": None,
                "files": {},
            }
        entries.append(entry)

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
                    "id": entry["id"],
                    "errors": entry["errors"],
                    "variants": {
                        name: {key: state[key] for key in ("directory", "source")}
                        for name, state in entry["variants"].items()
                    },
                }
                for entry in entries
            ],
        },
    )

    def save(
        status: Literal["preparing", "running", "complete"]
    ) -> dict[str, JsonValue]:
        result = {
            "schema_version": 1,
            "status": status,
            "summary": summarize(entries),
            "samples": entries,
        }
        write_json(output / "run.json", result)
        return result

    save("preparing")
    for sample, entry in zip(samples, entries):
        for variant, key in variants.items():
            state = entry["variants"][variant]
            if state["status"] != "pending":
                continue
            variant_dir = output / state["directory"]
            try:
                pcm, normalization = normalize_audio(dataset_root / sample.paths[key])
                duration_s = len(pcm) / (SAMPLE_RATE * 2)
                state["normalization"] = normalization
                state["input"] = {
                    "sha256": hashlib.sha256(pcm).hexdigest(),
                    "duration_s": duration_s,
                }
                if duration_s >= timeout_s:
                    state["status"] = "invalid"
                    state["errors"].append(
                        f"input duration {duration_s:.3f}s is not below timeout "
                        f"{timeout_s}s"
                    )
                    continue
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
                            "sha256": state["input"]["sha256"],
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
                logger.exception(f"variant {entry['id']}/{variant} preparation failed")
                state["status"] = "error"
                state["errors"].append(
                    f"preparation failed: {type(exc).__name__}: {exc}"
                )
                save("preparing")

    save("running")
    for entry in entries:
        for variant, state in entry["variants"].items():
            if state["status"] != "pending":
                continue
            variant_dir = output / state["directory"]
            try:
                await run_session(
                    url,
                    (variant_dir / "input.pcm").read_bytes(),
                    scenario="continuous",
                    trace_path=variant_dir / "continuous.jsonl",
                    timeout_s=timeout_s,
                    profile=profile,
                )
                report = replay_run(variant_dir)
                write_json(variant_dir / "report.json", report)
                (case,) = report["cases"]
                playout = reconstruct_output(variant_dir, profile=profile)
            except Exception as exc:
                logger.exception(f"variant {entry['id']}/{variant} failed")
                state["status"] = "error"
                state["errors"].append(f"{type(exc).__name__}: {exc}")
            else:
                state["protocol_verdict"] = case["status"]
                state["violations"] = case["violations"]
                state["errors"].extend(playout["errors"])
                state["status"] = "fail" if playout["errors"] else case["status"]
                state["qualified"] = state["status"] == "pass"
                state["input_timing"] = playout["input_timing"]
                state["output"] = {
                    "media_pcm_sha256": playout["pcm_sha256"].get("output-media.wav"),
                    "media_duration_s": playout["media_samples"]
                    / playout["sample_rate"],
                    "playout_duration_s": playout["playout_samples"]
                    / playout["sample_rate"],
                    "initial_delay_s": playout["initial_delay_s"],
                }
            state["files"] = {
                path.name: str(path.relative_to(output))
                for path in sorted(variant_dir.iterdir())
                if not path.name.startswith(".")
            }
            save("running")
    return save("complete")
