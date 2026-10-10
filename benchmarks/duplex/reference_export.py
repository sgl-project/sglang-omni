# SPDX-License-Identifier: Apache-2.0
"""Export fixed observation windows without changing recorded sessions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import JsonValue

from benchmarks.duplex.profiles import PROFILES
from benchmarks.duplex.reference_audio import (
    FILES,
    POLICY,
    POLICY_VERSION,
    SCHEMA,
    SEND_RECEIPTS,
    analyze_variant,
    diagnostics,
    load_runs,
    sha_bytes,
    write_wav,
)
from benchmarks.duplex.reference_capture import TraceFormat, resolve_trace_format
from benchmarks.duplex.reference_core import read_json, utc_now
from benchmarks.duplex.run_artifacts import file_sha256
from benchmarks.duplex.v15_audio import normalize_audio, write_json
from benchmarks.duplex.v15_dataset import SUBSETS, list_sample_dirs


def export_runs(
    runs: list[Path],
    output: Path,
    engine: str,
    sample_ids: list[str] | None = None,
    dataset_root: Path | None = None,
    trace_format: str | None = None,
) -> dict[str, JsonValue]:
    """Export every selected variant, retaining missing and invalid outcomes."""
    capture_format = resolve_trace_format(trace_format)
    output = output.resolve()
    for run in runs:
        if output.is_relative_to(run.resolve()):
            raise ValueError("Reference output must be outside every source run")
        else:
            pass
    chosen, sources, superseded = load_runs(runs, capture_format)
    if sample_ids:
        selected = list(dict.fromkeys(sample_ids))
    elif dataset_root is not None:
        names, _ = list_sample_dirs(dataset_root.resolve())
        selected = [f"{subset}/{name}" for subset in SUBSETS for name in names[subset]]
    else:
        selected = list(chosen)
    for sample_id in selected:
        parts = sample_id.split("/")
        if len(parts) != 2 or any(
            source_path in ("", ".", "..") for source_path in parts
        ):
            raise ValueError(f"Invalid sample ID: {sample_id}")
        else:
            pass
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for sample_id in selected:
        row = {
            "sample_id": sample_id,
            "category": sample_id.split("/")[0],
            "variants": {},
            "files": {},
        }
        rows.append(row)
        if sample_id not in chosen:
            row["variants"] = {
                variant: {
                    "eligible": False,
                    "reasons": ["pair not captured in any supplied run"],
                }
                for variant in FILES
            }
            continue
        else:
            pass
        run, entry = chosen[sample_id]
        source_manifest = read_json(run / "manifest.json")
        # note (luojiaxuan): load_runs already required a legacy-protocol profile.
        profile_rate = (
            PROFILES[source_manifest["profile"]].output_sample_rate
            if capture_format is TraceFormat.LEGACY_PCM16
            else None
        )
        sidecar = bool((source_manifest.get("source") or {}).get("campaign_adapter"))
        transport = (source_manifest.get("config") or {}).get("transport") or {}
        sidecar = sidecar or transport.get("input_send_receipts") == SEND_RECEIPTS
        target = output / sample_id
        target.mkdir(parents=True)
        for variant, (input_name, output_name) in FILES.items():
            state = entry["variants"][variant]
            directory = (run / state["directory"]).resolve()
            if not directory.is_relative_to(run):
                raise ValueError(f"Variant directory escapes run: {directory}")
            else:
                pass
            record, pcm, audio = analyze_variant(
                directory,
                capture_format,
                (state.get("input") or {}).get("sha256"),
                receipts_required=sidecar,
                profile_rate=profile_rate,
            )
            reasons = record["window"]["reasons"]
            # note (luojiaxuan): A legacy session that failed its protocol replay can
            # be silent for protocol reasons, so silence is only valid after a pass.
            if (
                capture_format is TraceFormat.LEGACY_PCM16
                and state["protocol_verdict"] != "pass"
            ):
                reasons.append(
                    f"recorder protocol verdict {state['protocol_verdict']!r} is not pass"
                )
            else:
                pass
            if dataset_root is not None and pcm is not None:
                source = dataset_root / state["source"]["file"]
                if file_sha256(source) != state["source"]["sha256"]:
                    reasons.append("dataset source sha256 differs from run.json")
                elif sha_bytes(normalize_audio(source)[0]) != sha_bytes(pcm):
                    reasons.append("renormalized dataset source differs from input.pcm")
                else:
                    pass
            else:
                pass
            record["window"]["valid"] = not reasons
            record.update(
                eligible=not reasons,
                reasons=reasons,
                flags=[k for k, v in record.get("boundary", {}).items() if v is True],
                protocol_diagnostics=diagnostics(run, state),
            )
            if reasons:
                record.pop("output", None)
                record.pop("boundary", None)
            else:
                row["files"][input_name] = write_wav(
                    target / input_name, np.frombuffer(pcm, "<i2")
                )
                row["files"][output_name] = write_wav(target / output_name, audio)
            row["variants"][variant] = record
    variants = [v for r in rows for v in r["variants"].values()]
    manifest = {
        "schema_version": 1,
        "kind": SCHEMA,
        "engine": engine,
        "trace_format": capture_format.value,
        "created_utc": utc_now(),
        "policy_version": POLICY_VERSION,
        "policy": POLICY,
        "timeline": "FIFO receipt-time playout in [0,T_input], mono 16 kHz PCM16",
        "sources": sources,
        "superseded": superseded,
        "builder_sha256": {
            source_path.name: file_sha256(source_path)
            for source_path in (
                Path(__file__),
                Path(__file__).with_name("reference_audio.py"),
                Path(__file__).with_name("reference_capture.py"),
                Path(__file__).with_name("run_artifacts.py"),
            )
        },
        "counts": {
            "selected_pairs": len(rows),
            "selected_variants": len(variants),
            "eligible_variants": sum(v["eligible"] for v in variants),
            "eligible_pairs": sum(
                all(v["eligible"] for v in r["variants"].values()) for r in rows
            ),
        },
        "samples": rows,
    }
    write_json(output / "reference-manifest.json", manifest)
    return manifest
