# SPDX-License-Identifier: Apache-2.0
"""Render saved v1.0 unified scores as per-repeat Markdown tables."""

from __future__ import annotations

import shlex
from pathlib import Path

from benchmarks.duplex.fdb.common import V10_DIR, read_json
from benchmarks.duplex.reference_v10 import MANIFEST_KIND
from benchmarks.duplex.run_artifacts import file_sha256
from benchmarks.duplex.unified_v10 import (
    DEFINITIONS_VERSION,
    KIND,
    RESPONSE_WITHIN_S,
    describe,
)
from benchmarks.duplex.v10_dataset import SUBSET_TASKS, SUBSETS


def render_unified(repeat_dirs: list[Path]) -> list[str]:
    """Show each selected repeat without pooling metrics or confidence intervals."""
    selected_repeats = [
        repeat_dir
        for repeat_dir in repeat_dirs
        if (repeat_dir / V10_DIR / "sample-ids.txt").is_file()
        or (repeat_dir / V10_DIR / "reference" / "manifest.json").is_file()
    ]
    if not selected_repeats:
        return []
    else:
        pass
    lines = [
        "## FDB v1.0 unified metrics",
        "",
        "Each table describes one repeat. Sample rates are proportions among eligible "
        "samples; their 95% Wilson intervals describe sampling uncertainty. "
        "All-selected bounds allow every ineligible outcome to be either false "
        "or true and are not confidence intervals. Other 95% intervals resample "
        "whole samples. No metrics or intervals are averaged across repeats. "
        "A response latency marked `not reached` includes eligible samples "
        "without a response; `n/a` with 0 eligible means no data.",
    ]
    for repeat_dir in selected_repeats:
        tree = repeat_dir / V10_DIR / "reference"
        manifest_path = tree / "manifest.json"
        asr_path = tree / "asr.json"
        unified_path = tree / "unified.json"
        samples_path = tree / "unified-samples.jsonl"
        selection_path = repeat_dir / V10_DIR / "sample-ids.txt"
        lines += ["", f"### {repeat_dir.name}", ""]
        if not manifest_path.is_file():
            reason = "v1.0 export is incomplete (manifest.json is missing)"
        elif not asr_path.is_file():
            reason = "v1.0 ASR is incomplete (asr.json is missing)"
        elif not unified_path.is_file() or not samples_path.is_file():
            reason = "the unified output pair is missing or incomplete"
        else:
            reason = ""
        if reason:
            lines += unavailable(repeat_dir, reason)
            continue
        else:
            pass

        manifest = read_json(manifest_path)
        unified = read_json(unified_path)
        if manifest.get("kind") != MANIFEST_KIND:
            reason = "the export manifest has an unsupported kind"
        elif (
            unified.get("kind") != KIND
            or unified.get("schema_version") != 1
            or unified.get("definitions_version") != DEFINITIONS_VERSION
        ):
            reason = "the unified output has an unsupported kind, schema, or definition"
        elif unified.get("manifest_sha256") != file_sha256(
            manifest_path
        ) or unified.get("asr_sha256") != file_sha256(asr_path):
            reason = "the unified output is stale for the current manifest or ASR"
        else:
            selected_subsets = {
                subset
                for subset, counts in manifest["counts"].items()
                if counts["selected"] > 0
            }
            if selection_path.is_file():
                selected_ids = set(selection_path.read_text().splitlines())
                exported_ids = {sample["sample_id"] for sample in manifest["samples"]}
                if selected_ids != exported_ids:
                    reason = "the export manifest does not match the selected samples"
                else:
                    pass
            else:
                pass
            scored_subsets = {
                subset
                for subset in selected_subsets
                if SUBSET_TASKS[subset] != "backchannel"
            }
            if not reason and set(unified["subsets"]) != scored_subsets:
                reason = (
                    "the unified output does not cover every selected scored subset"
                )
            else:
                pass
            if not reason and any(
                unified["subsets"][subset]["counts"]["selected"]
                != manifest["counts"][subset]["selected"]
                or unified["subsets"][subset]["counts"]["eligible"]
                != manifest["counts"][subset]["eligible"]
                for subset in scored_subsets
            ):
                reason = "the unified counts do not match the export manifest"
            else:
                pass
        if reason:
            lines += unavailable(repeat_dir, reason)
            continue
        else:
            pass

        links = ", ".join(
            f"[{name}]({repeat_dir.name}/{V10_DIR}/reference/{name})"
            for name in (
                "manifest.json",
                "asr.json",
                "unified.json",
                "unified-samples.jsonl",
            )
        )
        intervals = unified["intervals"]
        lines += [
            f"Artifacts: {links}.",
            f"Definitions: `{unified['definitions_version']}`; 95% intervals: "
            f"sample rates = {intervals['sample_rates']}, others = "
            f"{intervals['other']} ({intervals['bootstrap_replicates']} replicates, "
            f"seed {intervals['bootstrap_seed']}).",
            f"Input SHA-256: manifest `{unified['manifest_sha256']}`; "
            f"ASR `{unified['asr_sha256']}`.",
            "",
            "| Subset | Declared | Selected | Eligible | Ineligible | Coverage | "
            "Ineligible reasons | Minimum observable horizon after reference |",
            "|---|---:|---:|---:|---:|---:|---|---:|",
        ]
        for subset in SUBSETS:
            if subset not in unified["subsets"]:
                continue
            else:
                pass
            result = unified["subsets"][subset]
            counts = result["counts"]
            reasons = ", ".join(
                f"{name.replace('|', '&#124;')} ({count})"
                for name, count in sorted(counts["ineligible_reasons"].items())
            )
            horizon_seconds = result.get("min_observable_s")
            horizon_text = (
                "n/a" if horizon_seconds is None else f"{horizon_seconds:.3f} s"
            )
            lines.append(
                f"| `{subset}` | {counts['declared']} | {counts['selected']} | "
                f"{counts['eligible']} | {counts['ineligible']} | "
                f"{result['coverage']:.3f} | {reasons or 'none'} | {horizon_text} |"
            )
        lines += [
            "",
            "| Subset | Metric | Value and interval |",
            "|---|---|---|",
        ]
        for subset in SUBSETS:
            if subset not in unified["subsets"]:
                continue
            else:
                pass
            for metric_name, metric in unified["subsets"][subset]["metrics"].items():
                lines.append(f"| `{subset}` | `{metric_name}` | {describe(metric)} |")
            horizon_seconds = unified["subsets"][subset].get("min_observable_s")
            if horizon_seconds is not None and horizon_seconds < max(RESPONSE_WITHIN_S):
                lines.append(
                    f"| `{subset}` | Observation limit | Minimum {horizon_seconds:.3f} s "
                    "after reference; response rates at longer thresholds are "
                    "lower bounds for samples with shorter windows. |"
                )
            else:
                pass
        if "icc_backchannel" in selected_subsets:
            lines += [
                "",
                "Backchannel is scored by the official reference evaluator only.",
            ]
        else:
            pass
    return lines


def unavailable(repeat_dir: Path, reason: str) -> list[str]:
    tree = repeat_dir / V10_DIR / "reference"
    run_name = repeat_dir.parent.name
    return [
        f"Unified metrics unavailable: {reason}. Repair the export or ASR first "
        "if needed, then backfill without repeating ASR:",
        "",
        "```bash",
        f"export FDB_WORK={shlex.quote(str(repeat_dir.parents[2]))}",
        'eval "$(python -m benchmarks.duplex.fdb env)"',
        'CUDA_VISIBLE_DEVICES="" "$SCORING_PYTHON" -m '
        "benchmarks.eval.benchmark_duplex_v10 unified-score "
        f"--tree {shlex.quote(str(tree))}",
        f"python -m benchmarks.duplex.fdb aggregate --run-name {shlex.quote(run_name)}",
        "```",
    ]
