# SPDX-License-Identifier: Apache-2.0
"""Rebuild the PersonaPlex quality report from retained experiment artifacts."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import tarfile
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
REQUIRED_CONDITIONS = {
    "hardware": {"gpu", "gpu_count", "driver"},
    "serving": {
        "revision",
        "benchmark_revision",
        "image",
        "sglang",
        "torch",
        "model",
        "model_revision",
    },
    "generation": {
        "concurrency",
        "sample_rate_hz",
        "input_unit_ms",
        "voice",
        "instructions",
        "quality",
        "greedy",
    },
    "selection": {"v10", "v15"},
    "evaluation": {
        "reference_revision",
        "asr_model",
        "asr_revision",
        "asr_sha256",
        "judge",
    },
}


def read_object(path: Path) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(path.read_bytes())


def build_report(artifacts: Path, conditions_path: Path) -> dict[str, JsonValue]:
    """Rebuild scores and provenance without model inference or network access."""
    conditions = read_object(conditions_path)
    missing = (set(REQUIRED_CONDITIONS) | {"experiment_date_utc"}) - conditions.keys()
    for group, fields in REQUIRED_CONDITIONS.items():
        values = conditions.get(group)
        if isinstance(values, dict):
            missing.update(f"{group}.{field}" for field in fields - values.keys())
        else:
            missing.add(group)
    if missing:
        raise ValueError(f"Missing experiment conditions: {sorted(missing)}")
    else:
        pass
    sources = {
        "fdb_v10_reference": "v10-reference/summary.json",
        "fdb_v10_unified": "v10-reference/unified.json",
        "fdb_v15_timing": "v15-scores/summary.json",
        "fdb_v15_qwen_behavior": "v15-judge/summary.json",
        "greedy_comparison": "parity-warm/parity.json",
        "mimi_encoding_comparison": "mimi-encoding-parity.json",
        "scoring_environment": "scoring-environment.json",
    }
    if (artifacts / "diagnostics.json").is_file():
        sources["diagnostics"] = "diagnostics.json"
    else:
        pass
    reports = {
        name: read_object(artifacts / relative) for name, relative in sources.items()
    }
    timing = reports["fdb_v15_timing"]["engines"]["personaplex"]
    reports["fdb_v15_timing"] = {
        group: {
            key: value
            for key, value in measurements.items()
            if key.startswith("timing_") or key == "asr_files"
        }
        for group, measurements in timing.items()
    }
    parity = reports["greedy_comparison"]
    reports["greedy_comparison"] = {
        key: parity[key]
        for key in (
            "input_sha256",
            "full_reply",
            "caller_duration",
            "session_extra_samples",
        )
    }
    provenance = {
        relative: hashlib.sha256((artifacts / relative).read_bytes()).hexdigest()
        for relative in sources.values()
    }
    return {
        "schema_version": 2,
        "conditions": conditions,
        "measurements": reports,
        "provenance": {
            "source_sha256": provenance,
            "conditions_sha256": hashlib.sha256(
                conditions_path.read_bytes()
            ).hexdigest(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--bundle", type=Path, help="Write the report inputs as a portable tar.gz"
    )
    arguments = parser.parse_args()
    report = build_report(arguments.artifacts, arguments.conditions)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.bundle is not None:
        with tarfile.open(arguments.bundle, "w:gz") as archive:
            archive.add(arguments.conditions, arcname="conditions.json")
            for relative in report["provenance"]["source_sha256"]:
                archive.add(arguments.artifacts / relative, arcname=relative)
    else:
        pass
    if arguments.check:
        recorded = (
            json.dumps(read_object(arguments.output), indent=2, sort_keys=True) + "\n"
        )
        if rendered != recorded:
            print(
                "".join(
                    difflib.unified_diff(
                        recorded.splitlines(keepends=True),
                        rendered.splitlines(keepends=True),
                        fromfile=str(arguments.output),
                        tofile="regenerated",
                    )
                )
            )
            raise SystemExit(1)
        else:
            print(f"Verified {arguments.output}")
    else:
        arguments.output.write_text(rendered)
        print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
