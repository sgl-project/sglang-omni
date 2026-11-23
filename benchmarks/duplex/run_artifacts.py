# SPDX-License-Identifier: Apache-2.0
"""Load and account for recorded duplex runs and their aligned transcripts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

Timeline = Literal["simulated_playout", "media"]
TIMELINES = {
    "simulated_playout": {
        "audio": "output-playout.wav",
        "meaning": "simulated zero-buffer client playout placed at receipt time; "
        "not acoustic timing and not paper-identical latency",
    },
    "media": {
        "audio": "output-media.wav",
        "meaning": "concatenated model output samples from session start, "
        "ignoring receipt time",
    },
}
RUN_KINDS = ("full-duplex-bench-v1.5-paired", "full-duplex-bench-v1.0")


def file_sha256(path: Path) -> str:
    with path.open("rb") as artifact_file:
        return hashlib.file_digest(artifact_file, "sha256").hexdigest()


def load_run(run_dir: Path) -> tuple[dict[str, JsonValue], dict[str, JsonValue], str]:
    """Return the run manifest, run.json and the manifest digest that pins them."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") not in RUN_KINDS:
        raise ValueError(f"{run_dir} is not a Full-Duplex-Bench run directory")
    else:
        pass
    return (
        manifest,
        json.loads((run_dir / "run.json").read_text()),
        file_sha256(manifest_path),
    )


def create_output(output: Path, run_dir: Path) -> None:
    """Create a fresh directory outside the recorded run, which stays immutable."""
    if output.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError(f"output {output} must be outside the run directory {run_dir}")
    else:
        pass
    output.mkdir(parents=True, exist_ok=False)


def accounting(
    manifest: dict[str, JsonValue], run: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """Declared, available and selected counts; every non-pass variant is listed."""
    declared = manifest["dataset"]["inventory"]["declared"]
    observed = manifest["dataset"]["inventory"]["observed"]
    samples = run["samples"]
    variants = [
        (sample["id"], variant_name, variant_state)
        for sample in samples
        for variant_name, variant_state in sample["variants"].items()
    ]
    return {
        "run_status": run["status"],
        "declared_pairs": sum(declared.values()),
        "available_pairs": sum(observed.values()),
        "per_subset": {
            subset: {
                "declared": declared[subset],
                "available": observed[subset],
                "selected": sum(sample["subset"] == subset for sample in samples),
            }
            for subset in declared
        },
        "selected_pairs": len(samples),
        "selected_variants": len(variants),
        "attempted_variants": sum(
            variant_state["status"] not in ("pending", "invalid")
            for _, _, variant_state in variants
        ),
        "variant_status": dict(
            Counter(variant_state["status"] for _, _, variant_state in variants)
        ),
        "qualified_pairs": sum(
            all(
                variant_state["qualified"]
                for variant_state in sample["variants"].values()
            )
            for sample in samples
        ),
        "load": run.get("load"),
        "failures": [
            {
                "sample_id": sample_id,
                "variant": variant_name,
                "status": variant_state["status"],
                "errors": variant_state["errors"],
                "violations": variant_state["violations"],
            }
            for sample_id, variant_name, variant_state in variants
            if variant_state["status"] != "pass"
        ],
    }


def load_output_transcripts(
    transcripts_dir: Path, run_dir: Path, manifest_sha256: str, timeline: Timeline
) -> tuple[dict[tuple[str, str], dict[str, JsonValue]], dict[str, JsonValue]]:
    """Map (sample_id, variant) to ASR evidence after checking it matches this run."""
    transcripts = json.loads((transcripts_dir / "transcripts.json").read_text())
    if transcripts["run"]["manifest_sha256"] != manifest_sha256:
        raise ValueError(f"{transcripts_dir} transcribes a different run")
    elif transcripts["timeline"] != timeline:
        raise ValueError(
            f"{transcripts_dir} uses timeline {transcripts['timeline']}, not {timeline}"
        )
    else:
        pass
    evidence = {}
    for transcript_entry in transcripts["variants"]:
        if transcript_entry["status"] != "transcribed":
            continue
        else:
            pass
        if (
            file_sha256(run_dir / transcript_entry["audio"])
            != transcript_entry["audio_sha256"]
        ):
            raise ValueError(f"{transcript_entry['audio']} changed after transcription")
        else:
            pass
        evidence[(transcript_entry["sample_id"], transcript_entry["variant"])] = {
            "transcript": transcript_entry["transcript"],
            "timestamp_source": "asr_aligned",
            "duration_s": transcript_entry["duration_s"],
            "source_sha256": transcript_entry["audio_sha256"],
        }
    return evidence, {
        "path": str(transcripts_dir.resolve()),
        "sha256": file_sha256(transcripts_dir / "transcripts.json"),
        "asr": transcripts["asr"],
    }
