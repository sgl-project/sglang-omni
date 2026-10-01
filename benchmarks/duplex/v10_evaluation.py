# SPDX-License-Identifier: Apache-2.0
"""Score recorded Full-Duplex-Bench v1.0 runs offline from their ASR transcripts."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import JsonValue

import benchmarks.duplex.v10_scoring as v10_scoring
from benchmarks.duplex.artifacts import source_fingerprint
from benchmarks.duplex.run_artifacts import (
    TIMELINES,
    Timeline,
    accounting,
    create_output,
    file_sha256,
    load_output_transcripts,
    load_run,
)
from benchmarks.duplex.v10_dataset import Task
from benchmarks.duplex.v10_scoring import TASKS
from benchmarks.duplex.v15_audio import write_json

RUN_KIND = "full-duplex-bench-v1.0"
VARIANT = "input"
EVALUATION_FILES = (
    "benchmarks/duplex/v10_dataset.py",
    "benchmarks/duplex/v10_scoring.py",
    "benchmarks/duplex/v10_evaluation.py",
    "benchmarks/duplex/run_artifacts.py",
)


def unscored(sample: dict[str, JsonValue], reason: str) -> dict[str, JsonValue]:
    return {
        "sample_id": sample["id"],
        "task": sample["task"],
        "scoring": None,
        "unscored_reason": reason,
    }


def score_run(
    run_dir: Path,
    output: Path,
    *,
    transcripts_dir: Path,
    timeline: Timeline = "simulated_playout",
    backchannel_reference: Path | None = None,
) -> dict[str, JsonValue]:
    """Score qualified transcripts; absent human references leave timing unscored."""
    manifest, run, manifest_sha256 = load_run(run_dir)
    if manifest["kind"] != RUN_KIND:
        raise ValueError(f"{run_dir} is not a v1.0 run directory")
    else:
        pass
    transcripts, asr_provenance = load_output_transcripts(
        transcripts_dir, run_dir, manifest_sha256, timeline
    )
    reference = None
    if backchannel_reference is not None:
        reference = json.loads(backchannel_reference.read_text(encoding="utf-8"))
    else:
        pass
    create_output(output, run_dir)

    segment_cache: dict[Path, dict[str, JsonValue]] = {}

    def speech_segments(output_wav: Path) -> dict[str, JsonValue]:
        if output_wav not in segment_cache:
            detected = v10_scoring.silero_speech_segments(output_wav)
            segment_cache[output_wav] = {
                "audio": str(output_wav.relative_to(run_dir)),
                "audio_sha256": file_sha256(output_wav),
                "segments": detected["segments"],
                "vad": detected["vad"],
            }
        else:
            pass
        return segment_cache[output_wav]

    selected: dict[str, Task] = {}
    score_records = []
    sample_scores = []
    for sample in run["samples"]:
        task: Task = sample["task"]
        variant_state = sample["variants"][VARIANT]
        assert task in TASKS
        selected[sample["id"]] = task
        if sample["errors"]:
            sample_scores.append(unscored(sample, "invalid_sample"))
            continue
        else:
            pass
        if not variant_state["qualified"]:
            sample_scores.append(
                unscored(sample, f"not_qualified:{variant_state['status']}")
            )
            continue
        else:
            pass
        transcript_evidence = transcripts.get((sample["id"], VARIANT))
        if transcript_evidence is None:
            sample_scores.append(unscored(sample, "missing_transcript"))
            continue
        else:
            pass
        chunks = transcript_evidence["transcript"]["chunks"]
        output_wav = run_dir / variant_state["directory"] / TIMELINES[timeline]["audio"]
        if task == "backchannel":
            output_vad = speech_segments(output_wav)
            score_record = v10_scoring.score_backchannel(
                sample_id=sample["id"],
                chunks=chunks,
                output_segments=output_vad["segments"],
                input_duration_s=variant_state["input"]["duration_s"],
                reference=(
                    reference.get(sample["id"].rpartition("/")[2])
                    if reference is not None
                    else None
                ),
            )
            score_record["output_vad"] = output_vad
        elif task == "pause_handling":
            output_vad = speech_segments(output_wav)
            score_record = v10_scoring.score_pause_handling(
                sample_id=sample["id"],
                chunks=chunks,
                input_duration_s=variant_state["input"]["duration_s"],
                output_segments=output_vad["segments"],
            )
            score_record["output_vad"] = output_vad
        elif task == "turn_taking":
            output_vad = speech_segments(output_wav)
            score_record = v10_scoring.score_turn_taking(
                sample_id=sample["id"],
                chunks=chunks,
                turn_end_s=sample["events"][0][0],
                input_duration_s=variant_state["input"]["duration_s"],
                output_segments=output_vad["segments"],
            )
            score_record["output_vad"] = output_vad
        else:
            interruption_start_s, interruption_end_s = sample["events"][0]
            output_vad = speech_segments(output_wav)
            score_record = v10_scoring.score_interruption(
                sample_id=sample["id"],
                chunks=chunks,
                interruption_start_s=interruption_start_s,
                interruption_end_s=interruption_end_s,
                input_duration_s=variant_state["input"]["duration_s"],
                output_segments=output_vad["segments"],
            )
            score_record["output_vad"] = output_vad
        score_records.append(score_record)
        sample_scores.append(
            {
                "sample_id": sample["id"],
                "task": task,
                "scoring": score_record,
                "unscored_reason": None,
            }
        )

    source = source_fingerprint()
    repository_root = Path(__file__).resolve().parents[2]
    source["files_sha256"].update(
        {path: file_sha256(repository_root / path) for path in EVALUATION_FILES}
    )
    result = {
        "schema_version": 1,
        "kind": "fdb-v10-score",
        "source": source,
        "run": {"path": str(run_dir.resolve()), "manifest_sha256": manifest_sha256},
        "accounting": accounting(manifest, run),
        "timeline": {"name": timeline, **TIMELINES[timeline]},
        "asr": asr_provenance,
        "backchannel_reference": (
            {
                "path": str(backchannel_reference.resolve()),
                "sha256": file_sha256(backchannel_reference),
            }
            if backchannel_reference is not None
            else None
        ),
        "scoring": v10_scoring.summarize(score_records, selected),
        "samples": sample_scores,
    }
    write_json(output / "score.json", result)
    return result
