# SPDX-License-Identifier: Apache-2.0
"""Score recorded Full-Duplex-Bench v1.0 runs offline from their ASR transcripts."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import JsonValue

from benchmarks.duplex import v10_scoring, v15_scoring
from benchmarks.duplex.artifacts import source_fingerprint
from benchmarks.duplex.v10_dataset import Task
from benchmarks.duplex.v10_scoring import TASKS
from benchmarks.duplex.v15_audio import write_json
from benchmarks.duplex.v15_evaluation import (
    TIMELINES,
    Timeline,
    accounting,
    create_output,
    file_sha256,
    load_output_transcripts,
    load_run,
)

RUN_KIND = "full-duplex-bench-v1.0"
VARIANT = "input"
EVALUATION_FILES = (
    "benchmarks/duplex/v10_dataset.py",
    "benchmarks/duplex/v10_scoring.py",
    "benchmarks/duplex/v10_evaluation.py",
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
    """Score every qualified, transcribed sample; list why the rest are not.

    backchannel_reference is the upstream human timing distribution JSON, keyed by
    sample name; without it backchannel timing is left unscored.
    """
    manifest, run, manifest_sha256 = load_run(run_dir)
    if manifest["kind"] != RUN_KIND:
        raise ValueError(f"{run_dir} is not a v1.0 run directory")
    transcripts, asr_provenance = load_output_transcripts(
        transcripts_dir, run_dir, manifest_sha256, timeline
    )
    reference = None
    if backchannel_reference is not None:
        reference = json.loads(backchannel_reference.read_text(encoding="utf-8"))
    create_output(output, run_dir)

    segment_cache: dict[Path, dict[str, JsonValue]] = {}

    def onset_vad(output_wav: Path) -> dict[str, JsonValue]:
        if output_wav not in segment_cache:
            detected = v15_scoring.silero_speech_segments(output_wav)
            segment_cache[output_wav] = {
                "audio": str(output_wav.relative_to(run_dir)),
                "audio_sha256": file_sha256(output_wav),
                "segments": detected["segments"],
                "vad": detected["vad"],
            }
        return segment_cache[output_wav]

    selected: dict[str, Task] = {}
    records = []
    rows = []
    for sample in run["samples"]:
        task: Task = sample["task"]
        state = sample["variants"][VARIANT]
        assert task in TASKS
        selected[sample["id"]] = task
        if sample["errors"]:
            rows.append(unscored(sample, "invalid_sample"))
            continue
        if not state["qualified"]:
            rows.append(unscored(sample, f"not_qualified:{state['status']}"))
            continue
        evidence = transcripts.get((sample["id"], VARIANT))
        if evidence is None:
            rows.append(unscored(sample, "missing_transcript"))
            continue
        chunks = evidence["transcript"]["chunks"]
        output_wav = run_dir / state["directory"] / TIMELINES[timeline]["audio"]
        if task == "backchannel":
            vad = onset_vad(output_wav)
            record = v10_scoring.score_backchannel(
                sample_id=sample["id"],
                chunks=chunks,
                output_segments=vad["segments"],
                input_duration_s=state["input"]["duration_s"],
                reference=(
                    reference.get(sample["id"].rpartition("/")[2])
                    if reference is not None
                    else None
                ),
            )
            record["output_vad"] = vad
        elif task == "pause_handling":
            record = v10_scoring.score_pause_handling(
                sample_id=sample["id"],
                chunks=chunks,
                input_duration_s=state["input"]["duration_s"],
            )
        elif task == "turn_taking":
            # Note (Jeffro): The annotated span starts where the user turn ends.
            event = sample["events"][0]
            record = v10_scoring.score_response(
                sample_id=sample["id"], task=task, chunks=chunks, event_end_s=event[0]
            )
        else:
            event = sample["events"][0]
            vad = onset_vad(output_wav)
            speaking = any(start <= event[0] < end for start, end in vad["segments"])
            record = v10_scoring.score_response(
                sample_id=sample["id"],
                task=task,
                chunks=chunks,
                event_end_s=event[1],
                speaking_at_onset=speaking,
            )
            record["onset_vad"] = vad
        records.append(record)
        rows.append(
            {
                "sample_id": sample["id"],
                "task": task,
                "scoring": record,
                "unscored_reason": None,
            }
        )

    source = source_fingerprint()
    repo = Path(__file__).resolve().parents[2]
    source["files_sha256"].update(
        {path: file_sha256(repo / path) for path in EVALUATION_FILES}
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
        "scoring": v10_scoring.summarize(records, selected),
        "samples": rows,
    }
    write_json(output / "score.json", result)
    return result
