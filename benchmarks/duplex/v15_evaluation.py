# SPDX-License-Identifier: Apache-2.0
"""Score recorded Full-Duplex-Bench v1.5 paired runs offline."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

import httpx
import soundfile
from pydantic import JsonValue

from benchmarks.duplex import v15_behavior, v15_scoring
from benchmarks.duplex.artifacts import source_fingerprint
from benchmarks.duplex.v15_audio import write_json

Timeline = Literal["simulated_playout", "media"]

# Note (wenyao): Dataset subset names differ from the scorer's category names.
SUBSET_CATEGORIES = {
    "user_interruption": "interruption",
    "user_backchannel": "backchannel",
    "talking_to_other": "talking_to_other",
    "background_speech": "background_speech",
}
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
EVALUATION_FILES = (
    "benchmarks/duplex/v15_audio.py",
    "benchmarks/eval/benchmark_duplex_v15.py",
    "benchmarks/duplex/v15_scoring.py",
    "benchmarks/duplex/v15_behavior.py",
    "benchmarks/duplex/v15_evaluation.py",
)


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


# TODO: move these utils to a separate file
def load_run(run_dir: Path) -> tuple[dict[str, JsonValue], dict[str, JsonValue], str]:
    """Return the run manifest, run.json and the manifest digest that pins them."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") not in RUN_KINDS:
        raise ValueError(f"{run_dir} is not a Full-Duplex-Bench run directory")
    return (
        manifest,
        json.loads((run_dir / "run.json").read_text()),
        file_sha256(manifest_path),
    )


def create_output(output: Path, run_dir: Path) -> None:
    """Create a fresh directory outside the recorded run, which stays immutable."""
    if output.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError(f"output {output} must be outside the run directory {run_dir}")
    output.mkdir(parents=True, exist_ok=False)


def accounting(
    manifest: dict[str, JsonValue], run: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """Declared, available and selected counts; every non-pass variant is listed."""
    declared = manifest["dataset"]["inventory"]["declared"]
    observed = manifest["dataset"]["inventory"]["observed"]
    samples = run["samples"]
    variants = [
        (sample["id"], name, state)
        for sample in samples
        for name, state in sample["variants"].items()
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
            state["status"] not in ("pending", "invalid") for _, _, state in variants
        ),
        "variant_status": dict(Counter(state["status"] for _, _, state in variants)),
        "qualified_pairs": sum(
            all(state["qualified"] for state in sample["variants"].values())
            for sample in samples
        ),
        "failures": [
            {
                "sample_id": sample_id,
                "variant": name,
                "status": state["status"],
                "errors": state["errors"],
                "violations": state["violations"],
            }
            for sample_id, name, state in variants
            if state["status"] != "pass"
        ],
    }


def load_output_transcripts(
    transcripts_dir: Path, run_dir: Path, manifest_sha256: str, timeline: Timeline
) -> tuple[dict[tuple[str, str], dict[str, JsonValue]], dict[str, JsonValue]]:
    """Map (sample_id, variant) to ASR evidence after checking it matches this run."""
    document = json.loads((transcripts_dir / "transcripts.json").read_text())
    if document["run"]["manifest_sha256"] != manifest_sha256:
        raise ValueError(f"{transcripts_dir} transcribes a different run")
    elif document["timeline"] != timeline:
        raise ValueError(
            f"{transcripts_dir} uses timeline {document['timeline']}, not {timeline}"
        )
    evidence = {}
    for entry in document["variants"]:
        if entry["status"] != "transcribed":
            continue
        if file_sha256(run_dir / entry["audio"]) != entry["audio_sha256"]:
            raise ValueError(f"{entry['audio']} changed after transcription")
        evidence[(entry["sample_id"], entry["variant"])] = {
            "transcript": entry["transcript"],
            "timestamp_source": "asr_aligned",
            "duration_s": entry["duration_s"],
            "source_sha256": entry["audio_sha256"],
        }
    return evidence, {
        "path": str(transcripts_dir.resolve()),
        "sha256": file_sha256(transcripts_dir / "transcripts.json"),
        "asr": document["asr"],
    }


def observed_end_s(variant_dir: Path, timeline: Timeline) -> float:
    """Length of the observed output window on the chosen timeline."""
    playout = json.loads((variant_dir / "playout.json").read_text())
    if timeline == "media":
        return playout["media_samples"] / playout["sample_rate"]
    else:
        origin_s, last_s = None, 0.0
        with (variant_dir / "continuous.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                last_s = max(last_s, record["time_s"])
                if (
                    origin_s is None
                    and record["direction"] == "send"
                    and record["event"].get("type") == "input_audio_buffer.append"
                ):
                    origin_s = record["time_s"]
        # Note (wenyao): Queued playout can outlast the last received event.
        return max(
            last_s - origin_s, playout["playout_samples"] / playout["sample_rate"]
        )


def score_run(
    run_dir: Path,
    output: Path,
    *,
    timeline: Timeline,
    dataset_root: Path | None = None,
    segments_path: Path | None = None,
    transcripts_dir: Path | None = None,
    judgements_path: Path | None = None,
    judge: dict[str, str | float] | None = None,
) -> dict[str, JsonValue]:
    """Score timing for qualified variants and behavior for judged paired transcripts."""
    manifest, run, manifest_sha256 = load_run(run_dir)
    if dataset_root is None:
        dataset_root = Path(manifest["dataset"]["root"])
    supplied = None
    if segments_path is not None:
        supplied = json.loads(segments_path.read_text())["segments"]
        supplied_sha256 = file_sha256(segments_path)
    asr_evidence, asr_provenance = {}, None
    if transcripts_dir is not None:
        asr_evidence, asr_provenance = load_output_transcripts(
            transcripts_dir, run_dir, manifest_sha256, timeline
        )
    judgements = []
    if judgements_path is not None:
        judgements.extend(v15_behavior.load_offline_judgements(judgements_path))
    create_output(output, run_dir)

    segment_cache: dict[str, dict[str, JsonValue]] = {}

    def speech_segments(relative: str) -> dict[str, JsonValue]:
        if relative in segment_cache:
            return segment_cache[relative]
        elif supplied is not None:
            if relative not in supplied:
                raise ValueError(f"{segments_path} has no segments for {relative}")
            entry = {
                "segments": supplied[relative]["segments"],
                "source": {
                    "kind": "supplied_segments",
                    "file": str(segments_path.resolve()),
                    "file_sha256": supplied_sha256,
                    "entry_source": supplied[relative]["source"],
                },
            }
        else:
            detected = v15_scoring.silero_speech_segments(run_dir / relative)
            entry = {
                "segments": detected["segments"],
                "source": {
                    "kind": "silero_vad",
                    "audio_sha256": file_sha256(run_dir / relative),
                    "vad": detected["vad"],
                },
            }
        segment_cache[relative] = entry
        return entry

    selected = {}
    records = {"overlap": [], "clean": []}
    behavior_inputs = []
    samples = []
    for sample in run["samples"]:
        category = SUBSET_CATEGORIES[sample["subset"]]
        selected[sample["id"]] = category
        span = sample["event_span_s"]
        is_valid = span is not None and not sample["errors"]
        row = {
            "sample_id": sample["id"],
            "category": category,
            "event": span,
            "errors": sample["errors"],
            "variants": {},
        }
        samples.append(row)
        for variant, state in sample["variants"].items():
            if not is_valid or not state["qualified"]:
                reason = (
                    "invalid_sample"
                    if not is_valid
                    else f"not_qualified:{state['status']}"
                )
                row["variants"][variant] = {"timing": None, "unscored_reason": reason}
                continue
            input_wav = f"{state['directory']}/input.wav"
            output_wav = f"{state['directory']}/{TIMELINES[timeline]['audio']}"
            inputs, outputs = speech_segments(input_wav), speech_segments(output_wav)
            output_info = soundfile.info(str(run_dir / output_wav))
            record = v15_scoring.score_event_timing(
                sample_id=sample["id"],
                category=category,
                input_segments=inputs["segments"],
                output_segments=outputs["segments"],
                event_start_s=span[0],
                event_end_s=span[1],
                input_duration_s=state["input"]["duration_s"],
                observed_end_s=observed_end_s(run_dir / state["directory"], timeline),
                output_duration_s=output_info.frames / output_info.samplerate,
                protocol_valid=True,
                timeline=timeline,
                # Note (wenyao): Speech at WAV EOF is censored even after protocol completion.
                observation_complete=True,
                evaluation="overlap" if variant == "overlap" else "clean_reference",
                segment_source={
                    "input": {"audio": input_wav, **inputs["source"]},
                    "output": {"audio": output_wav, **outputs["source"]},
                },
            )
            records[variant].append(record)
            row["variants"][variant] = {"timing": record, "unscored_reason": None}

        variant_statuses = [state["status"] for state in sample["variants"].values()]
        if not is_valid or not all(
            state["qualified"] for state in sample["variants"].values()
        ):
            behavior = {
                "status": "unscorable",
                "sample_id": sample["id"],
                "category": category,
                "reason": (
                    "invalid_sample"
                    if not is_valid
                    else f"pair_not_qualified:{'/'.join(variant_statuses)}"
                ),
            }
        else:
            transcripts = {
                "noisy_output": asr_evidence.get((sample["id"], "overlap")),
                "clean_output": asr_evidence.get((sample["id"], "clean")),
            }
            for role, key, audio in (
                ("noisy_input", "input_transcript", "input"),
                ("clean_input", "clean_input_transcript", "clean_input"),
            ):
                transcripts[role] = None
                if key in sample["paths"]:
                    path = dataset_root / sample["paths"][key]
                    if file_sha256(path) != sample["sha256"][key]:
                        raise ValueError(f"{path} changed since recording")
                    transcripts[role] = {
                        "transcript": json.loads(path.read_text(encoding="utf-8")),
                        "timestamp_source": "provided_aligned",
                        "duration_s": sample["audio"][audio]["duration_s"],
                        "source_sha256": sample["sha256"][audio],
                    }
            behavior = v15_behavior.build_behavior_input(
                sample_id=sample["id"],
                category=category,
                event_start_s=span[0],
                event_end_s=span[1],
                transcripts=transcripts,
                metadata=sample["metadata"],
            )
        behavior_inputs.append(behavior)
        row["behavior_input"] = {
            key: behavior.get(key) for key in ("status", "reason", "input_hash")
        }

    (output / "judge-inputs.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in behavior_inputs)
    )
    write_json(
        output / "segments.json",
        {"schema_version": 1, "segments": dict(sorted(segment_cache.items()))},
    )
    judge_provenance = None
    if judge is not None:
        with httpx.Client() as client:
            called = [
                v15_behavior.call_openai_compatible_judge(item, client=client, **judge)
                for item in behavior_inputs
                if item["status"] == "ready"
            ]
        (output / "judgements.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in called)
        )
        judgements.extend(called)
        judge_provenance = {
            key: judge[key] for key in ("base_url", "model", "api_key_env")
        }

    source = source_fingerprint()
    repo = Path(__file__).resolve().parents[2]
    source["files_sha256"].update(
        {path: file_sha256(repo / path) for path in EVALUATION_FILES}
    )
    result = {
        "schema_version": 1,
        "kind": "fdb-v15-score",
        "source": source,
        "run": {"path": str(run_dir.resolve()), "manifest_sha256": manifest_sha256},
        "dataset_root": str(dataset_root.resolve()),
        "accounting": accounting(manifest, run),
        "timeline": {"name": timeline, **TIMELINES[timeline]},
        "timing": v15_scoring.summarize_timing(
            records["overlap"], records["clean"], selected
        ),
        "behavior": {
            "rubric": v15_behavior.BEHAVIOR_RUBRIC,
            "input_status": dict(Counter(item["status"] for item in behavior_inputs)),
            "asr": asr_provenance,
            "judgements": {
                "offline_file": (
                    str(judgements_path.resolve()) if judgements_path else None
                ),
                "offline_file_sha256": (
                    file_sha256(judgements_path) if judgements_path else None
                ),
                "judge": judge_provenance,
            },
            "summary": v15_behavior.summarize_behavior(
                behavior_inputs, judgements, selected
            ),
        },
        "samples": samples,
    }
    write_json(output / "score.json", result)
    return result
